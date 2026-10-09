"""Probe 7: 端到端 —— 聚合器由它自己的 loss 训练, 再当目标蒸馏给学生。

两段式, 聚合器和学生完全解耦:
  第 1 段(聚合器, 用它自己的 loss, 只用正常帧):
      L_A = d( g(x_t), g(x_{t+1}) )        <- 正常相邻两帧之间要"别动"
            - lam * d( g(x_t), g(x_anom) ) <- 碰到伪异常要"动"
      训练起点严格等于"三个层取平均"(残差形式 + 末层零初始化), 所以它比均值走远了没有,
      是从零开始量的。训完冻结。
  第 2 段(学生, 照旧用 top-k 重建 loss):
      目标 = 4 个聚合结果, OCBE 也吃这 4 个聚合结果(同源), 学生 4 个 block, 反相对齐。

对照: probe_e2e.py 里的 mean4_ocbe4 —— 同样结构、同样同源, 只是聚合换成"直接取平均"。
      两者只差"怎么聚合", 所以 AUC 之差就是聚合这一步的净效果。

评测口径照抄 eval.py(逐场景 concat 的 global AUC, headline=topk)。

Usage:  python probe_e2e_learned.py --gpu 0 --scenes railway --agg_frames 128 --agg_epochs 30
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../../../..')))
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import anomaly_readouts, build_appearance_teacher, build_bottleneck, topk_loss
from probe_agg_kinds import build as build_agg, cosd, cutpaste
from probe_e2e import GROUPS, MultiHeadDecoder, RECON_AT, SCENES, DATA_ROOT, frame_idx

RECON = [3, 2, 1, 0]                 # 反相对齐: 最浅的聚合结果 -> 最后一个学生 block


class FrameDataset(Dataset):
    def __init__(self, scene_dir, img, mean, std):
        self.paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
        if not self.paths:
            raise FileNotFoundError(f'no train .jpg under {scene_dir}')
        self.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


def train_aggregators(scene_dir, teacher, ln, dev, img, n_pairs, epochs, lam, lr, hidden, seed,
                      kind='mlp', margin=0.15, dev_pen_w=0.1, bs=8):
    """第 1 段: 在正常帧上用 L_A(v2) 训练 4 个聚合器, 返回冻结好的列表。"""
    paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
    pairs = [(p, q) for p, q in zip(paths, paths[1:]) if os.path.dirname(p) == os.path.dirname(q)]
    step = max(1, len(pairs) // n_pairs)
    pairs = pairs[::step][:n_pairs]
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    gen = torch.Generator().manual_seed(seed)
    cache = {k: [] for k in ('n', 'b', 'p')}
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            xa = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in chunk]).to(dev)
            xb = torch.stack([tf(Image.open(q).convert('RGB')) for _, q in chunk]).to(dev)
            xp = cutpaste(xa, 64, gen)
            for k, xx in zip(('n', 'b', 'p'), (xa, xb, xp)):
                cache[k].append(torch.stack([ln(t) for t in teacher(xx)['patches']], 1))
    feat = {k: torch.cat(cache[k], 0).permute(1, 0, 2, 3) for k in cache}   # (12, N, P, D)
    N = feat['n'].shape[1]
    n_tr = int(N * 0.75)
    grid = img // 16

    aggs = []
    for gi, idx in enumerate(GROUPS):
        torch.manual_seed(seed)
        agg = build_agg(kind, teacher.embed_dim, hidden, grid).to(dev)
        m_tr = {k: feat[k][list(idx)][:, :n_tr] for k in feat}
        m_va = {k: feat[k][list(idx)][:, n_tr:] for k in feat}
        if kind != 'mean':
            opt = torch.optim.AdamW(agg.parameters(), lr=lr)
            best, best_state = -1e9, None
            for ep in range(epochs):
                perm = torch.randperm(n_tr, device=dev)
                for s in range(0, n_tr, 32):
                    sel = perm[s:s + 32]
                    g_n = agg(m_tr['n'][:, sel])[0]
                    mean_n = m_tr['n'][:, sel].mean(0)
                    dp = (g_n - mean_n).pow(2).sum(-1).mean() / mean_n.pow(2).sum(-1).mean()
                    L = cosd(g_n, agg(m_tr['b'][:, sel])[0]).mean() \
                        + lam * (margin - cosd(g_n, agg(m_tr['p'][:, sel])[0])).clamp_min(0).mean() \
                        + dev_pen_w * dp
                    opt.zero_grad(); L.backward(); opt.step()
                with torch.no_grad():                     # 留出帧早停(不碰测试数据)
                    gn, gp, gb = agg(m_va['n'])[0], agg(m_va['p'])[0], agg(m_va['b'])[0]
                    ratio = cosd(gn, gp).mean().item() / max(cosd(gn, gb).mean().item(), 1e-9)
                if ratio > best:
                    best = ratio
                    best_state = {k: v.detach().clone() for k, v in agg.state_dict().items()}
            if best_state:
                agg.load_state_dict(best_state)
        for p in agg.parameters():
            p.requires_grad_(False)
        agg.eval()
        aggs.append(agg)
    del cache, feat
    torch.cuda.empty_cache()
    return aggs


def make_targets(aggs, taps, ln):
    return [aggs[g](torch.stack([ln(taps[i]) for i in GROUPS[g]], 0))[0] for g in range(4)]


def forward_all(teacher, ocbe, student, aggs, ln, xb):
    taps = teacher(xb)['patches']
    tgt = make_targets(aggs, taps, ln)
    return tgt, student(ocbe(tgt))


@torch.no_grad()
def score_video(teacher, ocbe, student, aggs, ln, vid_dir, tf, dev, k, bs):
    frames = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')), key=frame_idx)
    scores = []
    for i in range(0, len(frames), bs):
        xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in frames[i:i + bs]]).to(dev)
        tgt, rec = forward_all(teacher, ocbe, student, aggs, ln, xb)
        scores.append(anomaly_readouts(tgt, rec, k=k)['topk'].detach().cpu().numpy())
    return np.concatenate(scores), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=['railway'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--code_dim', type=int, default=64)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--agg_frames', type=int, default=128, help='第1段用来训聚合器的正常帧对数')
    ap.add_argument('--agg_epochs', type=int, default=30)
    ap.add_argument('--agg_hidden', type=int, default=512)
    ap.add_argument('--agg_lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--dev', type=float, default=0.1)
    ap.add_argument('--agg_kind', default='conv', choices=['mean', 'mlp', 'attn', 'attn_mlp', 'conv'])
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev).eval()
    print(f'[{a.agg_kind}] 聚合器先训({a.agg_frames}对帧 × {a.agg_epochs}ep, L_A=不动−{a.lam}×要动) '
          f'再冻结; 学生 {a.depth} block, OCBE 吃 4 个聚合结果', flush=True)

    for scene_key in a.scenes:
        scene_dir = os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key))
        t0 = time.time()
        aggs = train_aggregators(scene_dir, teacher, ln, dev, a.img, a.agg_frames, a.agg_epochs,
                                 a.lam, a.agg_lr, a.agg_hidden, a.seed, kind=a.agg_kind,
                                 margin=a.margin, dev_pen_w=a.dev)
        print(f'[{scene_key}] 聚合器训完 ({time.time() - t0:.0f}s)，冻结', flush=True)

        ocbe = build_bottleneck(n_taps=4, code_dim=a.code_dim, use_tap_norm=True).to(dev)
        student = MultiHeadDecoder(4, RECON, code_dim=a.code_dim, embed_dim=teacher.embed_dim,
                                   depth=a.depth).to(dev)
        opt = torch.optim.AdamW(list(ocbe.parameters()) + list(student.parameters()),
                                lr=a.lr, weight_decay=a.wd)
        ds = FrameDataset(scene_dir, a.img, teacher.img_mean, teacher.img_std)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                        drop_last=True, generator=torch.Generator().manual_seed(a.seed))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=len(dl),
                                                    epochs=a.epochs)
        print(f'[{scene_key}] {len(ds)} 帧, {len(dl)} batch/epoch', flush=True)
        for ep in range(a.epochs):
            ocbe.train(); student.train()
            tot = n = 0
            for xb in dl:
                xb = xb.to(dev, non_blocking=True)
                tgt, rec = forward_all(teacher, ocbe, student, aggs, ln, xb)
                loss = topk_loss(tgt, rec, k=a.k)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * xb.size(0); n += xb.size(0)
            print(f'  ep{ep + 1:02d}/{a.epochs} loss={tot / max(n, 1):.4f}  ({time.time() - t0:.0f}s)',
                  flush=True)
        ocbe.eval(); student.eval()
        tf = T.Compose([T.Resize((a.img, a.img)), T.ToTensor(),
                        T.Normalize(teacher.img_mean, teacher.img_std)])
        vids = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'test', '*'))
                      if os.path.isdir(d))
        per_s, per_y = [], []
        for vp in vids:
            s, n = score_video(teacher, ocbe, student, aggs, ln, vp, tf, dev, a.k, 32)
            per_s.append(s)
            lab = np.load(vp + '.npy', allow_pickle=True)
            per_y.append(np.asarray(lab).astype(int).ravel()[-n:])
        y = np.concatenate(per_y)
        auc = roc_auc_score(y, np.concatenate(per_s)) * 100
        print(f'>>> [{a.agg_kind}] {scene_key}: AUC={auc:.2f}  ({int(y.sum())}/{len(y)} 异常帧)  '
              f'总耗时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
