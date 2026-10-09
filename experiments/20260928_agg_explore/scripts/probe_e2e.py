"""Probe 5: 端到端对比"学生 4 个 block 该拿什么当目标"。

前面几个探针都是在冻结教师的特征上量几何量, 只能说明"目标在教师空间里怎么动",
说明不了"学生能不能拟合出来"。真正决定检出的是后者, 只有端到端能测。

四个变体, 除了"目标怎么来"以外全部相同: OCBE 吃全部 12 层教师(已定), 学生 4 个 block,
同一个 seed, 同一份数据, 同一套超参(照 baseline: k=5 code_dim=64 ep15 bs64 img256 lr5e-4)。
反相对齐保持: 最浅的目标由学生最后一个 block 还原。

  raw12     目标 = 12 个原始教师层      -> 4 个 block, 每个 block 挂 3 个头
  rep4      目标 = 4 个原始层 [2,5,8,11] -> 4 个 block, 每个 1 个头   (只减目标数, 不合并)
  mean4     目标 = 4 个组均值(LN后平均)  -> 4 个 block, 每个 1 个头   (组会方案: 合并)
  rawmean4  目标 = 4 个组均值(不LN)      -> 4 个 block, 每个 1 个头   (合并, 但不做逐层归一化)

判读:
  mean4 ≈ rep4  -> 合并本身没影响, 差别只在"目标变少"(那 raw12 该赢)
  mean4 < rep4  -> 平均确实在伤
  mean4 > rep4  -> 合并起了别的好作用(比如降噪), 要改结论

评测口径照抄 eval.py(逐场景 concat 的 global AUC, headline=topk), 保证同一把尺子。

Usage:  python probe_e2e.py --gpu 0 --variant mean4 --scenes railway --epochs 15
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
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

from model import (anomaly_readouts, build_appearance_teacher, build_bottleneck, build_student,
                   topk_loss)
from model.student import Block

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
REPS = [2, 5, 8, 11]
OLD_TAPS = [5, 7, 9, 11]
VARIANTS = ['raw12', 'rep4', 'mean4', 'rawmean4', 'old4tap', 'rep4_ocbe4', 'mean4_ocbe4',
            'base12']
# 反相对齐: 目标 i 由哪个 block 出。最浅的目标 -> 最后一个 block
RECON_AT = {'raw12': [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0], 'rep4': [3, 2, 1, 0],
            'mean4': [3, 2, 1, 0], 'rawmean4': [3, 2, 1, 0], 'old4tap': [3, 2, 1, 0], 'rep4_ocbe4': [3, 2, 1, 0], 'mean4_ocbe4': [3, 2, 1, 0],
            # base12 = 你们基线的原样复刻(12 层学生, taps [5,7,9,11]):
            # 最浅的 L5 由最后一个 block(11) 还原 -> 头 i 落在 block [11,8,5,2]
            'base12': [11, 8, 5, 2]}


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


class MultiHeadDecoder(nn.Module):
    """4 个 block, N 个头; 头 i 在 recon_at[i] 指定的 block 上输出(一个 block 可以挂多个头)。"""

    def __init__(self, n_taps, recon_at, code_dim=64, embed_dim=768, depth=4, heads=8):
        super().__init__()
        self.proj_in = nn.Linear(code_dim, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.heads = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_taps)])
        self.recon_at = list(recon_at)

    def forward(self, code):
        x = self.proj_in(code)
        outs = [None] * len(self.heads)
        by_block = {}
        for hi, bi in enumerate(self.recon_at):
            by_block.setdefault(bi, []).append(hi)
        last = max(by_block)
        for bi, blk in enumerate(self.blocks):
            x = blk(x)
            for hi in by_block.get(bi, []):
                outs[hi] = self.heads[hi](x)
            if bi >= last:
                break
        return outs


def ocbe_input(variant, taps, tgt):
    """OCBE 吃什么: 同源的变体直接吃学生要还原的那几个对象, 其余吃全 12 层教师输出。"""
    if variant in ('old4tap', 'base12'):
        return [taps[i] for i in OLD_TAPS]
    if variant == 'rep4_ocbe4':
        return [taps[i] for i in REPS]
    if variant == 'mean4_ocbe4':
        return tgt
    return list(taps)


def make_targets(variant, taps, ln):
    """taps: 12 x (B,P,768) 原始教师输出(shallow->deep)。返回按目标顺序排好的列表。"""
    if variant == 'raw12':
        return list(taps)
    if variant in ('rep4', 'rep4_ocbe4'):
        return [taps[i] for i in REPS]
    if variant in ('old4tap', 'base12'):
        return [taps[i] for i in OLD_TAPS]
    normed = [ln(t) for t in taps]
    if variant in ('mean4', 'mean4_ocbe4'):
        return [torch.stack([normed[i] for i in g]).mean(0) for g in GROUPS]
    if variant == 'rawmean4':
        return [torch.stack([taps[i] for i in g]).mean(0) for g in GROUPS]
    raise ValueError(variant)


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def forward_variant(teacher, ocbe, student, ln, variant, xb):
    taps = teacher(xb)['patches']
    tgt = make_targets(variant, taps, ln)
    code = ocbe(ocbe_input(variant, taps, tgt))
    return tgt, student(code)


def group_balanced_topk_loss(teacher_taps, student_recons, k=5, alpha=0.5):
    """Top-k per head, then balance the three heads owned by each student block.

    The ordinary loss averages all twelve heads, so a shared four-block state can
    improve the easiest member of a group while leaving another member behind.
    This loss keeps the same per-patch/top-k objective and adds a bounded penalty
    for the within-group worst head. It changes training only; test-time readout
    remains the ordinary mean over all heads for a clean comparison.
    """
    per = []
    for A, Ahat in zip(teacher_taps, student_recons):
        d = 1.0 - torch.nn.functional.cosine_similarity(A, Ahat, dim=-1)
        per.append(d.topk(k, dim=-1).values.mean(-1))  # (B,)
    losses = torch.stack(per, dim=1)  # (B, 12)
    groups = losses.reshape(losses.shape[0], 4, 3)
    mean = groups.mean(dim=-1)
    worst = groups.max(dim=-1).values
    return (mean + alpha * (worst - mean)).mean()


@torch.no_grad()
def score_video(teacher, ocbe, student, ln, variant, vid_dir, tf, dev, k, bs):
    frames = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')), key=frame_idx)
    scores = []
    for i in range(0, len(frames), bs):
        xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in frames[i:i + bs]]).to(dev)
        tgt, rec = forward_variant(teacher, ocbe, student, ln, variant, xb)
        scores.append(anomaly_readouts(tgt, rec, k=k)['topk'].detach().cpu().numpy())
    return np.concatenate(scores), len(frames)


def evaluate(scene_key, teacher, ocbe, student, ln, variant, dev, k, bs, img):
    scene_dir = os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key))
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(),
                    T.Normalize(teacher.img_mean, teacher.img_std)])
    vids = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'test', '*'))
                  if os.path.isdir(d))
    per_s, per_y = [], []
    for vp in vids:
        s, n = score_video(teacher, ocbe, student, ln, variant, vp, tf, dev, k, bs)
        per_s.append(s)
        lab = np.load(vp + '.npy', allow_pickle=True)
        per_y.append(np.asarray(lab).astype(int).ravel()[-n:])
    y = np.concatenate(per_y)
    return roc_auc_score(y, np.concatenate(per_s)) * 100, int(y.sum()), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--variant', required=True, choices=VARIANTS)
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
    ap.add_argument('--loss_mode', choices=['mean', 'group_balanced'], default='mean',
                    help='raw12 training loss; evaluation always uses ordinary top-k mean')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')

    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    n_in = {'old4tap': 4, 'rep4_ocbe4': 4, 'mean4_ocbe4': 4, 'base12': 4}.get(a.variant, 12)
    ocbe = build_bottleneck(n_taps=n_in, code_dim=a.code_dim,
                           use_tap_norm=True).to(dev)
    n_taps = 12 if a.variant == 'raw12' else 4
    depth = 12 if a.variant == 'base12' else a.depth      # base12 = 复刻基线, 学生也是 12 层
    student = MultiHeadDecoder(n_taps, RECON_AT[a.variant], code_dim=a.code_dim,
                               embed_dim=teacher.embed_dim, depth=depth).to(dev)
    ln = nn.LayerNorm(teacher.embed_dim).to(dev).eval()          # 冻结: 不进优化器
    params = list(ocbe.parameters()) + list(student.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n_par = sum(p.numel() for p in params) / 1e6
    run_name = a.variant if a.loss_mode == 'mean' else f'{a.variant}_{a.loss_mode}'
    print(f'[{run_name}] targets={n_taps} heads, student={a.depth} blocks, OCBE eats 12 layers, '
          f'trainable={n_par:.1f}M, seed={a.seed}, loss={a.loss_mode}', flush=True)

    for scene_key in a.scenes:
        scene_dir = os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key))
        ds = FrameDataset(scene_dir, a.img, teacher.img_mean, teacher.img_std)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(a.seed))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                    steps_per_epoch=len(dl), epochs=a.epochs)
        print(f'[{scene_key}] {len(ds)} frames, {len(dl)} batches/epoch', flush=True)
        t0 = time.time()
        for ep in range(a.epochs):
            ocbe.train(); student.train()
            tot = n = 0
            for xb in dl:
                xb = xb.to(dev, non_blocking=True)
                tgt, recons = forward_variant(teacher, ocbe, student, ln, a.variant, xb)
                if a.loss_mode == 'group_balanced':
                    if a.variant != 'raw12':
                        raise ValueError('group_balanced is currently defined for raw12 only')
                    loss = group_balanced_topk_loss(tgt, recons, k=a.k)
                else:
                    loss = topk_loss(tgt, recons, k=a.k)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * xb.size(0); n += xb.size(0)
            print(f'  ep{ep + 1:02d}/{a.epochs} loss={tot / max(n, 1):.4f}  ({time.time() - t0:.0f}s)',
                  flush=True)
        ocbe.eval(); student.eval()
        auc, npos, ntot = evaluate(scene_key, teacher, ocbe, student, ln, a.variant, dev, a.k,
                                   32, a.img)
        print(f'>>> [{run_name}] {scene_key}: AUC={auc:.2f}  ({npos}/{ntot} 异常帧)  '
              f'总耗时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
