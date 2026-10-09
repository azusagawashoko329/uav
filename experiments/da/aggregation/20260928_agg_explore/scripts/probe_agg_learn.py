"""Probe 6: 用一个"自己的 loss"训练聚合器 —— 相邻三层怎么聚成一个蒸馏目标。

方向: 相邻三层 -> 聚合 -> 当目标蒸馏给学生。研究问题是"怎么聚合"; 聚合器由它自己的 loss
训练, 跟 top-k 重建 loss 是两个独立的东西。

设计 L_A 的唯一硬约束（前面探针量出来的）:
  "保住三个层的信息"已经被"取平均"做满了(从均值解码回组员的 R^2 = 0.88~0.98, 几乎无损)。
  所以任何写成"别丢东西"的 loss, 训出来的聚合器只会停在均值、一步都走不动。
  => L_A 必须要求一件"均值做不到的事"。

均值做不到: 它分不清哪个成员在携带信号, 三个平摊, 把灵敏的层和迟钝的层一起拉平
(探针 2/3: 合并后只剩最好组员的 60~70%)。

这里用的 L_A (v2 —— v1 版见下方"崩在哪"):
    L = d( g(x_t), g(x_{t+1}) )                       <- 正常相邻两帧之间要"别动"
        + lam * relu( margin - d(g(x_t), g(x_anom)) ) <- 碰到伪异常要"动", 但够用就行
        + dev * || g - mean ||^2 / ||mean||^2         <- 别离均值走太远
  只用正常帧, 无标签。

v1 崩在哪(实测): v1 用 -lam*d(正常,异常) 直接最大化位移, 这一项没有上界(余弦距离最大 2.0),
  而最小化项下限是 0 => 梯度永远有得赚。32 帧 5 epoch 时比值好看(2~8 倍), 96 帧 30 epoch 就
  跑到退化解: 要么输出常数(位移->0), 要么让异常帧的输出和正常帧"反着来"(位移->2), 后者跟
  检出异常毫无关系(保真 R^2 变负)。v2 用 margin 堵死"反着来", 用 dev 限制偏离, 并在留出的帧上早停。

聚合器形状: g = 组内均值 + 小网络(concat 三层) 的修正量, 小网络末层零初始化
  => 训练起点严格等于均值, "比均值走远了没有"从零开始量, 不掺初始化运气。

评估(在没参与训练的帧上): 异常位移 / 正常波动 / 保真R2, 与 mean / 最好的组员 / 按敏感度加权 同比。

Usage:  python probe_agg_learn.py --gpu 0 --scenes railway bike --frames 96 --epochs 30
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../../../..')))
import argparse
import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from model import build_appearance_teacher

DATA_ROOT = os.environ.get('UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
KINDS = ('n', 'b', 'p')          # 当前帧 / 下一帧(正常波动) / 伪异常


def load_pairs(scene_dir, img, mean, std, limit):
    paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
    if not paths:
        raise FileNotFoundError(f'no train .jpg under {scene_dir}')
    pairs = [(p, q) for p, q in zip(paths, paths[1:]) if os.path.dirname(p) == os.path.dirname(q)]
    step = max(1, len(pairs) // limit)
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])
    return tf, pairs[::step][:limit]


def cutpaste(x, size, gen):
    """伪异常: 把图里一块贴到另一个位置(对应数据集里"新物体出现"那类)。"""
    b, _, h, w = x.shape
    out = x.clone()
    for i in range(b):
        sy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        sx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        dy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        dx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        out[i, :, dy:dy + size, dx:dx + size] = x[i, :, sy:sy + size, sx:sx + size]
    return out


def cosd(a, b):
    return 1.0 - torch.cosine_similarity(a, b, dim=-1)


class Aggregator(nn.Module):
    """g = 组内均值 + MLP(concat 三层); MLP 末层零初始化 -> 起点严格是均值。"""

    def __init__(self, embed_dim=768, hidden=512, n=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * n, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        # 输出逐 token 归一化: 幅度恒定, 堵掉"把输出压小让余弦失真"这条捷径
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, members):
        """members: (n, M, D) 已逐层归一化。"""
        mean = members.mean(0)
        concat = members.permute(1, 0, 2).reshape(mean.shape[0], -1)
        return self.out_norm(mean + self.net(concat))


def ridge(X, Y, lam_scale=1e-3):
    xtx = X.T @ X
    lam = lam_scale * xtx.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0], device=X.device, dtype=X.dtype), X.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=['railway', 'bike'])
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--margin', type=float, default=0.15, help='异常位移够这个值就不再推')
    ap.add_argument('--dev', type=float, default=0.1, help='偏离均值的惩罚权重')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--paste', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev).eval()
    print(f'L_A v2 = 不动 + {a.lam}×relu({a.margin}−要动) + {a.dev}×偏离惩罚 | '
          f'hidden={a.hidden} lr={a.lr} epochs={a.epochs} (留出帧早停)', flush=True)

    for scene_key in a.scenes:
        tf, pairs = load_pairs(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                               a.img, teacher.img_mean, teacher.img_std, a.frames)
        n_fit = max(1, len(pairs) // 2)
        gen = torch.Generator().manual_seed(a.seed)
        cache = {True: {k: [] for k in KINDS}, False: {k: [] for k in KINDS}}
        with torch.no_grad():
            for i in range(0, len(pairs), a.bs):
                chunk = pairs[i:i + a.bs]
                xa = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in chunk]).to(dev)
                xb = torch.stack([tf(Image.open(q).convert('RGB')) for _, q in chunk]).to(dev)
                xp = cutpaste(xa, a.paste, gen)
                dst = cache[i < n_fit]
                for k, xx in zip(KINDS, (xa, xb, xp)):
                    dst[k].append(torch.stack([ln(t) for t in teacher(xx)['patches']], 0))

        def flat(split):
            """-> {kind: (12, M, D)}, M = 帧数×patch 数; 以及 eval 帧数"""
            out = {k: [torch.cat([c[l] for c in cache[split][k]], 0) for l in range(12)]
                   for k in KINDS}
            n_frames = out['n'][0].shape[0]
            return {k: torch.stack([v.reshape(-1, 768) for v in out[k]]) for k in KINDS}, n_frames

        F_fit, n_fit_frames = flat(True)
        F_ev, n_ev_frames = flat(False)
        n_par = sum(p.numel() for p in Aggregator().parameters())
        print(f'\n===== scene={scene_key}  {len(pairs)} 对帧 (fit {n_fit_frames} / eval {n_ev_frames}) '
              f'| 聚合器 {n_par / 1e6:.2f}M 参数 =====')
        print('   组          聚合方式       异常位移   正常波动   位移/波动   保真R2     输出幅度')
        for gi, idx in enumerate(GROUPS):
            fit = {k: F_fit[k][list(idx)] for k in KINDS}        # (3, M, D)
            ev = {k: F_ev[k][list(idx)] for k in KINDS}
            # --- 训练聚合器(fit 帧训练, 留出帧早停; 不碰测试数据) ---
            agg = Aggregator(teacher.embed_dim, a.hidden).to(dev)
            opt = torch.optim.AdamW(agg.parameters(), lr=a.lr)
            M = fit['n'].shape[1]
            n_val = max(1024, M // 5)
            val_sel = slice(M - n_val, M)
            tr = {k: fit[k][:, :M - n_val] for k in KINDS}
            va = {k: fit[k][:, val_sel] for k in KINDS}
            best, best_state, best_ep = -1e9, None, 0
            step_bs = 4096
            Mt = tr['n'].shape[1]
            for ep in range(a.epochs):
                perm = torch.randperm(Mt, device=dev)
                for s in range(0, Mt, step_bs):
                    sel = perm[s:s + step_bs]
                    g_n = agg(tr['n'][:, sel])
                    mean_n = tr['n'][:, sel].mean(0)
                    dev_pen = (g_n - mean_n).pow(2).sum(-1).mean() / mean_n.pow(2).sum(-1).mean()
                    L = cosd(g_n, agg(tr['b'][:, sel])).mean() \
                        + a.lam * (a.margin - cosd(g_n, agg(tr['p'][:, sel]))).clamp_min(0).mean() \
                        + a.dev * dev_pen
                    opt.zero_grad(); L.backward(); opt.step()
                with torch.no_grad():                      # 越界监测 + 早停(用留出帧)
                    g_n, g_p, g_b = agg(va['n']), agg(va['p']), agg(va['b'])
                    ratio = cosd(g_n, g_p).mean().item() / max(cosd(g_n, g_b).mean().item(), 1e-9)
                    off = (g_n - va['n'].mean(0)).norm(dim=-1).mean().item() / \
                          va['n'].mean(0).norm(dim=-1).mean().item()
                if ratio > best:
                    best, best_ep = ratio, ep + 1
                    best_state = {k: v.detach().clone() for k, v in agg.state_dict().items()}
            if best_state is not None:
                agg.load_state_dict(best_state)
            # --- 在 eval 帧上量 ---
            per_gap = torch.tensor([cosd(ev['n'][j], ev['p'][j]).mean().item() for j in range(3)],
                                   device=dev)
            w_gapw = per_gap / per_gap.sum()
            with torch.no_grad():
                cands = {
                    'mean': ev['n'].mean(0),
                    'gapw': (w_gapw.view(3, 1, 1) * ev['n']).sum(0),
                    'best': ev['n'][int(per_gap.argmax())],
                    'learned': agg(ev['n']),
                }
                cands_p = {
                    'mean': ev['p'].mean(0),
                    'gapw': (w_gapw.view(3, 1, 1) * ev['p']).sum(0),
                    'best': ev['p'][int(per_gap.argmax())],
                    'learned': agg(ev['p']),
                }
                cands_b = {
                    'mean': ev['b'].mean(0),
                    'gapw': (w_gapw.view(3, 1, 1) * ev['b']).sum(0),
                    'best': ev['b'][int(per_gap.argmax())],
                    'learned': agg(ev['b']),
                }
            # 保真: 从聚合结果线性解码回每个组员(fit 上解, eval 上报)
            for name in ('mean', 'gapw', 'best', 'learned'):
                gn = cands[name]
                # 保真: 从聚合结果线性解码回每个组员 —— 映射在 fit 帧上解, 在 eval 帧上报告
                with torch.no_grad():
                    fit_g = {'mean': fit['n'].mean(0),
                             'gapw': (w_gapw.view(3, 1, 1) * fit['n']).sum(0),
                             'best': fit['n'][int(per_gap.argmax())],
                             'learned': agg(fit['n'])}[name]
                r2 = []
                for j in range(3):
                    w_ = ridge(fit_g[::3], fit['n'][j][::3])
                    pred = gn @ w_
                    ye = ev['n'][j]
                    r2.append((1 - (ye - pred).pow(2).sum() / (ye - ye.mean(0)).pow(2).sum()).item())
                d = cosd(gn, cands_p[name]).mean().item()
                fl = cosd(gn, cands_b[name]).mean().item()
                norm = gn.norm(dim=-1).mean().item()
                print(f'   G{gi} {idx}  {name:<10}  {d:.4f}   {fl:.5f}   {d / max(fl, 1e-9):>7.2f}   '
                      f'{sum(r2) / 3:.3f}   |g|={norm:.1f}')
        del cache, F_fit, F_ev
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
