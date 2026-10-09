"""Probe 8: 固定住"不塌的 loss", 换聚合结构来比 —— 相邻三层到底怎么聚。

loss 固定为 v2(见 probe_agg_learn.py 的说明), 它对所有结构一视同仁:
    L = d( g(x_t), g(x_{t+1}) )                        <- 正常相邻两帧之间要"别动"
        + lam * relu( margin - d(g(x_t), g(x_anom)) )  <- 碰到伪异常要"动", 够用就行
        + dev * || g - mean ||^2 / ||mean||^2          <- 别离均值走太远
(v1 用 -lam*d(异常) 直接最大化, 无上界 -> 训久了跑到"输出常数"或"跟正常帧反着来"两种退化解。)

比的聚合结构(全部设计成"起点严格等于均值", 差别只在能往哪走):
  mean       三个层直接取平均                        (参照, 不训练)
  mlp        均值 + MLP(concat 三层)                 (组会提的 concat+MLP)
  attn       跨层注意力: 每个 patch 独立地在三层间加权求和, 分数由内容算出
  attn_mlp   跨层注意力 + 一个残差 MLP(可以超出"三层凸组合"这个范围)
  conv       均值 + patch 网格上的 1x1 -> 3x3 深度可分离 -> 1x1

另外报告 attn 系列的注意力分布(标准差/熵), 用来检查它是不是又塌成了均匀平均
(LAA 的 w 和 DLBM v008 的层注意力都塌过)。

Usage:  python probe_agg_kinds.py --gpu 1 --scenes railway bike --frames 96 --epochs 30
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
KINDS = ('n', 'b', 'p')
AGGS = ['mean', 'mlp', 'attn', 'attn_mlp', 'conv']


def load_pairs(scene_dir, img, mean, std, limit):
    paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
    if not paths:
        raise FileNotFoundError(f'no train .jpg under {scene_dir}')
    pairs = [(p, q) for p, q in zip(paths, paths[1:]) if os.path.dirname(p) == os.path.dirname(q)]
    step = max(1, len(pairs) // limit)
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])
    return tf, pairs[::step][:limit]


def cutpaste(x, size, gen):
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


def _zero_last(layer):
    if isinstance(layer, nn.Linear):
        nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Conv2d):
        nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)


class Base(nn.Module):
    """子类实现 residual(members, mean) -> (N,P,D); 输出统一是 LN(mean + residual), 起点=均值。"""

    def __init__(self, embed_dim):
        super().__init__()
        self.out_norm = nn.LayerNorm(embed_dim)

    def residual(self, members, mean):
        raise NotImplementedError

    def forward(self, members, want_attn=False):
        """members: (n, N, P, D) 已逐层归一化 -> (N, P, D)"""
        mean = members.mean(0)
        r = self.residual(members, mean)
        out = self.out_norm(mean + r) if r is not None else mean
        return (out, None) if not want_attn else (out, getattr(self, 'last_attn', None))


class MeanAgg(Base):
    def __init__(self, embed_dim, **kw):
        super().__init__(embed_dim)

    def residual(self, members, mean):
        return None


class MlpAgg(Base):
    def __init__(self, embed_dim, hidden=512, n=3, **kw):
        super().__init__(embed_dim)
        self.net = nn.Sequential(nn.Linear(embed_dim * n, hidden), nn.GELU(),
                                 nn.Linear(hidden, embed_dim))
        _zero_last(self.net[-1])

    def residual(self, members, mean):
        n, N, P, D = members.shape
        concat = members.permute(1, 2, 0, 3).reshape(N, P, n * D)
        return self.net(concat)


class AttnAgg(Base):
    """每个 patch 独立地在三层之间做注意力。分数由 [该层特征, 三层均值] 算出, 零初始化 -> 起点均匀。"""

    def __init__(self, embed_dim, hidden=256, n=3, with_mlp=False, **kw):
        super().__init__(embed_dim)
        self.n = n
        self.score = nn.Sequential(nn.Linear(embed_dim * 2, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))
        _zero_last(self.score[-1])
        self.mlp = None
        if with_mlp:
            self.mlp = nn.Sequential(nn.Linear(embed_dim * n, hidden), nn.GELU(),
                                     nn.Linear(hidden, embed_dim))
            _zero_last(self.mlp[-1])

    def residual(self, members, mean):
        n, N, P, D = members.shape
        feats = torch.cat([members, mean.unsqueeze(0).expand(n, N, P, D)], dim=-1)
        logits = self.score(feats).squeeze(-1)                  # (n, N, P)
        attn = logits.softmax(0)
        self.last_attn = attn
        mix = (attn.unsqueeze(-1) * members).sum(0)             # 三层的凸组合
        if self.mlp is None:
            return mix - mean                                   # residual 形式回到均值起点
        concat = members.permute(1, 2, 0, 3).reshape(N, P, n * D)
        return (mix - mean) + self.mlp(concat)


class ConvAgg(Base):
    def __init__(self, embed_dim, hidden=256, n=3, grid=16, **kw):
        super().__init__(embed_dim)
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim * n, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 1))
        _zero_last(self.net[-1])

    def residual(self, members, mean):
        n, N, P, D = members.shape
        g = self.grid
        concat = members.permute(1, 2, 0, 3).reshape(N, P, n * D)          # (N,P,nD)
        x = concat.reshape(N, g, g, n * D).permute(0, 3, 1, 2)             # (N,nD,g,g)
        y = self.net(x).permute(0, 2, 3, 1).reshape(N, P, D)
        return y


def build(kind, embed_dim, hidden, grid):
    if kind == 'mean':
        return MeanAgg(embed_dim)
    if kind == 'mlp':
        return MlpAgg(embed_dim, hidden)
    if kind == 'attn':
        return AttnAgg(embed_dim, hidden)
    if kind == 'attn_mlp':
        return AttnAgg(embed_dim, hidden, with_mlp=True)
    if kind == 'conv':
        return ConvAgg(embed_dim, hidden, grid=grid)
    raise ValueError(kind)


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
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--dev', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev).eval()
    grid = a.img // 16
    print(f'loss 固定(v2): 不动 + {a.lam}×relu({a.margin}−要动) + {a.dev}×偏离 | '
          f'结构: {" ".join(AGGS)} | grid={grid}×{grid}', flush=True)

    for scene_key in a.scenes:
        tf, pairs = load_pairs(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                               a.img, teacher.img_mean, teacher.img_std, a.frames)
        gen = torch.Generator().manual_seed(a.seed)
        cache = {k: [] for k in KINDS}
        with torch.no_grad():
            for i in range(0, len(pairs), a.bs):
                chunk = pairs[i:i + a.bs]
                xa = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in chunk]).to(dev)
                xb = torch.stack([tf(Image.open(q).convert('RGB')) for _, q in chunk]).to(dev)
                xp = cutpaste(xa, 64, gen)
                for k, xx in zip(KINDS, (xa, xb, xp)):
                    cache[k].append(torch.stack([ln(t) for t in teacher(xx)['patches']], 1))
        feat = {k: torch.cat(cache[k], 0).permute(1, 0, 2, 3) for k in KINDS}   # (12,N,P,D)
        N = feat['n'].shape[1]
        n_tr = int(N * 0.75)
        print(f'\n===== scene={scene_key}  {N} 对帧 (训练 {n_tr} / 留出 {N - n_tr}) =====')
        print('   组           结构        异常位移   正常波动   位移/波动   保真R2   备注')
        for gi, idx in enumerate(GROUPS):
            f_tr = {k: feat[k][list(idx)][:, :n_tr] for k in KINDS}
            f_va = {k: feat[k][list(idx)][:, n_tr:] for k in KINDS}
            per_gap = torch.tensor([cosd(f_va['n'][j], f_va['p'][j]).mean().item() for j in range(3)],
                                   device=dev)
            for kind in AGGS:
                torch.manual_seed(a.seed)
                agg = build(kind, teacher.embed_dim, a.hidden, grid).to(dev)
                if kind != 'mean':
                    opt = torch.optim.AdamW(agg.parameters(), lr=a.lr)
                    best, best_state = -1e9, None
                    for ep in range(a.epochs):
                        perm = torch.randperm(n_tr, device=dev)
                        for s in range(0, n_tr, 32):
                            sel = perm[s:s + 32]
                            g_n = agg(f_tr['n'][:, sel])[0]
                            mean_n = f_tr['n'][:, sel].mean(0)
                            dev_pen = ((g_n - mean_n).pow(2).sum(-1).mean()
                                       / mean_n.pow(2).sum(-1).mean())
                            L = cosd(g_n, agg(f_tr['b'][:, sel])[0]).mean() \
                                + a.lam * (a.margin - cosd(g_n, agg(f_tr['p'][:, sel])[0])
                                           ).clamp_min(0).mean() \
                                + a.dev * dev_pen
                            opt.zero_grad(); L.backward(); opt.step()
                        with torch.no_grad():
                            g_n, g_p, g_b = agg(f_va['n'])[0], agg(f_va['p'])[0], agg(f_va['b'])[0]
                            ratio = cosd(g_n, g_p).mean().item() / max(
                                cosd(g_n, g_b).mean().item(), 1e-9)
                        if ratio > best:
                            best = ratio
                            best_state = {k: v.detach().clone() for k, v in agg.state_dict().items()}
                    if best_state:
                        agg.load_state_dict(best_state)
                with torch.no_grad():
                    g_n, attn = agg(f_va['n'], want_attn=True)
                    g_p = agg(f_va['p'])[0]
                    g_b = agg(f_va['b'])[0]
                    f_tr_g = agg(f_tr['n'])[0]
                d = cosd(g_n, g_p).mean().item()
                fl = cosd(g_n, g_b).mean().item()
                r2 = []
                for j in range(3):
                    w_ = ridge(f_tr_g.reshape(-1, 768)[::3], f_tr['n'][j].reshape(-1, 768)[::3])
                    ye = f_va['n'][j].reshape(-1, 768)
                    pred = g_n.reshape(-1, 768) @ w_
                    r2.append((1 - (ye - pred).pow(2).sum() / (ye - ye.mean(0)).pow(2).sum()).item())
                note = ''
                if attn is not None:
                    a_ = attn.reshape(3, -1)
                    note = f'attn标准差={a_.std(0).mean():.4f} (均匀=0)'
                print(f'   G{gi} {idx}  {kind:<9}  {d:.4f}   {fl:.5f}   {d / max(fl, 1e-9):>7.2f}   '
                      f'{sum(r2) / 3:.3f}   {note}')
        del cache, feat
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
