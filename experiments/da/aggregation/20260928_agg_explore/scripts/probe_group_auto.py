"""Probe 4: 一次比较多种"三个层怎么合并"的设计。

背景（探针 1-3）：
  * 组内三层不一样, 合并不是空操作; 均值也不丢内容。
  * 但均值让三个成员平起平坐 —— 组里混进一个"对异常没反应"的层(第0层/第11层),
    异常信号就被拉平到最好组员的 60~70%。
  * "层间差"在局部指标上能超过最好组员, 但它可能只是把所有变化都放大了。

全部冻结、不训练。权重和预测器只在 fit 帧上算, 在 eval 帧上报告, 不循环。

比较的合并方式（每组的三个层记作 t0 t1 t2）:
  mean       t0+t1+t2 再除以3                 (当前设计)
  gapw       按各层"对异常的反应强度"加权
  diff       t2 - t0                          (层间差)
  auto_gap   让数据自己算权重, 只要求"异常时动得越大越好"
  auto_ratio 让数据自己算权重, 同时要求"异常时动得大"且"正常帧之间别乱动"
  residual   用另外两层预测这一层, 取预测不掉的残差, 三个残差再合并
  best       三个层里对异常反应最强的那个 (上界参考)

四个报告指标:
  top5位移   每帧取最大的 5 个 patch 求平均 —— 这是检测器真正读的那个数
  局部位移   只算"贴上去的那块"区域
  正常波动   正常帧之间的位移 (越大说明这个合并方式对所有变化都敏感, 不只是异常)
  R2         还能不能反推回三个原层

Usage:  python probe_group_auto.py --gpu 0 --frames 96
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
PATCH = 16
AGGS = ['mean', 'gapw', 'diff', 'auto_gap', 'auto_ratio', 'residual', 'best']


def load_pairs(scene_dir, img, mean, std, limit):
    """成对的帧 (a, b), b 是同一个子视频里的下一帧 —— 用来量"正常帧之间的自身波动"。"""
    paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
    if not paths:
        raise FileNotFoundError(f'no train .jpg under {scene_dir}')
    pairs = [(p, q) for p, q in zip(paths, paths[1:]) if os.path.dirname(p) == os.path.dirname(q)]
    step = max(1, len(pairs) // limit)
    pairs = pairs[::step][:limit]
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])
    return tf, pairs


def cutpaste(x, size, gen):
    b, _, h, w = x.shape
    n = h // PATCH
    out = x.clone()
    mask = torch.zeros(b, n * n, dtype=torch.bool, device=x.device)
    for i in range(b):
        sy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        sx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        dy = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        dx = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        out[i, :, dy:dy + size, dx:dx + size] = x[i, :, sy:sy + size, sx:sx + size]
        for r in range(dy // PATCH, (dy + size - 1) // PATCH + 1):
            for c in range(dx // PATCH, (dx + size - 1) // PATCH + 1):
                mask[i, r * n + c] = True
    return out, mask


def cosd(a, b):
    return 1.0 - torch.cosine_similarity(a, b, dim=-1)


def ridge(X, Y, lam_scale=1e-3):
    xtx = X.T @ X
    lam = lam_scale * xtx.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0], device=X.device, dtype=X.dtype), X.T @ Y)


def solve_w(Xn, Xp, Xb, use_ratio, steps=400):
    """在 fit 帧上找一个 w, 让三层按 w 加权合并后 异常位移大 (可选: 且正常帧之间别乱动)。

    X*: (N, 3, 768) —— 第 1 维是组里的三个层。
    """
    def mix(w, X):
        return (w.view(1, 3, 1) * X).sum(1)

    best_w, best_score = None, -1e9
    for init in ([1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]):
        w = torch.tensor(init, device=Xn.device, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(steps):
            wn = w / w.norm().clamp_min(1e-8)
            g = cosd(mix(wn, Xn), mix(wn, Xp)).mean()
            score = g / cosd(mix(wn, Xn), mix(wn, Xb)).mean().clamp_min(1e-6) if use_ratio else g
            opt.zero_grad(); (-score).backward(); opt.step()
        with torch.no_grad():
            wn = w / w.norm().clamp_min(1e-8)
            g = cosd(mix(wn, Xn), mix(wn, Xp)).mean()
            if use_ratio:
                g = g / cosd(mix(wn, Xn), mix(wn, Xb)).mean().clamp_min(1e-6)
            if g.item() > best_score:
                best_score, best_w = g.item(), wn.detach().clone()
    return best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=list(SCENES))
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--paste', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev)
    print('teacher 12 blocks | fit 前半 算权重/预测器, eval 后半 报告', flush=True)

    for scene_key in a.scenes:
        tf, pairs = load_pairs(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                               a.img, teacher.img_mean, teacher.img_std, a.frames)
        n_fit = max(1, len(pairs) // 2)
        half = {s: {'n': [], 'p': [], 'b': [], 'm': []} for s in (True, False)}
        gen = torch.Generator().manual_seed(a.seed)
        with torch.no_grad():
            for i in range(0, len(pairs), a.bs):
                chunk = pairs[i:i + a.bs]
                xa = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in chunk]).to(dev)
                xb = torch.stack([tf(Image.open(q).convert('RGB')) for _, q in chunk]).to(dev)
                xp, mask = cutpaste(xa, a.paste, gen)
                sel = half[i < n_fit]
                for k, xx in (('n', xa), ('b', xb), ('p', xp)):
                    sel[k].append(torch.stack([ln(t) for t in teacher(xx)['patches']], 1))
                sel['m'].append(mask.reshape(-1))

        def packed(k, s):
            return torch.cat(half[s][k], 0).reshape(-1, 12, 768)

        def mask_of(s):
            return torch.cat(half[s]['m'], 0)

        N_f, P_f, B_f = packed('n', True), packed('p', True), packed('b', True)
        N_e, P_e, B_e = packed('n', False), packed('p', False), packed('b', False)
        M_e = mask_of(False)
        f_fit = sum(h.shape[0] for h in half[True]['n'])      # 按实际收上来的帧数算(batch 会跨界)
        f_ev = sum(h.shape[0] for h in half[False]['n'])

        avg = lambda v: sum(v) / len(v)
        print(f'\n===== scene={scene_key}  {len(pairs)} 对帧 (fit {f_fit} / eval {f_ev}) =====')
        print('   组          合并方式     top5位移   局部位移   正常波动   保真R2')
        for gi, idx in enumerate(GROUPS):
            gf = {k: v.reshape(-1, 3, 768) for k, v in
                  (('n', N_f[:, list(idx)]), ('p', P_f[:, list(idx)]), ('b', B_f[:, list(idx)]))}
            ge = {k: v.reshape(-1, 3, 768) for k, v in
                  (('n', N_e[:, list(idx)]), ('p', P_e[:, list(idx)]), ('b', B_e[:, list(idx)]))}
            per_gap = torch.tensor([cosd(gf['n'][:, j], gf['p'][:, j]).mean().item() for j in range(3)],
                                   device=dev)
            w_gapw = per_gap / per_gap.sum()
            sub = slice(None, None, 4)
            w_gap = solve_w(gf['n'][sub], gf['p'][sub], gf['b'][sub], use_ratio=False)
            w_ratio = solve_w(gf['n'][sub], gf['p'][sub], gf['b'][sub], use_ratio=True)
            # 内部残差: 用另外两层预测这一层(在 fit 帧上最小二乘), 取预测不掉的部分
            resid_w = [ridge(torch.cat([gf['n'][:, j] for j in range(3) if j != l], -1),
                             gf['n'][:, l]) for l in range(3)]

            def combine(t3):
                oth = [torch.cat([t3[:, j] for j in range(3) if j != l], -1) for l in range(3)]
                return torch.stack([t3[:, l] - oth[l] @ resid_w[l] for l in range(3)], 0).mean(0)

            def build(g):
                return {
                    'mean': (g['n'].mean(1), g['p'].mean(1), g['b'].mean(1)),
                    'gapw': tuple((w_gapw.view(1, 3, 1) * g[k]).sum(1) for k in 'npb'),
                    'diff': tuple(g[k][:, 2] - g[k][:, 0] for k in 'npb'),
                    'auto_gap': tuple((w_gap.view(1, 3, 1) * g[k]).sum(1) for k in 'npb'),
                    'auto_ratio': tuple((w_ratio.view(1, 3, 1) * g[k]).sum(1) for k in 'npb'),
                    'residual': tuple(combine(g[k]) for k in 'npb'),
                    'best': tuple(g[k][:, int(per_gap.argmax())] for k in 'npb'),
                }

            cf, ce = build(gf), build(ge)
            for name in AGGS:
                gn, gp, gb = ce[name]
                # fit 上解"从合并结果反推回原层"的线性映射, eval 上报告保真度
                fn = cf[name][0]
                r2 = []
                for l in range(3):
                    w_ = ridge(fn, gf['n'][:, l])
                    ye = ge['n'][:, l]
                    pred = gn @ w_
                    r2.append((1 - (ye - pred).pow(2).sum() / (ye - ye.mean(0)).pow(2).sum()).item())
                d = cosd(gn, gp).reshape(f_ev, -1)
                top5 = d.topk(5, dim=-1).values.mean(-1).mean().item()
                pasted = d[M_e.reshape(f_ev, -1)].mean().item()
                fl = cosd(gn, gb).mean().item()
                extra = ''
                if name in ('gapw', 'auto_gap', 'auto_ratio'):
                    w_show = {'gapw': w_gapw, 'auto_gap': w_gap, 'auto_ratio': w_ratio}[name]
                    extra = '   w=' + '/'.join(f'{v:+.2f}' for v in w_show.tolist())
                print(f'   G{gi} {idx}  {name:<10}  {top5:.4f}   {pasted:.4f}   {fl:.5f}   '
                      f'{avg(r2):.3f}{extra}')


if __name__ == '__main__':
    main()
