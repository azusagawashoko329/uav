"""原始 train 点云分布 + logz 处理后对照 —— 让"尺度差异大 + 计数维拖尾"以及"logz 如何治好它们"肉眼可见
(数据真实来自 audit/scores/<scene>.npz 的 Xtr_raw)。

每场景一张 cloud_<scene>.png,左右两列 × 6 行:
  左列 = 原始 raw           右列 = logz 后(计数维 log1p + 全维 z-score)
  上 5 行 = 五维各自直方图,纵轴 log(贴地长尾抬起 → 拖尾可见);黑虚线=p99、灰点线=max;
           蓝=距离维 红=计数维。右列标注 skew 的变化(raw→logz)。
  最下 1 行 = "尺度尺子":左=raw(深条到 p99、浅条到 max,距离维钉 0 附近、计数维拉到几十上百 → 尺度差);
             右=logz(p1~p99 区间,五维 std=1、范围拉齐 → 尺度差被消除)。
拖尾只在 car-dense(highway/crossroads/bike)成立;少车场景计数维近全 0、高 skew 是稀疏伪影(图里照实呈现)。
同时终端打印每场景 q、d_min 真实分位(回答"d_min 贴 0 是不是追踪烂")。
Env: DMAD。例: python audit/cloud_dist.py
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'audit', 'scores')
FIG = os.path.join(_ROOT, 'audit', 'figs'); os.makedirs(FIG, exist_ok=True)
F = ['d_min', 'd_mean', 'n_close', 'count', 'max_local']
COUNT = [2, 3, 4]
BLUE, VERM = '#2c6fbb', '#d1495b'
COL = [BLUE, BLUE, VERM, VERM, VERM]
DENSE = ['highway', 'crossroads', 'bike']                  # 拖尾真正成立的 car-dense 场景
SCENES = ['highway', 'crossroads', 'bike', 'vehicle', 'solar', 'farmland', 'railway']


def logz_tr(X):                                            # logz:计数维 log1p,再全维 z-score
    L = X.astype(float).copy()
    for i in COUNT:
        L[:, i] = np.log1p(L[:, i])
    return (L - L.mean(0)) / (L.std(0) + 1e-6)


def cloud(X, q, scene):
    Z = logz_tr(X)
    fig = plt.figure(figsize=(13.5, 10.6))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1.15], hspace=0.6, wspace=0.16,
                          top=0.90, bottom=0.06, left=0.085, right=0.985)
    for i, name in enumerate(F):
        x = X[:, i]; z = Z[:, i]; sk = skew(x); szk = skew(z)
        # 左:raw
        axL = fig.add_subplot(gs[i, 0])
        bins = (np.arange(0, int(x.max()) + 2) - 0.5) if (i in COUNT and x.max() <= 60) \
            else np.linspace(0, x.max() * 1.001 + 1e-9, 60)
        axL.hist(x, bins=bins, color=COL[i]); axL.set_yscale('log')
        p90, p99 = np.percentile(x, [90, 99]); mx = x.max()
        axL.axvline(p99, color='k', ls='--', lw=0.8); axL.axvline(mx, color='gray', ls=':', lw=0.8)
        axL.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10)
        tag = '[计数]' if i in COUNT else '[距离]'
        axL.text(0.99, 0.90, f'p90={p90:.1f} p99={p99:.1f} max={mx:.0f} skew={sk:.1f} {tag}',
                 transform=axL.transAxes, ha='right', va='top', fontsize=7.6,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        axL.set_xlim(left=0)
        # 右:logz
        axR = fig.add_subplot(gs[i, 1])
        axR.hist(z, bins=np.linspace(z.min(), z.max() + 1e-9, 55), color=COL[i]); axR.set_yscale('log')
        axR.axvline(np.percentile(z, 99), color='k', ls='--', lw=0.8)
        axR.axvline(0, color='#aaaaaa', lw=0.6)
        chg = '(距离维:仅 z-score,形状/偏度不变)' if i not in COUNT else f'(计数维:偏度 {sk:.1f}→{szk:.1f},尾被压平)'
        axR.text(0.99, 0.90, f'skew {sk:.1f}→{szk:.1f}  std=1\n{chg}', transform=axR.transAxes,
                 ha='right', va='top', fontsize=7.0, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    # 底部尺度尺子:左 raw / 右 logz
    y = np.arange(5)
    axl = fig.add_subplot(gs[5, 0])
    mx, p99 = X.max(0), np.percentile(X, 99, 0)
    axl.barh(y, mx, color=COL, alpha=0.22, height=0.66)
    axl.barh(y, p99, color=COL, alpha=0.95, height=0.66)
    axl.set_yticks(y); axl.set_yticklabels(F, fontsize=8.5); axl.invert_yaxis()
    axl.set_xlim(0, mx.max() * 1.18)
    for i in range(5):
        axl.text(mx[i] + mx.max() * 0.01, y[i], f'max {mx[i]:.0f}', va='center', fontsize=7.5, color='#333')
    axl.set_title('raw 尺度尺子(深=p99 浅=max):距离维钉 0 附近、计数维拉到几十上百 → 尺度差大', fontsize=8.5)
    axl.set_xlabel('value (linear)')

    axr = fig.add_subplot(gs[5, 1])
    zp1, zp99 = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    axr.barh(y, zp99 - zp1, left=zp1, color=COL, alpha=0.9, height=0.66)
    axr.axvline(0, color='#888', lw=0.7, ls='--')
    axr.set_yticks(y); axr.set_yticklabels(F, fontsize=8.5); axr.invert_yaxis()
    axr.set_title('logz 尺度尺子(p1~p99 区间):五维 std=1、范围拉齐 → 尺度差消失', fontsize=8.5)
    axr.set_xlabel('z-score (std units)')

    # 列大标题 + 总标题
    fig.text(0.295, 0.935, '原始 raw', ha='center', fontsize=13, weight='bold')
    fig.text(0.74, 0.935, 'logz 后(计数维 log1p + 全维 z-score)', ha='center', fontsize=13, weight='bold')
    note = '★ car-dense:拖尾真实' if scene in DENSE else '○ 少车:近乎全 default/全 0,高 skew 是伪影'
    fig.text(0.5, 0.975, f'{scene}   train {len(X)} 帧   q={q:.2f}    {note}   '
             f'(纵轴 log;蓝=距离 红=计数;黑虚线=p99 灰点线=max)', ha='center', fontsize=11.5)
    p = os.path.join(FIG, f'cloud_{scene}.png'); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def main():
    print(f'{"scene":>11s}{"q":>6s}{"dmin_p50":>9s}{"dmin_p90":>9s}{"dmin=default%":>14s}{"nclose_max":>11s}')
    for sc in SCENES:
        f = os.path.join(D, f'{sc}.npz')
        if not os.path.exists(f):
            print(f'缺 {sc}'); continue
        d = np.load(f); X = d['Xtr_raw'].astype(float); q = float(d['q'])
        dmin = X[:, 0]; defrac = 100.0 * np.mean(np.isclose(dmin, 1.0))
        print(f'{sc:>11s}{q:6.2f}{np.percentile(dmin,50):9.3f}{np.percentile(dmin,90):9.3f}'
              f'{defrac:13.1f}%{X[:,2].max():11.0f}')
        cloud(X, q, sc)
    print(f'\nsaved cloud_<scene>.png x{len(SCENES)} (dpi 120) -> {FIG}')


if __name__ == '__main__':
    main()
