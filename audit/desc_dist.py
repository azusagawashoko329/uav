"""训练集五维描述子分布分析 —— logz 的重尾(拖尾)动机,自包含读 audit/scores/<scene>.npz 的 Xtr_raw。
逐维打印分位/max/skew/skew(log1p);并给关键论证:z-score 是线性变换、不改偏度(skew_raw == skew_zscore),
唯 log1p 把计数维偏度从 2~5 压到 ~1 → 故 logz = 计数维 log1p + 全维 z-score。
拖尾只在 car-dense(highway/crossroads/bike)成立;少车场景计数维近乎全 0、高 skew 是稀疏伪影(不作证据)。
出图见 audit/cloud_dist.py(cloud_<scene>.png)。Env: DMAD。例: python audit/desc_dist.py
"""
import os
import numpy as np
from scipy.stats import skew

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'audit', 'scores')
FEATS = ['d_min', 'd_mean', 'n_close', 'count', 'max_local']
COUNT_DIMS = [2, 3, 4]
SCENES = ['highway', 'crossroads', 'bike', 'vehicle', 'solar', 'farmland', 'railway']


def main():
    data = {}
    for sk in SCENES:
        f = os.path.join(D, f'{sk}.npz')
        if not os.path.exists(f):
            print(f'缺 {sk} 分数,跳过'); continue
        X = np.load(f)['Xtr_raw'].astype(float)
        data[sk] = X
        print(f'\n=== {sk}  train {len(X)} 帧  (★=计数维) ===')
        print(f'{"维":>10s}{"mean":>8s}{"std":>8s}{"p50":>7s}{"p90":>7s}{"p99":>7s}{"max":>7s}'
              f'{"skew":>8s}{"skew_log":>9s}')
        for i, name in enumerate(FEATS):
            x = X[:, i]; p50, p90, p99 = np.percentile(x, [50, 90, 99]); mark = '★' if i in COUNT_DIMS else ' '
            sr, sl = skew(x), skew(np.log1p(x))
            ss = f'{sr:8.2f}' if np.isfinite(sr) else f'{"--":>8s}'
            ls = f'{sl:9.2f}' if np.isfinite(sl) else f'{"--":>9s}'
            print(f'{name+mark:>10s}{x.mean():8.2f}{x.std():8.2f}{p50:7.2f}{p90:7.2f}{p99:7.2f}{x.max():7.2f}{ss}{ls}')

    print('\n=== 偏度三态(计数维):z-score 与 raw 完全相同(线性不改偏度),只有 log1p 压尾 ===')
    print(f'{"scene":>11s}{"dim":>10s}{"skew_raw":>10s}{"skew_zscore":>12s}{"skew_logz":>11s}')
    for sk, X in data.items():
        z = (X - X.mean(0)) / (X.std(0) + 1e-6)                 # 纯 z-score(线性变换)
        L = X.copy()
        for i in COUNT_DIMS:
            L[:, i] = np.log1p(L[:, i])
        lz = (L - L.mean(0)) / (L.std(0) + 1e-6)                # logz
        for i in COUNT_DIMS:
            g = lambda v: f'{v:.2f}' if np.isfinite(v) else '--'
            print(f'{sk:>11s}{FEATS[i]:>10s}{g(skew(X[:, i])):>10s}{g(skew(z[:, i])):>12s}{g(skew(lz[:, i])):>11s}')


if __name__ == '__main__':
    main()
