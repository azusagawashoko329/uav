"""审计-质疑2(test 上选 logz → 测试集过拟合):leave-one-scene-out 验证 logz 是否泛化。
每折只用其余 6 场景的平均融合 AUC 挑预处理(模拟没见过留出场景),再看留出场景。
预处理候选都用相同的合并 kNN-OOD(K=8)+ 相同 q 门控 + concat-rank,只变"联合前的标准化"。
读 experiments/da/audit/20260619_leakage_audit/scores/<scene>.npz,纯 CPU、自包含。Env: base/DMAD。例: python experiments/da/audit/20260619_leakage_audit/scripts/loso_preproc.py
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_EXP, 'scores')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
K, LOGZ_COLS = 8, [2, 3, 4]


def rank(s):
    srt = np.sort(s); return (np.searchsorted(srt, s, 'left') + np.searchsorted(srt, s, 'right')) / 2.0 / len(s)


def knn(tr, te):
    return NearestNeighbors(n_neighbors=min(K, len(tr))).fit(tr).kneighbors(te)[0].mean(1)


def p_zscore(Xtr, te):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; return knn((Xtr - mu) / sd, (te - mu) / sd)


def p_logz(Xtr, te):
    A, B = Xtr.copy(), te.copy()
    for i in LOGZ_COLS:
        A[:, i] = np.log1p(A[:, i]); B[:, i] = np.log1p(B[:, i])
    mu, sd = A.mean(0), A.std(0) + 1e-6; return knn((A - mu) / sd, (B - mu) / sd)


def p_robust(Xtr, te):
    med = np.median(Xtr, 0); q75, q25 = np.percentile(Xtr, [75, 25], 0); iqr = (q75 - q25) + 1e-6
    return knn((Xtr - med) / iqr, (te - med) / iqr)


PRE = {'z-score': p_zscore, 'logz': p_logz, 'robust': p_robust}


def fuse_auc(sa, sb, y, q):
    pa, pb = rank(sa), rank(sb)
    w = 0.5 * q * (1 - 2 * np.abs(pa - 0.5))
    return roc_auc_score(y, (1 - w) * pa + w * pb) * 100


def main():
    M = {}
    for sk in SCENES:
        f = os.path.join(D, f'{sk}.npz')
        if not os.path.exists(f):
            print(f'缺 {sk} 分数'); return
        d = np.load(f); sa, y, q, Xtr, te = d['sa'], d['y'], float(d['q']), d['Xtr_raw'], d['feat_te']
        M[sk] = {n: fuse_auc(sa, fn(Xtr, te), y, q) for n, fn in PRE.items()}

    names = list(PRE)
    print('=== 融合 AUC:场景 × 预处理 ===')
    print(f'{"scene":>12s}' + ''.join(f'{n:>10s}' for n in names))
    for sk in SCENES:
        print(f'{sk:>12s}' + ''.join(f'{M[sk][n]:10.2f}' for n in names))
    gmean = {n: np.mean([M[sk][n] for sk in SCENES]) for n in names}
    print(f'{"全局均值":>11s}' + ''.join(f'{gmean[n]:10.2f}' for n in names) + f'   ← 最优: {max(gmean, key=gmean.get)}')

    print('\n=== LOSO:每折只用其它 6 场景挑预处理 ===')
    print(f'{"留出场景":>11s}{"6折最优":>10s}{"留出logz":>10s}{"留出zscore":>11s}{"logz−zscore":>12s}')
    pick_logz = win = 0
    for s in SCENES:
        omean = {n: np.mean([M[o][n] for o in SCENES if o != s]) for n in names}
        best = max(omean, key=omean.get); dz = M[s]['logz'] - M[s]['z-score']
        pick_logz += (best == 'logz'); win += (dz >= 0)
        print(f'{s:>11s}{best:>10s}{M[s]["logz"]:10.2f}{M[s]["z-score"]:11.2f}{dz:+12.2f}')
    print(f'\n6 折挑出 logz: {pick_logz}/7  ·  留出场景 logz≥z-score: {win}/7')


if __name__ == '__main__':
    main()
