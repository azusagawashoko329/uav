"""行为分支描述子探针(纯 numpy,从 behavior/cache 重算特征,零检测、零 GPU)。
两张表:
  表A per-dim 单维行为 AUC(scene×5维)—— 证明不同场景吃不同维 + 给单维 oracle 上界。
  表B 聚合变体 AUC:base(5维z-score联合kNN) / maha(全协方差白化) / dimmean / dimmax /
       dimw-cv(判别力=变异系数加权) / dimw-tail(上尾比加权) / oracle单维。
复用 behavior/l1.py 的 descriptor/subvideos/缓存;det=None(缓存全 hit)。Env: base/DMAD。
例: python fusion/desc_probe.py
"""
import os, importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)

SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
APP = {'highway': 92.69, 'farmland': 99.47, 'bike': 98.08, 'solar': 86.11,
       'vehicle': 60.71, 'crossroads': 59.68, 'railway': 95.92}
K = l1.K


def rank(s):
    srt = np.sort(s); lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def knn(tr, te, k=K):
    nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(tr)
    d, _ = nn.kneighbors(te); return d.mean(1)


def inv_sqrt(C):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
    return V @ np.diag(w ** -0.5) @ V.T


def load_scene(sk):
    scene = l1.SCENES[sk]
    key = lambda s: sk + '/' + os.path.relpath(s, os.path.join(l1.DATA_ROOT, scene)).replace('/', '__')
    tr = np.concatenate([l1.subvideo_feats(None, s, key(s))[0] for s in l1.subvideos(scene, 'train')])
    te, ys = [], []
    for s in l1.subvideos(scene, 'test'):
        f, n = l1.subvideo_feats(None, s, key(s))
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        te.append(f[:n]); ys.append(lab)
    return tr, np.concatenate(te), np.concatenate(ys)


def auc(y, s):
    return roc_auc_score(y, s) * 100 if 0 < y.sum() < len(y) else float('nan')


def main():
    rowsA, rowsB = [], []
    for sk in SCENES:
        tr, te, y = load_scene(sk)
        mu, sd = tr.mean(0), tr.std(0) + 1e-6
        Trz, Tez = (tr - mu) / sd, (te - mu) / sd
        # per-dim 1D OOD
        ds = [knn(Trz[:, [i]], Tez[:, [i]]) for i in range(5)]
        a_dim = [auc(y, ds[i]) for i in range(5)]
        rowsA.append((sk, a_dim))
        # base 联合
        a_base = auc(y, knn(Trz, Tez))
        # maha 全协方差白化
        C = np.cov(tr.T) + 1e-6 * np.eye(5); W = inv_sqrt(C)
        a_maha = auc(y, knn((tr - mu) @ W, (te - mu) @ W))
        # per-dim 聚合
        R = np.stack([rank(d) for d in ds], 0)           # (5, N)
        a_mean = auc(y, R.mean(0)); a_max = auc(y, R.max(0))
        # 判别力权重(无监督,test 自身):变异系数 CV=std/mean,越散=越可能含异常信号
        cv = np.array([d.std() / (d.mean() + 1e-9) for d in ds])
        a_cv = auc(y, (cv[:, None] * R).sum(0) / cv.sum())
        a_ora = np.nanmax(a_dim)
        rowsB.append((sk, a_base, a_maha, a_mean, a_max, a_cv, a_ora))

    F = l1.FEATS
    print('\n=== 表A: per-dim 单维行为 AUC(★=该场景最佳维) ===')
    print(f'{"scene":11s}' + ''.join(f'{f:>10s}' for f in F))
    for sk, a in rowsA:
        bi = int(np.nanargmax(a))
        cells = ''.join((f'{v:9.2f}★' if i == bi else f'{v:9.2f} ') for i, v in enumerate(a))
        print(f'{sk:11s}{cells}')

    print('\n=== 表B: 聚合变体行为 AUC(对比纯外观) ===')
    hdr = ['scene', '外观', 'base', 'maha', 'dimmean', 'dimmax', 'dimw-cv', 'oracle单维']
    print(''.join(f'{h:>11s}' for h in hdr))
    cols = list(zip(*[r[1:] for r in rowsB]))
    means = [np.nanmean(c) for c in cols]
    for sk, *vals in rowsB:
        print(f'{sk:>11s}{APP[sk]:11.2f}' + ''.join(f'{v:11.2f}' for v in vals))
    print(f'{"均值":>10s}{np.mean(list(APP.values())):11.2f}' + ''.join(f'{m:11.2f}' for m in means))


if __name__ == '__main__':
    main()
