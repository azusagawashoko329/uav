"""联合 kNN 之前的"每维预处理"对比(纯 numpy,从 behavior/cache 重算,零检测)。
所有变体都是 [每维预处理 → 5维联合 kNN-OOD],只换预处理,看抗"尺度/重尾/相关裹挟"能力:
  base   每维 z-score(现状)
  robust 每维 (x-median)/IQR(抗离群)
  logz   计数维(n_close/count/max_local)log1p 后再 z-score(压重尾)
  rankU  每维 train∪test 合并 rank → 均匀[0,1](形状统一,离群免疫,transductive label-free)
  rankN  rankU 再 norm.ppf → 标准正态(形状统一且尾部有界)
  maha   全协方差白化(去维间相关,对照)
Env: DMAD。例: python fusion/scale_probe.py
"""
import os, importlib.util
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)

SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
APP = {'highway': 92.69, 'farmland': 99.47, 'bike': 98.08, 'solar': 86.11,
       'vehicle': 60.71, 'crossroads': 59.68, 'railway': 95.92}
K = l1.K


def knn(tr, te):
    nn = NearestNeighbors(n_neighbors=min(K, len(tr))).fit(tr)
    d, _ = nn.kneighbors(te); return d.mean(1)


def inv_sqrt(C):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
    return V @ np.diag(w ** -0.5) @ V.T


def qmap(tri, tei, normal=False):
    allv = np.concatenate([tri, tei]); srt = np.sort(allv)
    q = lambda x: (np.searchsorted(srt, x, 'left') + np.searchsorted(srt, x, 'right')) / 2.0 / len(srt)
    qt, qe = q(tri), q(tei)
    if normal:
        qt = norm.ppf(np.clip(qt, 1e-4, 1 - 1e-4)); qe = norm.ppf(np.clip(qe, 1e-4, 1 - 1e-4))
    return qt, qe


def p_zscore(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6; return (tr - mu) / sd, (te - mu) / sd


def p_robust(tr, te):
    med = np.median(tr, 0); q75, q25 = np.percentile(tr, [75, 25], 0); iqr = (q75 - q25) + 1e-6
    return (tr - med) / iqr, (te - med) / iqr


def p_logz(tr, te):
    tr2, te2 = tr.copy(), te.copy()
    for i in (2, 3, 4):
        tr2[:, i] = np.log1p(tr[:, i]); te2[:, i] = np.log1p(te[:, i])
    return p_zscore(tr2, te2)


def p_rank(tr, te, normal=False):
    TR, TE = [], []
    for i in range(5):
        qt, qe = qmap(tr[:, i], te[:, i], normal); TR.append(qt); TE.append(qe)
    return np.stack(TR, 1), np.stack(TE, 1)


def p_maha(tr, te):
    mu = tr.mean(0); C = np.cov(tr.T) + 1e-6 * np.eye(5); W = inv_sqrt(C)
    return (tr - mu) @ W, (te - mu) @ W


VARIANTS = [('base', p_zscore), ('robust', p_robust), ('logz', p_logz),
            ('rankU', lambda a, b: p_rank(a, b, False)),
            ('rankN', lambda a, b: p_rank(a, b, True)), ('maha', p_maha)]


def load_scene(sk):
    scene = l1.SCENES[sk]
    key = lambda s: sk + '/' + os.path.relpath(s, os.path.join(l1.DATA_ROOT, scene)).replace('/', '__')
    tr = np.concatenate([l1.subvideo_feats(None, s, key(s))[0] for s in l1.subvideos(scene, 'train')])
    te, ys = [], []
    for s in l1.subvideos(scene, 'test'):
        f, n = l1.subvideo_feats(None, s, key(s))
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        te.append(f[:n]); ys.append(lab)
    return tr.astype(float), np.concatenate(te).astype(float), np.concatenate(ys)


def main():
    print(''.join(f'{h:>10s}' for h in ['scene', '外观'] + [v[0] for v in VARIANTS]))
    accum = {v[0]: [] for v in VARIANTS}
    for sk in SCENES:
        tr, te, y = load_scene(sk)
        cells = [f'{sk:>10s}', f'{APP[sk]:10.2f}']
        for name, fn in VARIANTS:
            Tr, Te = fn(tr, te)
            a = roc_auc_score(y, knn(Tr, Te)) * 100 if 0 < y.sum() < len(y) else float('nan')
            accum[name].append(a); cells.append(f'{a:10.2f}')
        print(''.join(cells))
    print(f'{"均值":>9s}{np.mean(list(APP.values())):10.2f}' +
          ''.join(f'{np.mean(accum[v[0]]):10.2f}' for v in VARIANTS))


if __name__ == '__main__':
    main()
