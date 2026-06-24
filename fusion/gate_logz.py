"""把 logz 预处理接进门控融合,全 7 场景对比 base。
行为分:从 behavior/cache 重算(base=每维z-score / logz=计数维log1p后z-score)→ 联合 K=8 kNN-OOD。
外观分:读 experiments/fusion/scores/app_<scene>.npz。
融合:门控 w_beh = wmax·(1−2|p_app−0.5|),fused=(1−w)·p_app + w·p_beh,p=test 自身平均秩。
口径:每场景 concat-micro AUC(MADViT 可比)。Env: DMAD。例: python fusion/gate_logz.py
"""
import os, importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)
D = os.path.join(_ROOT, 'experiments/fusion/scores')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
K = l1.K


def rank(s):
    srt = np.sort(s); lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def knn(tr, te):
    nn = NearestNeighbors(n_neighbors=min(K, len(tr))).fit(tr); d, _ = nn.kneighbors(te); return d.mean(1)


def p_zscore(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6; return (tr - mu) / sd, (te - mu) / sd


def p_logz(tr, te):
    tr2, te2 = tr.copy(), te.copy()
    for i in (2, 3, 4):
        tr2[:, i] = np.log1p(tr[:, i]); te2[:, i] = np.log1p(te[:, i])
    return p_zscore(tr2, te2)


def gate(pa, pb, wmax=0.5):
    w = wmax * (1 - 2 * np.abs(pa - 0.5)); return (1 - w) * pa + w * pb


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


def auc(y, s):
    return roc_auc_score(y, s) * 100


def main():
    hdr = ['scene', '外观', 'base行为', 'logz行为', 'base门控', 'logz门控', 'Δ门控']
    print(''.join(f'{h:>10s}' for h in hdr))
    acc = {k: [] for k in ['app', 'fb', 'fl']}
    for sk in SCENES:
        tr, te, y = load_scene(sk)
        a = np.load(os.path.join(D, f'app_{sk}.npz')); sa, ya = a['scores'].astype(float), a['labels'].astype(int)
        if len(ya) != len(y) or not (ya == y).all():
            print(f'{sk:>10s}  对齐失败 行为{len(y)} 外观{len(ya)}'); continue
        pa = rank(sa)
        sbb, sbl = knn(*p_zscore(tr, te)), knn(*p_logz(tr, te))
        a_app, a_bb, a_bl = auc(y, sa), auc(y, sbb), auc(y, sbl)
        a_fb, a_fl = auc(y, gate(pa, rank(sbb))), auc(y, gate(pa, rank(sbl)))
        acc['app'].append(a_app); acc['fb'].append(a_fb); acc['fl'].append(a_fl)
        print(f'{sk:>10s}{a_app:10.2f}{a_bb:10.2f}{a_bl:10.2f}{a_fb:10.2f}{a_fl:10.2f}{a_fl - a_fb:+10.2f}')
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f'{"均值":>9s}{m["app"]:10.2f}{"":10s}{"":10s}{m["fb"]:10.2f}{m["fl"]:10.2f}{m["fl"] - m["fb"]:+10.2f}')


if __name__ == '__main__':
    main()
