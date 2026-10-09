"""门控 × 检测质量置信(§6 的 q 并回门控),全 7 场景。
行为分:logz 预处理 + 联合 K=8 kNN-OOD(从 behavior/cache 重算)。
检测质量:q=min(1, j0/jitter),jitter=mean(|Δn|)/mean(n) 在 train 帧上算(场景级,n=每帧车数=描述子count维)。
门控:w_beh = 0.5·q·(1−2|p_app−0.5|);fused=(1−w)·p_app + w·p_beh,p=test 自身平均秩。
对比"logz门控(无q)" vs "logz门控×q"。口径:concat-micro AUC。Env: DMAD。例: python fusion/gate_q.py
"""
import os, importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)
D = os.path.join(_ROOT, 'experiments/da/fusion/legacy_score_dump/scores')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
K, J0 = l1.K, 0.1


def rank(s):
    srt = np.sort(s); lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def knn(tr, te):
    nn = NearestNeighbors(n_neighbors=min(K, len(tr))).fit(tr); d, _ = nn.kneighbors(te); return d.mean(1)


def p_logz(tr, te):
    tr2, te2 = tr.copy(), te.copy()
    for i in (2, 3, 4):
        tr2[:, i] = np.log1p(tr[:, i]); te2[:, i] = np.log1p(te[:, i])
    mu, sd = tr2.mean(0), tr2.std(0) + 1e-6
    return (tr2 - mu) / sd, (te2 - mu) / sd


def load_scene(sk):
    scene = l1.SCENES[sk]
    key = lambda s: sk + '/' + os.path.relpath(s, os.path.join(l1.DATA_ROOT, scene)).replace('/', '__')
    trsubs = [l1.subvideo_feats(None, s, key(s))[0] for s in l1.subvideos(scene, 'train')]
    tr = np.concatenate(trsubs)
    # jitter: 每 train seq 内 count 差分(seq 内,不跨边界)
    dn = np.concatenate([np.abs(np.diff(f[:, 3])) for f in trsubs if len(f) > 1])
    nn = np.concatenate([f[:, 3] for f in trsubs])
    jit = dn.mean() / (nn.mean() + 1e-9)
    te, ys = [], []
    for s in l1.subvideos(scene, 'test'):
        f, n = l1.subvideo_feats(None, s, key(s))
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        te.append(f[:n]); ys.append(lab)
    return tr.astype(float), np.concatenate(te).astype(float), np.concatenate(ys), jit


def auc(y, s):
    return roc_auc_score(y, s) * 100


def main():
    hdr = ['scene', 'jitter', 'q', '外观', 'logz门控', '门控×q', 'Δ(vs外观)']
    print(''.join(f'{h:>10s}' for h in hdr))
    acc = {k: [] for k in ['app', 'g', 'gq']}
    for sk in SCENES:
        tr, te, y, jit = load_scene(sk)
        q = min(1.0, J0 / (jit + 1e-9))
        a = np.load(os.path.join(D, f'app_{sk}.npz')); sa, ya = a['scores'].astype(float), a['labels'].astype(int)
        if len(ya) != len(y) or not (ya == y).all():
            print(f'{sk:>10s}  对齐失败'); continue
        pa, pb = rank(sa), rank(knn(*p_logz(tr, te)))
        base_w = 1 - 2 * np.abs(pa - 0.5)
        g = (1 - 0.5 * base_w) * pa + 0.5 * base_w * pb
        gq = (1 - 0.5 * q * base_w) * pa + 0.5 * q * base_w * pb
        a_app, a_g, a_gq = auc(y, sa), auc(y, g), auc(y, gq)
        acc['app'].append(a_app); acc['g'].append(a_g); acc['gq'].append(a_gq)
        print(f'{sk:>10s}{jit:10.3f}{q:10.3f}{a_app:10.2f}{a_g:10.2f}{a_gq:10.2f}{a_gq - a_app:+10.2f}')
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f'{"均值":>9s}{"":10s}{"":10s}{m["app"]:10.2f}{m["g"]:10.2f}{m["gq"]:10.2f}{m["gq"] - m["app"]:+10.2f}')


if __name__ == '__main__':
    main()
