"""抽真实帧打印全链路数值,供 PIPELINE.md 举例(不改任何模型)。
vehicle: 找一个异常帧,行为高分、外观不笃定(展示行为补位)。
bike:    找一个正常帧,行为假阳高分(展示检测质量 q 压制)。Env: DMAD。
"""
import os, importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)
D = os.path.join(_ROOT, 'experiments/fusion/scores')
K, J0, F = l1.K, 0.1, l1.FEATS


def rank(s):
    srt = np.sort(s); return (np.searchsorted(srt, s, 'left') + np.searchsorted(srt, s, 'right')) / 2.0 / len(s)


def logz(tr, te):
    tr2, te2 = tr.copy(), te.copy()
    for i in (2, 3, 4):
        tr2[:, i] = np.log1p(tr[:, i]); te2[:, i] = np.log1p(te[:, i])
    mu, sd = tr2.mean(0), tr2.std(0) + 1e-6
    return (tr2 - mu) / sd, (te2 - mu) / sd, mu, sd


def load(sk):
    scene = l1.SCENES[sk]
    key = lambda s: sk + '/' + os.path.relpath(s, os.path.join(l1.DATA_ROOT, scene)).replace('/', '__')
    trsubs = [l1.subvideo_feats(None, s, key(s))[0] for s in l1.subvideos(scene, 'train')]
    tr = np.concatenate(trsubs)
    dn = np.concatenate([np.abs(np.diff(f[:, 3])) for f in trsubs if len(f) > 1])
    nn = np.concatenate([f[:, 3] for f in trsubs])
    jit = dn.mean() / nn.mean()
    te, ys = [], []
    for s in l1.subvideos(scene, 'test'):
        f, n = l1.subvideo_feats(None, s, key(s))
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        te.append(f[:n]); ys.append(lab)
    return tr.astype(float), np.concatenate(te).astype(float), np.concatenate(ys), jit


def trace(sk, idx, q, tr, te, y, sa, sb, Tez):
    pa, pb = rank(sa), rank(sb)
    bw = 1 - 2 * abs(pa[idx] - 0.5); w = 0.5 * q * bw
    fused = (1 - w) * pa[idx] + w * pb[idx]
    print(f'\n========== {sk}  全局帧 idx={idx}  标签={"异常" if y[idx] else "正常"} ==========')
    print(f' 5维原始 : ' + '  '.join(f'{F[i]}={te[idx,i]:.4f}' for i in range(5)))
    print(f' logz后  : ' + '  '.join(f'{F[i]}={Tez[idx,i]:+.3f}' for i in range(5)))
    print(f' 行为 sb={sb[idx]:.4f}  → 平均秩 p_beh={pb[idx]:.3f}')
    print(f' 外观 sa={sa[idx]:.4f}  → 平均秩 p_app={pa[idx]:.3f}')
    print(f' q={q:.3f}  base_w=1-2|{pa[idx]:.3f}-0.5|={bw:.3f}  w=0.5*{q:.3f}*{bw:.3f}={w:.3f}')
    print(f' fused=(1-{w:.3f})*{pa[idx]:.3f} + {w:.3f}*{pb[idx]:.3f} = {fused:.3f}')


for sk, want in [('vehicle', 1), ('bike', 0)]:
    tr, te, y, jit = load(sk)
    q = min(1.0, J0 / jit)
    a = np.load(os.path.join(D, f'app_{sk}.npz')); sa = a['scores'].astype(float)
    _, Tez, _, _ = logz(tr, te)
    nn = NearestNeighbors(n_neighbors=K).fit(logz(tr, te)[0]); sb = nn.kneighbors(Tez)[0].mean(1)
    pa, pb = rank(sa), rank(sb)
    # 选:标签=want,行为高分,外观不极端
    mask = (y == want) & (pb > (0.8 if want else 0.65))
    cand = np.where(mask)[0]
    if want == 1:
        cand = cand[np.argsort(np.abs(pa[cand] - 0.5))]   # 外观最不笃定的异常帧
    else:
        cand = cand[np.argsort(-pb[cand])]                # 行为假阳最高的正常帧
    print(f'\n### {sk}  jitter={jit:.3f}  q={q:.3f}  (候选{len(cand)}帧)')
    trace(sk, int(cand[0]), q, tr, te, y, sa, sb, Tez)
