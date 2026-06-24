"""审计-质疑1(test-rank transductive):三种标准化协议下的融合 AUC 对比。
  concat-rank   每场景所有 test 帧拼接后排序(当前;与 MADViT 全局 min-max 同范围,见 REPORT.md)
  per-subvideo  每 test 子视频内部排序(VAD 通行 per-video 归一化)
  causal-online 每帧只用本子视频"该帧及之前"排序(严格 inductive / 可在线)
行为分 sb 本身 inductive(test 帧→train 点云 kNN);唯一 test 依赖在 rank。读 audit/scores/<scene>.npz。
纯 CPU、自包含。Env: base/DMAD。例: python audit/norm_protocols.py
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'audit', 'scores')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
K, LOGZ_COLS = 8, [2, 3, 4]


def rank_concat(s):
    srt = np.sort(s); lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def rank_persub(s, lens):
    out = np.empty(len(s)); off = 0
    for L in lens:
        out[off:off + L] = rank_concat(s[off:off + L]); off += L
    return out


def rank_causal(s, lens):
    out = np.empty(len(s)); off = 0
    for L in lens:
        seg = s[off:off + L]
        for i in range(L):
            h = seg[:i + 1]
            out[off + i] = ((h < seg[i]).sum() + (h <= seg[i]).sum()) / 2.0 / (i + 1)
        off += L
    return out


def behavior_sb(Xtr_raw, feat_te):
    A, B = Xtr_raw.astype(float).copy(), feat_te.astype(float).copy()
    for i in LOGZ_COLS:
        A[:, i] = np.log1p(A[:, i]); B[:, i] = np.log1p(B[:, i])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    nn = NearestNeighbors(n_neighbors=K).fit((A - mu) / sd)
    return nn.kneighbors((B - mu) / sd)[0].mean(1)


def fuse(pa, pb, q):
    w = 0.5 * q * (1 - 2 * np.abs(pa - 0.5)); return (1 - w) * pa + w * pb


def main():
    print(''.join(f'{h:>13s}' for h in ['scene', '纯外观', 'concat-rank', 'per-subvideo', 'causal-online', 'q']))
    acc = {k: [] for k in ['app', 'concat', 'persub', 'causal']}
    for sk in SCENES:
        f = os.path.join(D, f'{sk}.npz')
        if not os.path.exists(f):
            print(f'{sk:>13s}   (无分数,跳过)'); continue
        d = np.load(f); sa, y, lens, q = d['sa'], d['y'], d['sub_lens'], float(d['q'])
        sb = behavior_sb(d['Xtr_raw'], d['feat_te'])
        r = {'app': roc_auc_score(y, sa) * 100}
        r['concat'] = roc_auc_score(y, fuse(rank_concat(sa), rank_concat(sb), q)) * 100
        r['persub'] = roc_auc_score(y, fuse(rank_persub(sa, lens), rank_persub(sb, lens), q)) * 100
        r['causal'] = roc_auc_score(y, fuse(rank_causal(sa, lens), rank_causal(sb, lens), q)) * 100
        for k in acc:
            acc[k].append(r[k])
        print(f'{sk:>13s}{r["app"]:13.2f}{r["concat"]:13.2f}{r["persub"]:13.2f}{r["causal"]:13.2f}{q:13.3f}')
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f'{"均值":>12s}{m["app"]:13.2f}{m["concat"]:13.2f}{m["persub"]:13.2f}{m["causal"]:13.2f}')
    print(f'\nΔ vs 纯外观:concat {m["concat"]-m["app"]:+.2f} | per-subvideo {m["persub"]-m["app"]:+.2f} | '
          f'causal-online {m["causal"]-m["app"]:+.2f}')


if __name__ == '__main__':
    main()
