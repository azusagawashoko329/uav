"""全场景晚融合:两分支各自 test-rank → mean(逐帧平均)→ 每场景 concat AUC,对比纯外观。
读 experiments/da/fusion/legacy_score_dump/scores/{app,beh}_<scene>.npz(test 帧分数+标签)。
默认全 7 场景都融合(有行为分数就 mean,无则退化为纯外观);融合分落盘 fused_<scene>.npz。
注:test-rank 是 transductive 的无标签后处理(用整场景 test 分数排序,不碰标签),离线评测口径下成立。
Env: base/DMAD。例: python fusion/fuse.py
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'experiments/da/fusion/legacy_score_dump/scores')
SCENES = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']  # 与 report 同序


def rank(s):
    """平均秩百分位 ∈[0,1]:平局取中间名次(常数→0.5)。必须用平均秩,否则 argsort 会把大量相同分数
    (如无车帧 kNN=0,farmland 平局率 99%)排成伪随机名次、凭空注入噪声,破坏"静默分支 mean 无害"。"""
    srt = np.sort(s)
    lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def main():
    print(f'{"scene":12s}{"纯外观":>9s}{"行为":>9s}{"mean融合":>10s}{"Δ":>9s}')
    A, F = [], []
    for sk in SCENES:
        ap = os.path.join(D, f'app_{sk}.npz')
        if not os.path.exists(ap):
            print(f'{sk:12s}{"(无外观分数,跳过)":>20s}'); continue
        a = np.load(ap); sa, y = a['scores'].astype(float), a['labels'].astype(int)
        auc_a = roc_auc_score(y, sa) * 100
        bp = os.path.join(D, f'beh_{sk}.npz')
        auc_b, auc_f, fused = float('nan'), auc_a, sa
        if os.path.exists(bp):
            b = np.load(bp); sb, yb = b['scores'].astype(float), b['labels'].astype(int)
            if len(yb) == len(y) and (yb == y).all():
                auc_b = roc_auc_score(y, sb) * 100
                fused = (rank(sa) + rank(sb)) / 2
                auc_f = roc_auc_score(y, fused) * 100
        np.savez(os.path.join(D, f'fused_{sk}.npz'), scores=fused, labels=y)
        A.append(auc_a); F.append(auc_f)
        bs = f'{auc_b:9.2f}' if auc_b == auc_b else f'{"--":>9s}'
        print(f'{sk:12s}{auc_a:9.2f}{bs}{auc_f:10.2f}{auc_f - auc_a:+9.2f}')
    print(f'{"均值":10s}{np.mean(A):9.2f}{"":9s}{np.mean(F):10.2f}{np.mean(F) - np.mean(A):+9.2f}')


if __name__ == '__main__':
    main()
