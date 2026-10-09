"""验证 per-scene 百分位标准化(用 train 正常分布的经验 CDF)是否改变各分支单场景 AUC。
读 experiments/da/fusion/legacy_score_dump/scores/{app,beh}_<scene>.npz(test scores+labels)与 *_train.npz(train CDF 锚),
对比"原始分数 AUC" vs "train-CDF 百分位后 AUC"。纯 numpy/sklearn,无模型。Env: base 或 DMAD。
例: python fusion/verify_percentile.py
"""
import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'experiments/da/fusion/legacy_score_dump/scores')


def cdf(test_s, train_s):
    """test 分数在 train 经验分布里的分位 ∈[0,1](单调;尾部 >train_max 饱和到 1)。"""
    ts = np.sort(train_s)
    return np.searchsorted(ts, test_s, side='right') / len(ts)


print(f'{"file":24s} {"原始AUC":>9s} {"百分位AUC":>10s} {"Δ":>9s}')
for f in sorted(glob.glob(os.path.join(D, '*.npz'))):
    b = os.path.basename(f)
    if b.endswith('_train.npz'):
        continue
    tr = f.replace('.npz', '_train.npz')
    if not os.path.exists(tr):
        continue
    d = np.load(f); s, y = d['scores'].astype(float), d['labels']
    if not (0 < y.sum() < len(y)):
        continue
    trs = np.load(tr)['scores'].astype(float)
    a0 = roc_auc_score(y, s) * 100
    a1 = roc_auc_score(y, cdf(s, trs)) * 100
    print(f'{b:24s} {a0:9.3f} {a1:10.3f} {a1 - a0:+9.4f}')
