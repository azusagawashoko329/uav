"""L1 行为子分支 —— 帧内"目标间距离 + 车数 + 局部打包密度"描述子 + 合并 kNN-OOD 打分(到正常点云 K 近邻均距)。
单帧,无跟踪,无时序。抓拥堵类异常:crossroads-s1(堵塞)、vehicle-s2(车道拥堵)。

训练口径(和外观分支一致):**整个大场景所有 seq 的 train 合并成一个正常点云**(不分 seq 训练)。
打分用 **合并 kNN-OOD**:测试帧到合并点云里 k 个最近正常点的平均距离 = 异常分。合并点云更密(正常
参照多)→ 分离更好(实测远胜分-seq kNN 和 LOF);kNN 局部性让不同 seq 自然落在各自邻域,单一尺度
使 concat 天然可比,无需 d_norm。报告**逐 seq**(不汇总平均)。

描述子(单帧):
  d_min, d_mean, n_close  —— 成对距离/间隔(ego 不变)
  count                   —— 车数。抓 vehicle-s2(车更多但间距≈正常 → 距离/密度失能,只有总数变)
  max_local               —— 局部打包度 = 各车 r0 内邻居数的最大值。抓 crossroads-s1(镜头移动,
                             "20 散开 vs 20 挤成团"车数都=20、count 失能;打包度只看局部聚集,与总数/区域无关)
消融:dist(3) / +count / +packing / +both,逐 seq 看每个场景到底吃哪个特征。
检测开 agnostic_nms(去跨类重复框)。独立于外观分支;对齐外观 eval(数字帧序、labels[-n:])。
highway 已放弃。Env: DMAD.  python behavior/l1.py --scenes crossroads,vehicle --gpu 0
"""
import argparse, os, glob, importlib.util, pickle
from collections import defaultdict
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location(
    'vehicle_detector', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector.py'))
_dm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dm)
build_vehicle_detector = _dm.build_vehicle_detector

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
SCENES = {'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
          'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
          'vehicle': 'Vehicle Roundabout'}
APP_REF = {'crossroads': 59.69, 'vehicle': 60.71, 'highway': 92.69}
D0 = 0.12       # "close pair" distance for n_close
R0 = 0.10       # radius for local packing (max_local)
K = 8           # kNN-OOD neighborhood
BATCH = 32
FEATS = ['d_min', 'd_mean', 'n_close', 'count', 'max_local']
SETS = [('dist(3)', [0, 1, 2]), ('+count', [0, 1, 2, 3]), ('+packing', [0, 1, 2, 4]), ('+both', [0, 1, 2, 3, 4])]


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def seq_of(sub):
    return os.path.basename(os.path.dirname(os.path.dirname(sub)))


def subvideo_centers(det, frames, cache_key):
    cp = os.path.join(CACHE, cache_key + '.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    h, w = cv2.imread(frames[0]).shape[:2]
    out = []
    for i in range(0, len(frames), BATCH):
        for d in det.detect(frames[i:i + BATCH]):
            b = d['boxes']
            c = (np.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h], 1).astype(np.float32)
                 if len(b) else np.zeros((0, 2), np.float32))
            out.append(c)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, 'wb') as f:
        pickle.dump(out, f)
    return out


def descriptor(c):
    n = len(c)
    if n < 2:
        return np.array([1., 1., 0., float(n), 0.], np.float32)
    D = squareform(pdist(c))                       # (n,n) full pairwise
    dd = D[np.triu_indices(n, 1)]                  # condensed upper-tri
    max_local = float((D < R0).sum(1).max() - 1)   # densest car's neighbor count (excl self)
    return np.array([float(dd.min()), float(dd.mean()), float((dd < D0).sum()), float(n), max_local], np.float32)


def subvideo_feats(det, sub, cache_key):
    frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=frame_idx)
    cents = subvideo_centers(det, frames, cache_key)
    feats = np.stack([descriptor(c) for c in cents]) if cents else np.zeros((0, 5), np.float32)
    return feats, len(frames)


def subvideos(scene, split):
    return [s for seq in sorted(glob.glob(os.path.join(DATA_ROOT, scene, 'sequence*')))
            for s in sorted(glob.glob(os.path.join(seq, split, '*'))) if os.path.isdir(s)]


def evalset(train_all, test, cols, tag):
    """MERGED training (all seqs pooled) + kNN-OOD (mean dist to k nearest normal). concat + per-seq AUC."""
    X = train_all[:, cols]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    nn = NearestNeighbors(n_neighbors=min(K, len(X))).fit((X - mu) / sd)
    subs = []
    for name, lab, f, n, sq in test:
        d, _ = nn.kneighbors((f[:, cols] - mu) / sd)
        subs.append((name, lab, d.mean(1)[:n]))
    y = np.concatenate([s[1] for s in subs]); allsc = np.concatenate([s[2] for s in subs])
    auc = roc_auc_score(y, allsc) * 100
    detail = '  '.join(
        f'{nm.split("/")[0].replace("sequence","s")}-{nm.split("/")[-1]}={roc_auc_score(lb, sc) * 100:4.1f}'
        for nm, lb, sc in subs if 0 < lb.sum() < len(lb))
    print(f'  [{tag:9s}] concat={auc:5.2f} | 逐seq: {detail}', flush=True)
    return auc


def run_scene(scene_key, det):
    scene = SCENES[scene_key]
    key = lambda s: scene_key + '/' + os.path.relpath(s, os.path.join(DATA_ROOT, scene)).replace('/', '__')
    train_all = [subvideo_feats(det, s, key(s))[0] for s in subvideos(scene, 'train')]   # MERGED: all seqs
    train_all = np.concatenate(train_all) if train_all else np.zeros((0, 5), np.float32)
    if len(train_all) < 50:
        print(f'[{scene_key}] too few train descriptors — vehicle-free, behavior silent', flush=True)
        return
    test = []
    for s in subvideos(scene, 'test'):
        f, n = subvideo_feats(det, s, key(s))
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        test.append((os.path.relpath(s, os.path.join(DATA_ROOT, scene)), lab, f, n, seq_of(s)))
    print(f'\n=== {scene_key} | 外观RD-topk={APP_REF.get(scene_key, 0):.1f} | 合并kNN-OOD, 合并训练, agnostic-NMS ===', flush=True)
    for tag, cols in SETS:
        evalset(train_all, test, cols, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='crossroads,vehicle')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()
    det = build_vehicle_detector(conf=a.conf, device=f'cuda:{a.gpu}')
    for sc in a.scenes.split(','):
        run_scene(sc, det)


if __name__ == '__main__':
    main()
