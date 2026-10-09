"""诊断/融合工具:导出行为分支(合并训练 + 合并 kNN-OOD,+both=5维特征)逐帧分数,按 normal vs anom
看量级,并落盘 npz(供融合)。复用 behavior/l1.py 的检测/描述子/合并训练,零改动已有文件。
--train 时额外导出 train 正常帧分数(leave-self-out kNN,去掉自身那个 dist=0 的近邻)作 CDF 锚。
Env: DMAD。可从仓库任意位置运行。例: python fusion/dump_behavior_scores.py --scenes crossroads,vehicle --train --gpu 0
"""
import argparse, os, importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)


def stats(s):
    p = np.percentile(s, [0, 5, 50, 95, 100])
    return (f'n={len(s):5d} mean={s.mean():.4f} std={s.std():.4f} '
            f'min={p[0]:.4f} p5={p[1]:.4f} med={p[2]:.4f} p95={p[3]:.4f} max={p[4]:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='crossroads,vehicle')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(_ROOT, 'experiments/da/fusion/legacy_score_dump/scores'))
    ap.add_argument('--train', action='store_true', help='额外 dump train 正常帧分数(leave-self-out)作 CDF 锚')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det = l1.build_vehicle_detector(device=f'cuda:{a.gpu}')
    cols = [0, 1, 2, 3, 4]   # +both = 全 5 维(与 README 锁定的 L1 特征集一致)
    K = l1.K
    for sk in a.scenes.split(','):
        scene = l1.SCENES[sk]
        key = lambda s: sk + '/' + os.path.relpath(s, os.path.join(l1.DATA_ROOT, scene)).replace('/', '__')
        train_all = np.concatenate([l1.subvideo_feats(det, s, key(s))[0] for s in l1.subvideos(scene, 'train')])
        X = train_all[:, cols]; mu, sd = X.mean(0), X.std(0) + 1e-6
        Xz = (X - mu) / sd
        nn = NearestNeighbors(n_neighbors=min(K, len(X))).fit(Xz)
        S, Y = [], []
        for s in l1.subvideos(scene, 'test'):
            f, n = l1.subvideo_feats(det, s, key(s))
            lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
            d, _ = nn.kneighbors((f[:, cols] - mu) / sd)
            S.append(d.mean(1)[:n]); Y.append(lab)
        S = np.concatenate(S); Y = np.concatenate(Y)
        np.savez(os.path.join(a.out, f'beh_{sk}.npz'), scores=S, labels=Y)
        nm, an = S[Y == 0], S[Y == 1]
        print(f'[{sk}] 行为 合并kNN(+both)  (落盘 beh_{sk}.npz)')
        print(f'   normal {stats(nm)}')
        print(f'   anom   {stats(an)}')
        print(f'   anom/normal mean 比 = {an.mean() / nm.mean():.3f}', flush=True)
        if a.train:
            # train 帧在点云内,kNN 含自身(dist=0)→ 用 K+1 近邻去掉第 0 个(leave-self-out)
            dtr, _ = NearestNeighbors(n_neighbors=min(K + 1, len(X))).fit(Xz).kneighbors(Xz)
            STR = dtr[:, 1:].mean(1)
            np.savez(os.path.join(a.out, f'beh_{sk}_train.npz'), scores=STR)
            print(f'   train  {stats(STR)}  (CDF锚, leave-self-out → beh_{sk}_train.npz)', flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
