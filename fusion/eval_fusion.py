"""统一评分 + 融合脚本 —— 读 train_fusion.py 的两分支产物(同在 ckpt/),出每场景 concat-micro AUC
(纯外观 vs 融合)。test 检测同样**从头重跑**(只缓存到本实验 det_cache/,不读 behavior/cache)。

逐 test 子视频严格对齐(同一 sub 同时出外观/行为分,labels[-n:]):
  外观分 sa : 加载 ckpt/<scene>.pth → eval.score_video 的 'topk' 读出(每帧)。
  行为分 sb : 加载 ckpt/<scene>.npz(Xtr_proc/mu/sd/q)→ 检测 → 5维描述子 → logz(存的 mu/sd)
              → 到正常点云 K 近邻平均距离。无 npz(无车场景)→ 行为静默,融合=纯外观。
  融合      : p=test 自身平均秩;门控 w = 0.5·q·(1−2|p_app−0.5|);fused=(1−w)·p_app + w·p_beh。

Env: DMAD。例: HF_HUB_OFFLINE=1 python fusion/eval_fusion.py --exp 20260617_all_model --gpu 0
"""
import argparse, os, sys, glob, pickle, importlib.util, datetime
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scripts.experiment_paths import require_experiment
from model import build_appearance_teacher, build_bottleneck, build_student
import eval as _eval                          # 复用 score_video
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)

ALL = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
BATCH = l1.BATCH


def rank(s):
    srt = np.sort(s); lo = np.searchsorted(srt, s, 'left'); hi = np.searchsorted(srt, s, 'right')
    return (lo + hi) / 2.0 / len(s)


def logz_apply(X, mu, sd, cols):
    X2 = X.astype(float).copy()
    for i in cols:
        X2[:, i] = np.log1p(X2[:, i])
    return (X2 - mu) / sd


def fresh_feats(det, sub, cache_dir):
    """从头检测 test 子视频(不读 behavior/cache);只缓存到本实验 det_cache。与 train_fusion 同款。"""
    rel = os.path.relpath(sub, l1.DATA_ROOT)             # "<scene>/sequenceX/<split>/YY"
    scene, rest = rel.split(os.sep, 1)
    sub_dir = os.path.join(cache_dir, scene); os.makedirs(sub_dir, exist_ok=True)   # 按场景分文件夹
    cp = os.path.join(sub_dir, rest.replace(os.sep, '__') + '.pkl')
    frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=l1.frame_idx)
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            cents = pickle.load(f)
    else:
        h, w = cv2.imread(frames[0]).shape[:2]
        cents = []
        for i in tqdm(range(0, len(frames), BATCH), desc=f'检测 {os.path.basename(sub)}', leave=False, mininterval=2.0):
            for d in det.detect(frames[i:i + BATCH]):
                b = d['boxes']
                c = (np.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h], 1).astype(np.float32)
                     if len(b) else np.zeros((0, 2), np.float32))
                cents.append(c)
        with open(cp, 'wb') as f:
            pickle.dump(cents, f)
    feats = np.stack([l1.descriptor(c) for c in cents]) if cents else np.zeros((0, 5), np.float32)
    return feats, len(frames)


def load_appearance(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev); cfg = ck['cfg']
    taps, code_dim, depth, img, k = cfg['taps'], cfg['code_dim'], cfg['depth'], cfg['img'], cfg['k']
    teacher = build_appearance_teacher(layers=taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=len(taps), code_dim=code_dim, use_tap_norm=not cfg.get('no_tap_norm', False)).to(dev).eval()
    student = build_student(n_taps=len(taps), code_dim=code_dim, depth=depth).to(dev).eval()
    ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])
    return teacher, ocbe, student, tf, k


def run_scene(sk, exp, det, dev, bs):
    ckpt = os.path.join('experiments', exp, 'ckpt', sk, f'{sk}.pth')
    if not os.path.exists(ckpt):
        return None
    teacher, ocbe, student, tf, k = load_appearance(ckpt, dev)
    scene = l1.SCENES[sk]
    cache_dir = os.path.join('experiments', exp, 'det_cache')
    bp = os.path.join('experiments', exp, 'ckpt', sk, f'{sk}.npz')   # 行为产物(同场景目录);不存在=无车静默
    beh = np.load(bp) if os.path.exists(bp) else None
    nn = q = cols = mu = sd = None
    if beh is not None:
        nn = NearestNeighbors(n_neighbors=int(beh['K'])).fit(beh['Xtr_proc'])
        q, cols, mu, sd = float(beh['q']), list(beh['logz_cols']), beh['mu'], beh['sd']

    SA, SB, Y = [], [], []
    for sub in l1.subvideos(scene, 'test'):
        sc, n = _eval.score_video(teacher, ocbe, student, sub, tf, dev, k, bs)
        sa = sc['topk'][:n]
        lab = np.asarray(np.load(sub + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        if beh is not None:
            feats, _ = fresh_feats(det, sub, cache_dir)
            sb = nn.kneighbors(logz_apply(feats[:n], mu, sd, cols))[0].mean(1)
        else:
            sb = np.zeros(n)
        SA.append(sa); SB.append(sb); Y.append(lab)
    sa, sb, y = np.concatenate(SA), np.concatenate(SB), np.concatenate(Y)
    auc_a = roc_auc_score(y, sa) * 100
    if beh is None:
        return auc_a, float('nan'), 0.0, auc_a, y
    pa, pb = rank(sa), rank(sb)
    auc_b = roc_auc_score(y, sb) * 100
    w = 0.5 * q * (1 - 2 * np.abs(pa - 0.5))
    fused = (1 - w) * pa + w * pb
    auc_f = roc_auc_score(y, fused) * 100
    return auc_a, auc_b, q, auc_f, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='registered da/fusion/<id> path')
    ap.add_argument('--scenes', default='all')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()
    require_experiment(a.exp, dataset='da', family='fusion')
    outp = os.path.join('experiments', a.exp, 'results_fusion.csv')
    if os.path.exists(outp):
        raise FileExistsError(f'result already exists: {outp}; use a new run ID or experiment')
    scenes = ALL if a.scenes == 'all' else a.scenes.split(',')
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    det = l1.build_vehicle_detector(conf=a.conf, device=f'cuda:{a.gpu}')

    print(f'{"scene":>11s}{"外观":>9s}{"行为":>9s}{"q":>7s}{"融合":>9s}{"Δ":>8s}')
    A, Fz, rows = [], [], []
    for sk in scenes:
        r = run_scene(sk, a.exp, det, dev, a.bs)
        if r is None:
            print(f'{sk:>11s}   (无 ckpt,跳过)'); continue
        auc_a, auc_b, q, auc_f, _ = r
        A.append(auc_a); Fz.append(auc_f); rows.append((sk, auc_a, auc_b, q, auc_f))
        bsv = f'{auc_b:9.2f}' if auc_b == auc_b else f'{"--":>9s}'
        print(f'{sk:>11s}{auc_a:9.2f}{bsv}{q:7.3f}{auc_f:9.2f}{auc_f-auc_a:+8.2f}', flush=True)
    if A:
        print(f'{"均值":>10s}{np.mean(A):9.2f}{"":9s}{"":7s}{np.mean(Fz):9.2f}{np.mean(Fz)-np.mean(A):+8.2f}')
        with open(outp, 'w') as f:
            f.write('scene,appearance,behavior,q,fused\n')
            for sk, aa, ab, q, af in rows:
                f.write(f'{sk},{aa:.2f},{ab:.2f},{q:.3f},{af:.2f}\n')
        print(f'\nwrote {outp}', flush=True)


if __name__ == '__main__':
    main()
