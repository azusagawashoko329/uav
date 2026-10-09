"""统一训练脚本 —— 一个环境(DMAD: timm+ultralytics)、一个脚本顺序产出两分支产物,固定 seed 可复现。

  外观分支(梯度训练):冻结 DINOv3 teacher → 训 OCBE+student(top-k 重建损失,超参同 train.py)
                       → experiments/<exp>/ckpt/<scene>/<scene>.pth   {ocbe, student, cfg, scene}
  行为分支(无训练,拟合正常点云状态):冻结 YOLOv8x-VisDrone **从头重跑检测**(不读 behavior/cache,
                       检测结果只缓存到本实验 experiments/<exp>/det_cache/)→ 5维描述子 → logz 标准化
                       + 检测质量 q → experiments/<exp>/ckpt/<scene>/<scene>.npz
                       {Xtr_proc, mu, sd, q, jitter, K, logz_cols, D0, R0, feats}

产物按场景分目录:ckpt/<scene>/ 同时存放该场景的权重(.pth)与点云状态(.npz)。每场景训练前 set_seed
(默认 42)→ 单场景可复现。复用 model/ 与 behavior/l1.py 组件,不修改任何已有文件。Env: DMAD。
例:
  HF_HUB_OFFLINE=1 python fusion/train_fusion.py --scenes railway --gpu 0 --seed 42
"""
import argparse, os, glob, time, sys, importlib.util, pickle, datetime, random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scripts.experiment_paths import require_experiment
from model import build_appearance_teacher, build_bottleneck, build_student, topk_loss
import train as _train                      # 复用 FrameDataset / SCENES / DATA_ROOT(顶层只 import model+torch,安全)
_spec = importlib.util.spec_from_file_location('l1', os.path.join(_ROOT, 'behavior', 'l1.py'))
l1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(l1)   # 触发 ultralytics(DMAD 有)

ALL = ['highway', 'farmland', 'bike', 'solar', 'vehicle', 'crossroads', 'railway']
LOGZ_COLS = [2, 3, 4]                        # n_close / count / max_local 取 log1p 压重尾(见 PIPELINE §2.4)
BATCH = l1.BATCH


def set_seed(seed):
    """固定全部随机源 → 单场景训练可复现。cudnn.deterministic 对 ViT(无卷积)开销很小。"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(wid):
    s = (torch.initial_seed() + wid) % 2 ** 32
    np.random.seed(s); random.seed(s)


def logz_fit(X):
    """X(N×5 原始描述子)→ (标准化后点云, mu, sd)。计数维先 log1p,再全维 z-score。"""
    X2 = X.astype(float).copy()
    for i in LOGZ_COLS:
        X2[:, i] = np.log1p(X2[:, i])
    mu, sd = X2.mean(0), X2.std(0) + 1e-6
    return (X2 - mu) / sd, mu, sd


def fresh_feats(det, sub, cache_dir):
    """从头检测一个子视频(不读 behavior/cache);检测中心只缓存到本实验 cache_dir(首次必跑检测)。
    逻辑与 l1.subvideo_centers 一致(同检测器、同 agnostic-NMS),保证描述子可复现。"""
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


# ----------------------------------------------------------------------------- 外观分支(策略不变)
def train_appearance(scene_key, teacher, dev, a):
    target = os.path.join('experiments', a.exp, 'ckpt', scene_key, f'{scene_key}.pth')
    if os.path.exists(target):
        raise FileExistsError(f'checkpoint already exists: {target}; use a new experiment ID')
    scene = l1.SCENES[scene_key]
    ocbe = build_bottleneck(n_taps=len(a.taps), code_dim=a.code_dim, use_tap_norm=not a.no_tap_norm).to(dev)
    student = build_student(n_taps=len(a.taps), code_dim=a.code_dim, depth=a.depth).to(dev)
    ds = _train.FrameDataset(os.path.join(_train.DATA_ROOT, scene), a.img, teacher.img_mean, teacher.img_std)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, worker_init_fn=seed_worker, generator=g)
    params = list(ocbe.parameters()) + list(student.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=len(dl), epochs=a.epochs)
    print(f'  [外观] {len(ds)} frames, {len(dl)} batches/ep | img={a.img} taps={a.taps} k={a.k} '
          f'ep={a.epochs} bs={a.bs} lr={a.lr} wd={a.wd} seed={a.seed}', flush=True)
    for ep in range(a.epochs):
        ocbe.train(); student.train(); tot = n = 0; t0 = time.time()
        for bi, x in enumerate(tqdm(dl, desc=f'ep{ep+1}/{a.epochs}', leave=False, mininterval=2.0)):
            if a.max_steps and bi >= a.max_steps:
                break
            x = x.to(dev, non_blocking=True)
            with torch.no_grad():
                taps = teacher(x)['patches']
            loss = topk_loss(taps, student(ocbe(taps)), k=a.k)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        print(f'    ep{ep+1:02d}/{a.epochs} loss={tot/max(n,1):.4f} lr={sched.get_last_lr()[0]:.2e} '
              f'({time.time()-t0:.0f}s)', flush=True)
    cfg = dict(taps=a.taps, code_dim=a.code_dim, depth=a.depth, img=a.img, k=a.k,
               no_tap_norm=a.no_tap_norm, loss='topk', seed=a.seed)
    out = os.path.join('experiments', a.exp, 'ckpt', scene_key); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f'{scene_key}.pth')
    if os.path.exists(p):
        raise FileExistsError(f'checkpoint already exists: {p}; use a new experiment ID')
    torch.save({'ocbe': ocbe.state_dict(), 'student': student.state_dict(), 'cfg': cfg, 'scene': scene}, p)
    print(f'  [外观] saved {p}', flush=True)


# ----------------------------------------------------------------------------- 行为分支(从头检测)
def fit_behavior(scene_key, det, a):
    target = os.path.join('experiments', a.exp, 'ckpt', scene_key, f'{scene_key}.npz')
    if os.path.exists(target):
        raise FileExistsError(f'behavior state already exists: {target}; use a new experiment ID')
    scene = l1.SCENES[scene_key]
    cache_dir = os.path.join('experiments', a.exp, 'det_cache')
    t0 = time.time()
    trsubs = [fresh_feats(det, s, cache_dir)[0] for s in l1.subvideos(scene, 'train')]
    Xtr = np.concatenate(trsubs)
    if len(Xtr) < 50:
        print(f'  [行为] {scene_key} 描述子<50,无车场景,跳过(融合时行为静默)', flush=True); return
    dn = np.concatenate([np.abs(np.diff(f[:, 3])) for f in trsubs if len(f) > 1])   # seq 内车数差分
    nn = np.concatenate([f[:, 3] for f in trsubs])
    jit = float(dn.mean() / (nn.mean() + 1e-9)); q = float(min(1.0, 0.1 / (jit + 1e-9)))
    Xproc, mu, sd = logz_fit(Xtr)
    out = os.path.join('experiments', a.exp, 'ckpt', scene_key); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f'{scene_key}.npz')
    if os.path.exists(p):
        raise FileExistsError(f'behavior state already exists: {p}; use a new experiment ID')
    np.savez(p, Xtr_proc=Xproc, mu=mu, sd=sd, q=q, jitter=jit, K=l1.K,
             logz_cols=np.array(LOGZ_COLS), D0=l1.D0, R0=l1.R0, feats=np.array(l1.FEATS))
    print(f'  [行为] {len(Xtr)} 正常描述子(从头检测 {time.time()-t0:.0f}s)| jitter={jit:.3f} q={q:.3f} | saved {p}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='all', help='all 或 逗号分隔(railway,highway,...)')
    ap.add_argument('--only', default='both', choices=['both', 'appearance', 'behavior'])
    ap.add_argument('--exp', required=True, help='registered da/fusion/<id> path')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--taps', type=int, nargs='+', default=[5, 7, 9, 11])
    ap.add_argument('--code_dim', type=int, default=64)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--no_tap_norm', action='store_true')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--conf', type=float, default=0.25, help='检测置信阈(行为分支)')
    ap.add_argument('--max_steps', type=int, default=0, help='>0 限制每 epoch batch 数(smoke test)')
    a = ap.parse_args()
    require_experiment(a.exp, dataset='da', family='fusion')
    scenes = ALL if a.scenes == 'all' else a.scenes.split(',')
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')

    teacher = det = None
    if a.only in ('both', 'appearance'):
        teacher = build_appearance_teacher(layers=a.taps).to(dev).eval()
    if a.only in ('both', 'behavior'):
        det = l1.build_vehicle_detector(conf=a.conf, device=f'cuda:{a.gpu}')

    for sk in scenes:
        set_seed(a.seed)                      # 每场景从同一 seed 出发 → 可复现
        print(f'\n===== {sk} (seed={a.seed}) =====', flush=True)
        if a.only in ('both', 'appearance'):
            train_appearance(sk, teacher, dev, a)
        if a.only in ('both', 'behavior'):
            fit_behavior(sk, det, a)
    print(f'\n完成。产物:experiments/{a.exp}/ckpt/<scene>/<scene>.pth(外观)+ .npz(行为)', flush=True)


if __name__ == '__main__':
    main()
