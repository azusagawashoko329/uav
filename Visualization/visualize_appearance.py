"""外观分支异常热力可视化 —— per-patch 重建误差(cos 距离)→ heatmap 叠到测试帧 → 每场景合成一个视频。

红 = 重建误差大 = 外观分支判为异常的地方(具体是哪些 patch 重建不出教师特征)。
用 code_dim=64 baseline 权重(`experiments/20260611_rd_baseline/ckpt/<scene>.pth`)。
每场景把该场景**所有 test 子视频按序拼成一个 mp4** → `Visualization/output/<scene>.mp4`(共 7 个)。

两遍:pass1 前向所有 test 帧、收集 per-patch 误差(定该场景全局色阶);pass2 读帧 → 叠热力 + 标注 → 写视频。
**去噪处理(关键)**:① 色阶下限取高分位(默认 p96),正常帧的中等误差 patch 直接透明;② 16×16 误差图高斯平滑,
孤立单 patch 噪点被邻域稀释、连续异常区保留;③ 硬阈值 + 陡 alpha,低于阈值完全透明。
Env: DMAD(HF_HUB_OFFLINE=1)。例: HF_HUB_OFFLINE=1 python Visualization/visualize_appearance.py --scenes solar --gpu 0
"""
import os, sys, glob, argparse, math
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from model import build_appearance_teacher, build_bottleneck, build_student
from model.loss import _per_patch_cosine_dist

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
          'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
          'vehicle': 'Vehicle Roundabout'}
OUT = os.path.join(_ROOT, 'Visualization', 'output')
MAXW = 960   # 输出视频最大宽度(保持比例缩放)


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def test_subvideos(scene):
    return sorted(d for d in glob.glob(os.path.join(DATA_ROOT, scene, 'sequence*', 'test', '*')) if os.path.isdir(d))


@torch.no_grad()
def heat_and_score(teacher, ocbe, student, xb, k):
    """xb:(B,3,img,img) → per-patch 误差 (B,P) 跨 tap 平均 + topk 帧分 (B,)。"""
    taps = teacher(xb)['patches']
    recon = student(ocbe(taps))
    acc = 0.0
    score = 0.0
    for A, Ahat in zip(taps, recon):
        d = _per_patch_cosine_dist(A, Ahat)             # (B,P)
        acc = acc + d
        score = score + d.topk(k, dim=-1).values.mean(-1)
    n = len(taps)
    return (acc / n).float().cpu().numpy(), (score / n).float().cpu().numpy()


def run_scene(sk, exp, dev, a):
    scene = SCENES[sk]
    ck = torch.load(os.path.join(_ROOT, 'experiments', exp, 'ckpt', f'{sk}.pth'), map_location=dev)
    cfg = ck['cfg']
    taps, code_dim, depth, img, k = cfg['taps'], cfg['code_dim'], cfg['depth'], cfg['img'], cfg['k']
    teacher = build_appearance_teacher(layers=taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=len(taps), code_dim=code_dim,
                            use_tap_norm=not cfg.get('no_tap_norm', False)).to(dev).eval()
    student = build_student(n_taps=len(taps), code_dim=code_dim, depth=depth).to(dev).eval()
    ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])

    subs = test_subvideos(scene)
    metas, dmaps, scores, labels = [], [], [], []      # pass1 收集
    for sub in subs:
        frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=frame_idx)
        n = len(frames)
        lab = np.asarray(np.load(sub + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        sub_name = f"{os.path.basename(os.path.dirname(os.path.dirname(sub)))}/{os.path.basename(sub)}"
        for i in range(0, n, a.bs):
            chunk = frames[i:i + a.bs]
            xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in chunk]).to(dev)
            dm, sc = heat_and_score(teacher, ocbe, student, xb, k)
            for j, f in enumerate(chunk):
                metas.append((f, sub_name)); dmaps.append(dm[j]); scores.append(float(sc[j]))
            labels.extend(lab[i:i + len(chunk)].tolist())
        print(f'  [{sk}] pass1 {sub_name}: {n} frames', flush=True)

    alld = np.stack(dmaps)                              # (N,P)
    P = alld.shape[1]; hw = int(round(math.sqrt(P)))
    vmin, vmax = np.percentile(alld, [a.pmin, a.pmax])   # 高分位下限 → 正常帧中等误差 patch 透明
    print(f'  [{sk}] {len(metas)} frames, patch {hw}x{hw}, color range p{a.pmin}={vmin:.4f} p{a.pmax}={vmax:.4f}', flush=True)

    h0, w0 = cv2.imread(metas[0][0]).shape[:2]
    scale = min(1.0, MAXW / w0)
    OW, OH = int(round(w0 * scale)), int(round(h0 * scale))
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f'{sk}.mp4')
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (OW, OH))

    for idx, (fpath, sub_name) in enumerate(metas):
        img_bgr = cv2.resize(cv2.imread(fpath), (OW, OH))
        dm = dmaps[idx].reshape(hw, hw).astype(np.float32)
        dm = cv2.GaussianBlur(dm, (3, 3), 0.8)                          # ① 16x16 平滑:压孤立单 patch 噪点
        norm = np.clip((dm - vmin) / (vmax - vmin + 1e-9), 0, 1)
        norm[norm < a.thresh] = 0.0                                     # ② 硬阈值:低误差完全透明
        norm_up = np.clip(cv2.resize(norm, (OW, OH), interpolation=cv2.INTER_CUBIC), 0, 1)
        norm_up = cv2.GaussianBlur(norm_up, (0, 0), max(1.0, OW / 180)) # 上采样后柔化边界
        heat = cv2.applyColorMap((norm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = (np.clip(norm_up, 0, 1) ** a.gamma * a.alpha_max)[..., None]   # ③ 陡 alpha:只有强信号显红
        out = (img_bgr * (1 - alpha) + heat * alpha).astype(np.uint8)

        gt = labels[idx]; sc = scores[idx]
        tag = 'ANOMALY' if gt == 1 else 'normal'
        col = (0, 0, 255) if gt == 1 else (0, 200, 0)
        cv2.rectangle(out, (0, 0), (OW - 1, OH - 1), col, 3 if gt == 1 else 1)
        cv2.putText(out, f'{sk} | {sub_name} | f{idx}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f'GT:{tag}  score:{sc:.3f}', (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(out, 'red heat = high recon error (appearance anomaly)', (8, OH - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(out)
    vw.release()
    print(f'  [{sk}] saved {outp}  ({OW}x{OH}, {len(metas)} frames @ {a.fps}fps)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default=','.join(SCENES))
    ap.add_argument('--exp', default='20260611_rd_baseline')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--pmin', type=float, default=96.0, help='色阶下限分位(越高正常帧越干净)')
    ap.add_argument('--pmax', type=float, default=99.8, help='色阶上限分位')
    ap.add_argument('--thresh', type=float, default=0.12, help='归一化误差硬阈值,低于此完全透明')
    ap.add_argument('--gamma', type=float, default=1.8, help='alpha 陡度(越大低值越透明)')
    ap.add_argument('--alpha_max', type=float, default=0.65)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    for sk in a.scenes.split(','):
        print(f'=== {sk} ===', flush=True)
        run_scene(sk, a.exp, dev, a)


if __name__ == '__main__':
    main()
