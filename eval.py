"""Eval the reverse-distillation pipeline per SCENE — MADViT global-concat protocol.

For each scene with a trained checkpoint: enumerate ALL test sub-videos across ALL sequences, score
every frame (NUMERIC order, aligned to the .npy labels) by the SAME reconstruction discrepancy used
in training, then CONCAT all videos' per-frame scores+labels -> ONE AUC per scene (roc_auc_score;
higher score = more anomalous). Headline read-out is 'topk' (k = training k, the training-consistent
score); 'max'/'mean' are read-out variants of the identical discrepancy.

Loads experiments/<exp>/ckpt/<scene>.pth, writes experiments/<exp>/results.csv.
Example:  python eval.py --exp 20260611_rd_baseline --gpu 0
"""
import argparse, os, glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve

from model import build_appearance_teacher, build_bottleneck, build_student, anomaly_readouts

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
READOUTS = ['topk', 'max', 'mean']
ALL_SCENES = 'bike,crossroads,farmland,highway,railway,solar,vehicle'


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


@torch.no_grad()
def score_video(teacher, ocbe, student, vid_dir, tf, dev, k, bs):
    frames = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')), key=frame_idx)   # numeric order -> aligns labels
    out = {r: [] for r in READOUTS}
    for i in range(0, len(frames), bs):
        xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in frames[i:i + bs]]).to(dev)
        taps = teacher(xb)['patches']
        sc = anomaly_readouts(taps, student(ocbe(taps)), k=k)
        for r in READOUTS:
            out[r].append(sc[r].cpu().numpy())
    return {r: np.concatenate(out[r]) for r in READOUTS}, len(frames)


def run_scene(scene_key, exp, dev, bs):
    scene = SCENES.get(scene_key, scene_key)
    ckpt_path = os.path.join('experiments', exp, 'ckpt', f'{scene_key}.pth')
    if not os.path.exists(ckpt_path):
        return None
    ck = torch.load(ckpt_path, map_location=dev)
    cfg = ck['cfg']
    taps, code_dim, depth, img, k = cfg['taps'], cfg['code_dim'], cfg['depth'], cfg['img'], cfg['k']
    teacher = build_appearance_teacher(layers=taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=len(taps), code_dim=code_dim).to(dev).eval()
    student = build_student(n_taps=len(taps), code_dim=code_dim, depth=depth).to(dev).eval()
    ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])

    scene_dir = os.path.join(DATA_ROOT, scene)
    vids = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'test', '*')) if os.path.isdir(d))
    agg = {r: [] for r in READOUTS}; labels = []
    for vp in vids:
        sc, n = score_video(teacher, ocbe, student, vp, tf, dev, k, bs)
        for r in READOUTS:
            agg[r].append(sc[r][:n])
        lab = np.load(vp + '.npy', allow_pickle=True)            # sibling <subvideo>.npy
        labels.append(np.asarray(lab).astype(int).ravel()[-n:])
    y = np.concatenate(labels)
    res = {r: (roc_auc_score(y, np.concatenate(agg[r])) * 100, eer_of(y, np.concatenate(agg[r]))) for r in READOUTS}
    return res, int(y.sum()), len(y), len(vids), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--scenes', default=ALL_SCENES)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    scenes = a.scenes.split(',')

    rows = {}
    for sc in scenes:
        r = run_scene(sc, a.exp, dev, a.bs)
        if r is None:
            print(f'[{sc}] no ckpt yet, skip', flush=True); continue
        res, npos, ntot, nvid, k = r
        rows[sc] = res
        print(f'[{sc}] {nvid} test vids, {npos}/{ntot} anom, k={k} | '
              + ' '.join(f'{rd}={res[rd][0]:.2f}(eer{res[rd][1]:.3f})' for rd in READOUTS), flush=True)

    outp = os.path.join('experiments', a.exp, 'results.csv')
    with open(outp, 'w') as f:
        f.write('scene,' + ','.join(f'{r}_AUC,{r}_EER' for r in READOUTS) + '\n')
        for sc in scenes:
            if sc in rows:
                f.write(sc + ',' + ','.join(f'{rows[sc][r][0]:.2f},{rows[sc][r][1]:.3f}' for r in READOUTS) + '\n')
    print(f'\nwrote {outp}', flush=True)
    print('\n=== AUC (%) | headline = topk (training-consistent) ===')
    print('scene'.ljust(12) + ''.join(r.rjust(10) for r in READOUTS))
    for sc in scenes:
        if sc in rows:
            print(sc.ljust(12) + ''.join(f'{rows[sc][r][0]:10.2f}' for r in READOUTS))


if __name__ == '__main__':
    main()
