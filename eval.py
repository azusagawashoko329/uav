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

from model import (build_appearance_teacher, build_bottleneck, build_student, build_adapter,
                   anomaly_readouts, anomaly_readouts_weighted)

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
READOUTS = ['topk', 'max', 'mean', 'flat']   # 'flat' = MADViT flatten-cosine (matched score for --loss full)
ALL_SCENES = 'bike,crossroads,farmland,highway,railway,solar,vehicle'


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


@torch.no_grad()
def score_video(teacher, ocbe, student, vid_dir, tf, dev, k, bs, adapter=None):
    frames = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')), key=frame_idx)   # numeric order -> aligns labels
    out = {r: [] for r in READOUTS}
    for i in range(0, len(frames), bs):
        xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in frames[i:i + bs]]).to(dev)
        taps = teacher(xb)['patches']
        code = ocbe([adapter.aggregate(taps)]) if (adapter is not None and adapter.input_agg) else ocbe(taps)
        recon = student(code)
        sc = (anomaly_readouts_weighted(taps, recon, adapter.w, k=k) if adapter is not None
              else anomaly_readouts(taps, recon, k=k))
        for r in READOUTS:
            out[r].append(sc[r].cpu().numpy())
    return {r: np.concatenate(out[r]) for r in READOUTS}, len(frames)


def run_scene(scene_key, exp, dev, bs, agg='concat', base='experiments'):
    scene = SCENES.get(scene_key, scene_key)
    ckpt_path = os.path.join(base, exp, 'ckpt', f'{scene_key}.pth')
    if not os.path.exists(ckpt_path):
        return None
    ck = torch.load(ckpt_path, map_location=dev)
    cfg = ck['cfg']
    code_dim, depth, img, k = cfg['code_dim'], cfg['depth'], cfg['img'], cfg['k']
    use_tap_norm = not cfg.get('no_tap_norm', False)             # baseline ckpts lack the key -> True
    is_laa = cfg.get('tap_adapter', False)
    inp_agg = cfg.get('adapter_input', False)                    # variant A: w also aggregates OCBE input
    # LAA: tap all 12, student reconstructs all 12, learned-weight scoring; baseline: the fixed cfg['taps'].
    taps = list(range(12)) if is_laa else cfg['taps']
    teacher = build_appearance_teacher(layers=taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=(1 if inp_agg else len(taps)), code_dim=code_dim, use_tap_norm=use_tap_norm).to(dev).eval()
    student = build_student(n_taps=len(taps), code_dim=code_dim, depth=depth).to(dev).eval()
    ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
    adapter = None
    if is_laa:
        adapter = build_adapter(n_layers=len(taps), embed_dim=teacher.embed_dim, input_agg=inp_agg).to(dev).eval()
        adapter.load_state_dict(ck['adapter'])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])

    scene_dir = os.path.join(DATA_ROOT, scene)
    vids = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'test', '*')) if os.path.isdir(d))
    per_scores = {r: [] for r in READOUTS}; per_labels = []
    for vp in vids:
        sc, n = score_video(teacher, ocbe, student, vp, tf, dev, k, bs, adapter=adapter)
        for r in READOUTS:
            per_scores[r].append(sc[r][:n])
        lab = np.load(vp + '.npy', allow_pickle=True)            # sibling <subvideo>.npy
        per_labels.append(np.asarray(lab).astype(int).ravel()[-n:])
    y = np.concatenate(per_labels)
    if agg == 'mean':                                            # ASTT-style: per-video AUC, then average (macro)
        res = {}
        for r in READOUTS:
            pv = [(roc_auc_score(l, s) * 100, eer_of(l, s)) for s, l in zip(per_scores[r], per_labels)
                  if 0 < l.sum() < len(l)]                        # skip single-class videos (AUC undefined)
            res[r] = (float(np.mean([a for a, _ in pv])), float(np.mean([e for _, e in pv])))
    else:                                                        # concat (micro) — the MADViT-matched protocol
        res = {r: (roc_auc_score(y, np.concatenate(per_scores[r])) * 100,
                   eer_of(y, np.concatenate(per_scores[r]))) for r in READOUTS}
    return res, int(y.sum()), len(y), len(vids), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--base', default='experiments', help='experiment root dir (e.g. experiment_LAA)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--scenes', default=ALL_SCENES)
    ap.add_argument('--agg', default='concat', choices=['concat', 'mean'],
                    help="concat = global micro-AUC (MADViT-matched); mean = per-video AUC averaged (ASTT-style macro)")
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    scenes = a.scenes.split(',')

    rows = {}
    for sc in scenes:
        r = run_scene(sc, a.exp, dev, a.bs, agg=a.agg, base=a.base)
        if r is None:
            print(f'[{sc}] no ckpt yet, skip', flush=True); continue
        res, npos, ntot, nvid, k = r
        rows[sc] = res
        print(f'[{sc}] ({a.agg}) {nvid} test vids, {npos}/{ntot} anom, k={k} | '
              + ' '.join(f'{rd}={res[rd][0]:.2f}(eer{res[rd][1]:.3f})' for rd in READOUTS), flush=True)

    outp = os.path.join(a.base, a.exp, 'results.csv' if a.agg == 'concat' else f'results_{a.agg}.csv')
    with open(outp, 'w') as f:
        f.write('scene,' + ','.join(f'{r}_AUC,{r}_EER' for r in READOUTS) + '\n')
        for sc in scenes:
            if sc in rows:
                f.write(sc + ',' + ','.join(f'{rows[sc][r][0]:.2f},{rows[sc][r][1]:.3f}' for r in READOUTS) + '\n')
    print(f'\nwrote {outp}', flush=True)
    print(f'\n=== AUC (%) | agg={a.agg} | headline = topk (training-consistent) ===')
    print('scene'.ljust(12) + ''.join(r.rjust(10) for r in READOUTS))
    for sc in scenes:
        if sc in rows:
            print(sc.ljust(12) + ''.join(f'{rows[sc][r][0]:10.2f}' for r in READOUTS))


if __name__ == '__main__':
    main()
