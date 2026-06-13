"""R1 — relational-descriptor kNN-OOD separability for the MOTION branch (validation-first).

Question this answers (the go/no-go for the motion branch): do EGO-INVARIANT pairwise relational
features — inter-vehicle distance / closing speed / collision-urgency over a short causal window —
separate anomalous from normal frames on the relational-anomaly target scenes (crossroads / vehicle /
highway), BEFORE we invest in any learned spatio-temporal network? If yes, build the model; if no,
rethink. Single-vehicle scene-relative anomalies (line-crossing) are OUT of L1 scope (need L2/L3).

INDEPENDENCE: touches NOTHING in the appearance branch (model/, train.py, eval.py) — it only uses the
local frozen detector+tracker (motion/detector.py) and never the RD teacher/student/loss. The two
branches share only the dataset and the eval protocol.

SEMANTIC ALIGNMENT (must match the appearance eval so the two branches' per-frame scores line up for
later fusion):
  * frames sorted by NUMERIC index (frame_idx), same as appearance eval;
  * one score PER FRAME (causal window [t-T+1, t] -> score assigned to frame t; partial window at the
    very start, never skip a frame);
  * labels read from the sibling <subvideo>.npy and aligned with `[-n:]` (n = #frames), identical to
    appearance eval;
  * per-scene MADViT protocol: train sub-videos -> normal manifold; test sub-videos -> per-frame
    kNN-OOD score; global-concat -> ONE AUC per scene.

Env: DMAD (ultralytics).  /home/lyx/miniconda3/envs/DMAD/bin/python motion/r1.py --scenes crossroads,vehicle,highway --gpu 0
"""
import argparse, os, glob, importlib.util, pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# Load the local detector.py by path (no package import; runs in the DMAD env that has ultralytics).
_spec = importlib.util.spec_from_file_location(
    'vehicle_detector', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector.py'))
_dm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dm)
build_vehicle_detector = _dm.build_vehicle_detector

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
APP_REF = {'crossroads': 59.69, 'vehicle': 60.71, 'highway': 92.69,  # appearance RD-topk (report) for context
           'bike': 98.08, 'farmland': 99.47, 'railway': 95.91, 'solar': 86.12}
T = 16            # causal window length (frames)
D0 = 0.12         # "close pair" distance threshold (fraction of frame)
MINLEN = 5        # drop tracks shorter than this (flicker)
FEATS = ['d_min', 'd_mean', 'n_close', 'closing_max', 'urgency_max']


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def track_centers(det, frame_paths, cache_key):
    """Track a sub-video -> per-frame {id: (cx,cy)} normalized to [0,1]; drop short (flicker) tracks.
    Cached to disk (tracking is the slow part) so re-runs of the descriptor/kNN are instant."""
    cp = os.path.join(CACHE, cache_key + '.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    raw, seen, wh = [], {}, None
    for img, d in det.track_clip(frame_paths):
        if wh is None:
            h, w = img.shape[:2]; wh = (w, h)
        fr = {}
        for (x1, y1, x2, y2), tid in zip(d['boxes'], d['ids']):
            tid = int(tid)
            fr[tid] = ((x1 + x2) / 2 / wh[0], (y1 + y2) / 2 / wh[1])
            seen[tid] = seen.get(tid, 0) + 1
        raw.append(fr)
    for fr in raw:                                  # remove flicker tracks
        for tid in [t for t in fr if seen[t] < MINLEN]:
            del fr[tid]
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, 'wb') as f:
        pickle.dump(raw, f)
    return raw


def descriptor(window):
    """window: list of up to T dicts {id:(cx,cy)} (window[-1] = current frame). Returns an
    EGO-INVARIANT relational vector (all pairwise -> camera translation cancels) + vehicle count
    (kept SEPARATE as a density baseline, NOT in the relational vector)."""
    cur = window[-1]; ids = list(cur); n = len(ids)
    if n < 2:                                       # no interaction possible -> "far / normal"
        return np.array([1., 1., 0., 0., 0.], np.float32), n
    P = {i: np.asarray(cur[i], np.float32) for i in ids}
    dd = [np.linalg.norm(P[ids[a]] - P[ids[b]]) for a in range(n) for b in range(a + 1, n)]
    dd = np.asarray(dd)
    closing_max = urg_max = 0.0
    last = len(window) - 1
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = ids[a], ids[b]
            f0 = next((k for k in range(last + 1) if ia in window[k] and ib in window[k]), None)
            if f0 is None or f0 == last:
                continue
            dp = np.linalg.norm(np.asarray(window[f0][ia]) - np.asarray(window[f0][ib]))
            dn = np.linalg.norm(P[ia] - P[ib])
            closing = (dp - dn) / (last - f0)        # >0 = approaching
            closing_max = max(closing_max, closing)
            if closing > 0:                          # collision urgency = closing rate / current gap;
                urg_max = max(urg_max, closing / max(dn, 0.01))   # floor gap so a near/exact hit peaks
    return np.array([float(dd.min()), float(dd.mean()), float((dd < D0).sum()),
                     float(closing_max), float(urg_max)], np.float32), n


def subvideo_feats(det, sub, cache_key):
    frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=frame_idx)
    raw = track_centers(det, frames, cache_key)
    feats, counts = [], []
    for t in range(len(raw)):
        f, n = descriptor(raw[max(0, t - T + 1): t + 1])    # causal window
        feats.append(f); counts.append(n)
    return np.asarray(feats, np.float32), np.asarray(counts, np.float32), len(frames)


def subvideos(scene, split):
    out = []
    for seq in sorted(glob.glob(os.path.join(DATA_ROOT, scene, 'sequence*'))):
        for sub in sorted(glob.glob(os.path.join(seq, split, '*'))):
            if os.path.isdir(sub):
                out.append(sub)
    return out


def run_scene(scene_key, det):
    scene = SCENES[scene_key]
    key = lambda sub: scene_key + '/' + os.path.relpath(sub, os.path.join(DATA_ROOT, scene)).replace('/', '__')
    # normal manifold from TRAIN
    train = [subvideo_feats(det, s, key(s))[0] for s in subvideos(scene, 'train')]
    train = np.concatenate(train) if train else np.zeros((0, len(FEATS)), np.float32)
    if len(train) < 50:
        print(f'[{scene_key}] too few train descriptors ({len(train)}) — scene likely vehicle-free, motion silent', flush=True)
        return None
    mu, sd = train.mean(0), train.std(0) + 1e-6
    nn = NearestNeighbors(n_neighbors=8).fit((train - mu) / sd)
    # TEST: per-frame kNN-OOD score
    per = []
    for s in subvideos(scene, 'test'):
        f, c, n = subvideo_feats(det, s, key(s))
        dist, _ = nn.kneighbors((f - mu) / sd)
        score = dist.mean(1)
        lab = np.asarray(np.load(s + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        per.append((os.path.relpath(s, os.path.join(DATA_ROOT, scene)), lab, score[:n], c[:n]))
    y = np.concatenate([p[1] for p in per]); s_rel = np.concatenate([p[2] for p in per])
    s_den = np.concatenate([p[3] for p in per])
    auc_rel = roc_auc_score(y, s_rel) * 100
    auc_den = roc_auc_score(y, s_den) * 100
    print(f'\n=== {scene_key} | {int(y.sum())}/{len(y)} anom frames | appearance RD-topk={APP_REF.get(scene_key,0):.1f} ===', flush=True)
    print(f'  RELATIONAL kNN-OOD AUC = {auc_rel:5.2f}   (density-count baseline = {auc_den:5.2f})', flush=True)
    for name, lab, sc, _ in per:                    # per-subvideo, to see WHERE relational fires
        if 0 < lab.sum() < len(lab):
            print(f'    {name:24s} relAUC={roc_auc_score(lab, sc)*100:5.2f}  ({int(lab.sum())}/{len(lab)} anom)', flush=True)
    return auc_rel, auc_den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='crossroads,vehicle,highway')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()
    det = build_vehicle_detector(conf=a.conf, device=f'cuda:{a.gpu}')
    rows = {}
    for sc in a.scenes.split(','):
        r = run_scene(sc, det)
        if r:
            rows[sc] = r
    print('\n=== R1 SUMMARY (relational kNN-OOD, ego-invariant, L1) ===', flush=True)
    print('scene'.ljust(12) + 'relAUC'.rjust(8) + 'density'.rjust(9) + 'appear'.rjust(8), flush=True)
    for sc in rows:
        print(sc.ljust(12) + f'{rows[sc][0]:8.2f}{rows[sc][1]:9.2f}{APP_REF.get(sc,0):8.2f}', flush=True)


if __name__ == '__main__':
    main()
