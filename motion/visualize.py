"""Render BoT-SORT vehicle tracking over ALL sub-videos of each scene into ONE mp4 per scene,
for visual inspection of tracking quality (ID stability, trail continuity, ID switches, miss/FP).

Per scene: concatenate every sub-video (all sequences, train+test) into one video, in sequence /
train-before-test / numeric order. The tracker is RESET per sub-video (clean per-clip IDs, no ghost
tracks across cuts); a top banner marks scene/seq/split/clip/frame so the position in the stream is
always clear. Each box shows trackID:class; a trail shows the recent path of every visible track.

Needs the env with ultralytics (DMAD):
  /home/lyx/miniconda3/envs/DMAD/bin/python motion/visualize.py --scenes crossroads --gpu 0
"""
import argparse, os, glob, importlib.util
from collections import deque, defaultdict
import numpy as np
import cv2

# Load the local detector.py by path (independent of the appearance branch; runs in the DMAD env).
_spec = importlib.util.spec_from_file_location(
    'vehicle_detector', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector.py'))
_dm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dm)
build_vehicle_detector, VEHICLE_CLASSES = _dm.build_vehicle_detector, _dm.VEHICLE_CLASSES

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def color_for(i):
    """Deterministic bright BGR color per track id (stable within a clip)."""
    h = (int(i) * 2654435761) & 0xFFFFFFFF
    return (80 + h % 176, 80 + (h >> 8) % 176, 80 + (h >> 16) % 176)


def list_clips(scene_dir):
    """All sub-videos: sequence order, train before test, numeric sub-video & frame order."""
    clips = []
    for seq in sorted(glob.glob(os.path.join(scene_dir, 'sequence*'))):
        for split in ('train', 'test'):
            for sub in sorted(glob.glob(os.path.join(seq, split, '*'))):
                if not os.path.isdir(sub):
                    continue
                frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=frame_idx)
                if frames:
                    clips.append((f'{os.path.basename(seq)}/{split}/{os.path.basename(sub)}', frames))
    return clips


def draw(img, det, history, banner, fidx):
    h, w = img.shape[:2]
    cur = []
    for (x1, y1, x2, y2), tid, cls in zip(det['boxes'], det['ids'], det['classes']):
        tid = int(tid)
        c = color_for(tid)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        history[tid].append((fidx, int((x1 + x2) / 2), int((y1 + y2) / 2)))
        cur.append(tid)
        cv2.putText(img, f'{tid}:{VEHICLE_CLASSES.get(int(cls), "?")}', (int(x1), max(11, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    for tid in cur:                                  # trail of each visible track, BROKEN on gaps>2 frames
        seg, prevf = [], None
        for f, x, y in history[tid]:
            if prevf is not None and f - prevf > 2:
                if len(seg) >= 2:
                    cv2.polylines(img, [np.array(seg, np.int32)], False, color_for(tid), 2, cv2.LINE_AA)
                seg = []
            seg.append((x, y)); prevf = f
        if len(seg) >= 2:
            cv2.polylines(img, [np.array(seg, np.int32)], False, color_for(tid), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, f'{banner}  tracks={len(cur)}', (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_scene(scene_key, gpu, out_dir, fps, trail, max_clips):
    scene = SCENES[scene_key]
    clips = list_clips(os.path.join(DATA_ROOT, scene))
    if max_clips:
        clips = clips[:max_clips]
    det = build_vehicle_detector(conf=0.1, device=f'cuda:{gpu}')   # low floor → BoT-SORT recovery
    h0, w0 = cv2.imread(clips[0][1][0]).shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f'{scene_key}.mp4')
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w0, h0))
    total = 0
    for ci, (label, frames) in enumerate(clips):
        history = defaultdict(lambda: deque(maxlen=trail))
        n = len(frames)
        for i, (img, d) in enumerate(det.track_clip(frames)):
            img = np.ascontiguousarray(img)
            draw(img, d, history, f'{scene_key} | {label} | {i + 1}/{n}', i)
            vw.write(img)
            total += 1
        print(f'[{scene_key}] clip {ci + 1}/{len(clips)} {label} ({n}f)', flush=True)
    vw.release()
    print(f'[{scene_key}] WROTE {outp}  ({total} frames, ~{total / fps:.0f}s @ {fps}fps)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default=','.join(SCENES))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='tracking_preview')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--trail', type=int, default=30, help='trail length (frames) per track')
    ap.add_argument('--max_clips', type=int, default=0, help='debug: limit clips per scene (0=all)')
    a = ap.parse_args()
    for s in a.scenes.split(','):
        run_scene(s, a.gpu, a.out, a.fps, a.trail, a.max_clips)


if __name__ == '__main__':
    main()
