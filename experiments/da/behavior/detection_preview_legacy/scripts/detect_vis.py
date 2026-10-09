"""检测质量预览 —— yolov8x 车辆检测,逐帧画框(类别+置信度),把一个场景的所有子视频(train+test)
拼成一个 mp4。纯检测,无跟踪无尾迹,就是看 DETECTION 质量。交付到 experiments/da/behavior/detection_preview_legacy/artifacts/<scene>.mp4。
Env: DMAD.  /home/lyx/miniconda3/envs/DMAD/bin/python experiments/da/behavior/detection_preview_legacy/scripts/detect_vis.py --scenes crossroads --gpu 0
"""
import argparse, os, glob, importlib.util
import numpy as np
import cv2

_spec = importlib.util.spec_from_file_location(
    'vehicle_detector', os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')), 'behavior', 'detector.py'))
_dm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dm)
build_vehicle_detector, VEHICLE_CLASSES = _dm.build_vehicle_detector, _dm.VEHICLE_CLASSES

DATA_ROOT = os.environ.get('UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
          'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
          'vehicle': 'Vehicle Roundabout'}
BATCH = 32
# distinct color per vehicle class (BGR)
CLS_COLOR = {3: (0, 255, 0), 4: (255, 200, 0), 5: (0, 165, 255), 6: (255, 0, 255),
             7: (200, 0, 200), 8: (0, 0, 255), 9: (255, 255, 0)}


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def list_clips(scene_dir):
    clips = []
    for seq in sorted(glob.glob(os.path.join(scene_dir, 'sequence*'))):
        for split in ('train', 'test'):
            for sub in sorted(glob.glob(os.path.join(seq, split, '*'))):
                if os.path.isdir(sub):
                    frames = sorted(glob.glob(os.path.join(sub, '*.jpg')), key=frame_idx)
                    if frames:
                        clips.append((f'{os.path.basename(seq)}/{split}/{os.path.basename(sub)}', frames))
    return clips


def draw(img, det, banner):
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), s, c in zip(det['boxes'], det['scores'], det['classes']):
        col = CLS_COLOR.get(int(c), (180, 180, 180))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f'{VEHICLE_CLASSES.get(int(c), "?")}{s:.2f}', (int(x1), max(10, int(y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, f'{banner}  det={len(det["boxes"])}', (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_scene(scene_key, gpu, out_dir, fps):
    det = build_vehicle_detector(conf=0.25, device=f'cuda:{gpu}')
    clips = list_clips(os.path.join(DATA_ROOT, SCENES[scene_key]))
    h0, w0 = cv2.imread(clips[0][1][0]).shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f'{scene_key}.mp4')
    if os.path.exists(outp):
        raise FileExistsError(f'preview already exists: {outp}')
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w0, h0))
    total = 0
    for ci, (label, frames) in enumerate(clips):
        n = len(frames)
        for i in range(0, n, BATCH):
            chunk = frames[i:i + BATCH]
            for j, (fp, d) in enumerate(zip(chunk, det.detect(chunk))):
                img = cv2.imread(fp)
                draw(img, d, f'{scene_key} | {label} | {i + j + 1}/{n}')
                vw.write(img); total += 1
        print(f'[{scene_key}] clip {ci + 1}/{len(clips)} {label} ({n}f)', flush=True)
    vw.release()
    print(f'[{scene_key}] WROTE {outp}  ({total} frames, ~{total / fps:.0f}s @ {fps}fps)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='crossroads,vehicle,highway')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts')))
    ap.add_argument('--fps', type=int, default=25)
    a = ap.parse_args()
    for s in a.scenes.split(','):
        run_scene(s, a.gpu, a.out, a.fps)


if __name__ == '__main__':
    main()
