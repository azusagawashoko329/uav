"""Lightweight single-stage VEHICLE detector for the MOTION branch (object-centric path).

Wraps a YOLOv8 pretrained on VisDrone (aerial/drone domain) and restricts outputs to VEHICLE
classes. A COCO-pretrained YOLO was tried first and FAILED on this top-down data (e.g. vehicle-
roundabout congestion detected ~0 cars); VisDrone weights detect the same jam at 7-9 cars.
Inference-only / frozen — it just supplies per-frame vehicle boxes to the downstream context-aware
behavior model. Fully independent of the DINOv3 appearance branch (no shared params, no gradients).

Restricting to vehicle classes is the main false-positive shield: scenes with no normal vehicles
(farmland / railway / solar / bike) yield no detections, so the motion branch stays SILENT there
and appearance handles them (see README_codex.md anomaly audit / the motion-branch FP design).

Lives under motion/ (not the appearance model/ package) and is loaded by path in motion/r1.py and
motion/visualize.py — it pulls in ultralytics, which the appearance pipeline must never depend on.
Runs in the DMAD conda env (the one with ultralytics); the appearance branch runs in base.
"""
import os
import numpy as np
import torch
from ultralytics import YOLO

# VisDrone class ids that count as "vehicle" (exclude 0 pedestrian / 1 people / 2 bicycle)
VEHICLE_CLASSES = {3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights', 'yolov8s-visdrone.pt')


class VehicleDetector:
    """Frozen YOLO restricted to vehicle classes.

    detect(images) -> list (one per image) of dict with:
        'boxes'   (N,4) float xyxy in pixels
        'scores'  (N,)  float confidence
        'classes' (N,)  int COCO class id (in VEHICLE_CLASSES)
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, iou=0.5, imgsz=640, device='cuda:0'):
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f'{weights} not found — fetch the VisDrone YOLOv8s once:\n'
                f"  from huggingface_hub import hf_hub_download\n"
                f"  hf_hub_download('mshamrai/yolov8s-visdrone', 'best.pt')  # then copy to model/weights/yolov8s-visdrone.pt")
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.classes = sorted(VEHICLE_CLASSES)
        self._tracker_args = None

    def _botsort_args(self):
        """BoT-SORT config namespace (bundled defaults; GMC off — it gave no quality change on this
        near-static-camera aerial data and costs CPU optical flow per frame)."""
        if self._tracker_args is None:
            import ultralytics
            from ultralytics.utils import IterableSimpleNamespace
            try:
                from ultralytics.utils import YAML; _load = YAML.load
            except ImportError:
                from ultralytics.utils import yaml_load as _load
            d = dict(_load(os.path.join(os.path.dirname(ultralytics.__file__), 'cfg', 'trackers', 'botsort.yaml')))
            d['gmc_method'] = 'none'
            self._tracker_args = IterableSimpleNamespace(**d)
        return self._tracker_args

    @torch.no_grad()
    def detect(self, images):
        """images: path | list[path] | np.ndarray(HWC) | list[ndarray]. Vehicle classes only."""
        res = self.model.predict(images, classes=self.classes, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            if b is None or b.shape[0] == 0:
                out.append({'boxes': np.zeros((0, 4), np.float32), 'scores': np.zeros((0,), np.float32),
                            'classes': np.zeros((0,), np.int64)})
            else:
                out.append({'boxes': b.xyxy.cpu().numpy().astype(np.float32),
                            'scores': b.conf.cpu().numpy().astype(np.float32),
                            'classes': b.cls.cpu().numpy().astype(np.int64)})
        return out

    @torch.no_grad()
    def track_clip(self, frame_paths):
        """Track vehicles through an ORDERED frame list as ONE clip.

        Runs our OWN per-frame detection and feeds the boxes to a STANDALONE BoT-SORT tracker. This
        deliberately bypasses ultralytics' `model.track(...)`: that path treats a loose image
        sequence (list / path / array) as INDEPENDENT images, so the tracker loses frame-to-frame
        association and IDs teleport between objects (33-61% of consecutive boxes jump >80 px on
        multi-vehicle scenes, even though raw detections are stable to ~5 px). The standalone tracker
        fed our detections is clean (0% teleport, cold). A fresh tracker per call restarts IDs per
        clip (no ghost tracks across cuts). Memory-safe: one frame on the GPU at a time. Yields
        (orig_img_bgr, det) per frame; det has boxes/scores/classes plus 'ids' (N,) int track ids
        (all >=0 — only confirmed tracks are emitted).
        """
        from ultralytics.trackers import BOTSORT
        tracker = BOTSORT(self._botsort_args())
        empty = {'boxes': np.zeros((0, 4), np.float32), 'scores': np.zeros((0,), np.float32),
                 'classes': np.zeros((0,), np.int64), 'ids': np.zeros((0,), np.int64)}
        for fp in frame_paths:
            r = self.model.predict(source=fp, classes=self.classes, conf=self.conf, iou=self.iou,
                                   imgsz=self.imgsz, device=self.device, verbose=False)[0]
            rows = tracker.update(r.boxes.cpu().numpy(), r.orig_img)   # [x1,y1,x2,y2,id,conf,cls,idx]
            if len(rows):
                rows = np.asarray(rows)
                det = {'boxes': rows[:, :4].astype(np.float32), 'ids': rows[:, 4].astype(np.int64),
                       'scores': rows[:, 5].astype(np.float32), 'classes': rows[:, 6].astype(np.int64)}
            else:
                det = dict(empty)
            yield r.orig_img, det


def build_vehicle_detector(weights=DEFAULT_WEIGHTS, conf=0.25, device='cuda:0'):
    return VehicleDetector(weights=weights, conf=conf, device=device)
