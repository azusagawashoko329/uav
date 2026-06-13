"""Frozen single-stage VEHICLE detector for the BEHAVIOR branch.

Wraps a YOLOv8x pretrained on VisDrone (aerial/drone domain) and restricts outputs to VEHICLE
classes. A COCO-pretrained YOLO was tried first and FAILED on this top-down data (e.g. vehicle-
roundabout congestion detected ~0 cars); VisDrone weights detect the same jam at 7-9 cars.
Inference-only / frozen — it just supplies per-frame vehicle boxes to the L1 descriptor. Fully
independent of the DINOv3 appearance branch (no shared params, no gradients).

Restricting to vehicle classes is the main false-positive shield: scenes with no normal vehicles
(farmland / railway / solar / bike) yield no detections, so the behavior branch stays SILENT there
and appearance handles them.

Lives under behavior/ (not the appearance model/ package) and is loaded by path in behavior/l1.py
and behavior/detect_vis.py — it pulls in ultralytics, which the appearance pipeline must never
depend on. Runs in the DMAD conda env (the one with ultralytics); the appearance branch runs in base.
"""
import os
import numpy as np
import torch
from ultralytics import YOLO

# VisDrone class ids that count as "vehicle" (exclude 0 pedestrian / 1 people / 2 bicycle)
VEHICLE_CLASSES = {3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights', 'yolov8x-visdrone.pt')


class VehicleDetector:
    """Frozen YOLO restricted to vehicle classes.

    detect(images) -> list (one per image) of dict with:
        'boxes'   (N,4) float xyxy in pixels
        'scores'  (N,)  float confidence
        'classes' (N,)  int VisDrone class id (in VEHICLE_CLASSES)
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, iou=0.5, imgsz=640, device='cuda:0'):
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f'{weights} not found — fetch the VisDrone YOLOv8x (quality-first, inference-only) once:\n'
                f"  from huggingface_hub import hf_hub_download\n"
                f"  hf_hub_download('mshamrai/yolov8x-visdrone', 'best.pt')  # then copy to behavior/weights/yolov8x-visdrone.pt")
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.classes = sorted(VEHICLE_CLASSES)

    @torch.no_grad()
    def detect(self, images):
        """images: path | list[path] | np.ndarray(HWC) | list[ndarray]. Vehicle classes only.
        agnostic_nms=True: VisDrone often double-labels one vehicle (e.g. car+van) and class-aware NMS
        keeps both → two boxes on one object → spurious d_min≈0 / inflated count. Agnostic NMS merges
        them across classes (verified: high-overlap pairs 25→2 per 30 crossroads frames)."""
        res = self.model.predict(images, classes=self.classes, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device, agnostic_nms=True, verbose=False)
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


def build_vehicle_detector(weights=DEFAULT_WEIGHTS, conf=0.25, device='cuda:0'):
    return VehicleDetector(weights=weights, conf=conf, device=device)
