"""UAV-VAD models.

Reverse-distillation appearance pipeline:
  frozen DINOv3 teacher  ->  OCBE bottleneck (trainable)  ->  student decoder (trainable)
  anomaly = discrepancy between the teacher's tapped patch features and the decoder's reconstruction.
"""
from .teacher import DINOv3Teacher, build_appearance_teacher
from .bottleneck import OCBE, build_bottleneck
from .student import StudentDecoder, build_student
from .loss import topk_loss, full_loss, anomaly_score, anomaly_readouts, per_layer_topk_dist
from .dlbm import BoundaryOutput, DistillationLayerBoundaryModule, build_dlbm
from .boundary_supernet import (all_boundary_topk_scores, build_fixed_suffix_ocbe,
                                build_fixed_suffix_student, suffix_code_from_all_layer_ocbe)

__all__ = [
    'DINOv3Teacher', 'build_appearance_teacher',
    'OCBE', 'build_bottleneck',
    'StudentDecoder', 'build_student',
    'topk_loss', 'full_loss', 'anomaly_score', 'anomaly_readouts', 'per_layer_topk_dist',
    'BoundaryOutput', 'DistillationLayerBoundaryModule', 'build_dlbm',
    'all_boundary_topk_scores', 'build_fixed_suffix_ocbe', 'build_fixed_suffix_student',
    'suffix_code_from_all_layer_ocbe',
]
