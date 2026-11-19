"""Non-formal HAP-RD ablation: identical LGGD/PDFB, alpha=0 in train and test.

Keep the eight published experiment specs and their pinned source bytes intact.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .bottleneck import OCBE
from .publication_rd import model_spec, PublicationDecoder, architecture_guard

VARIANT = 'lggd_no_laga'


def ablation_spec():
    spec = model_spec('hap_rd')
    spec.update(label='HAP-RD w/o LAGA', alpha=0.0)
    return spec


def build_ablation_model():
    spec = ablation_spec()
    bottleneck = OCBE(n_taps=12, code_dim=64, fuse_hidden=512, use_tap_norm=True)
    decoder = PublicationDecoder(spec)
    ablation_guard(bottleneck, decoder)
    return bottleneck, decoder


def ablation_guard(bottleneck, decoder):
    if decoder.spec != ablation_spec():
        raise ValueError('decoder spec is not the exact LLLG+gate alpha0 ablation')
    guard = architecture_guard(bottleneck, decoder, 'hap_rd')
    if guard['trainable_parameters'] != 30248000:
        raise ValueError('parameter count changed')
    guard['variant'] = VARIANT
    guard['match']['alpha'] = 0.0
    return guard


def ablation_match(taps, reconstructions):
    """B scores: mean of twelve independent Top-5 patch cosine distances.

Equivalent to four equally weighted three-layer group means (alpha=0).
Training adds only a batch mean; evaluation uses these same B scores.
"""
    if len(taps) != 12 or len(reconstructions) != 12:
        raise ValueError('exactly twelve matched teacher/head pairs required')
    layer_errors = []
    for teacher, student in zip(taps, reconstructions):
        if teacher.shape != student.shape or teacher.ndim != 3 or teacher.shape[1:] != (256, 768):
            raise ValueError('expected Bx256x768 matched features')
        distances = 1 - F.cosine_similarity(teacher, student, dim=-1)
        layer_errors.append(distances.topk(5, dim=-1).values.mean(-1))
    # Same reduction order as the alpha=.5 baseline, with only its max term removed.
    groups = torch.stack(layer_errors, dim=1).reshape(taps[0].shape[0], 4, 3)
    return groups.mean(-1).mean(1)
