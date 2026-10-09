"""Exact contiguous-suffix views of an all-layer OCBE/Student Probe.

For boundary ``k``, an all-layer OCBE first linear layer can be restricted to the input columns for
``[k, ..., L-1]``.  This is algebraically identical to zero-masking excluded tap chunks and also to
a fixed-suffix OCBE whose first-layer weights are copied from those columns.  Absolute Student
alignment then lets the matching decoder heads/blocks be copied without moving layer positions.
"""
import torch
import torch.nn.functional as F

from .bottleneck import build_bottleneck
from .loss import per_layer_topk_dist
from .student import build_student


def _validate_probe(ocbe, student, taps, boundary):
    layers = len(taps)
    if layers != ocbe.n_taps or layers != student.n_taps:
        raise ValueError(
            f'all-layer component mismatch: taps={layers}, ocbe={ocbe.n_taps}, student={student.n_taps}')
    if not 0 <= int(boundary) < layers:
        raise ValueError(f'boundary must lie in [0,{layers - 1}], got {boundary}')
    if student.tap_align != 'absolute':
        raise ValueError('boundary supernet requires absolute Student alignment')
    if student.tap_indices != list(range(layers)):
        raise ValueError(f'expected consecutive all-layer Student taps, got {student.tap_indices}')
    reference = taps[0].shape
    if len(reference) != 3 or reference[-1] != ocbe.embed_dim:
        raise ValueError(f'expected tap maps (B,P,{ocbe.embed_dim}), got {tuple(reference)}')
    if any(value.shape != reference for value in taps):
        raise ValueError('all tap maps must share shape')
    if len(ocbe.fuse) != 3 or not isinstance(ocbe.fuse[0], torch.nn.Linear):
        raise TypeError('unsupported OCBE fuse layout')
    return layers


def suffix_code_from_all_layer_ocbe(ocbe, taps, boundary):
    """Compute the exact code produced by retaining only tap chunks ``[boundary, ..., L-1]``."""
    layers = _validate_probe(ocbe, _OCBEValidationStudent(len(taps)), taps, boundary)
    selected = torch.cat([
        ocbe.tap_norms[layer](taps[layer]) for layer in range(int(boundary), layers)
    ], dim=-1)
    first = ocbe.fuse[0]
    column_start = int(boundary) * ocbe.embed_dim
    hidden = F.linear(selected, first.weight[:, column_start:], first.bias)
    hidden = ocbe.fuse[1](hidden)
    return ocbe.fuse[2](hidden)


class _OCBEValidationStudent:
    """Minimal adapter so OCBE-only validation uses the same invariant checks."""
    tap_align = 'absolute'

    def __init__(self, layers):
        self.n_taps = layers
        self.tap_indices = list(range(layers))


def build_fixed_suffix_ocbe(all_layer_ocbe, boundary):
    """Slice an all-layer OCBE into a standalone fixed-suffix OCBE with identical output."""
    layers = all_layer_ocbe.n_taps
    if not 0 <= int(boundary) < layers:
        raise ValueError(f'boundary must lie in [0,{layers - 1}], got {boundary}')
    hidden = all_layer_ocbe.fuse[0].out_features
    fixed = build_bottleneck(
        n_taps=layers - int(boundary), code_dim=all_layer_ocbe.code_dim,
        embed_dim=all_layer_ocbe.embed_dim, fuse_hidden=hidden,
        use_tap_norm=all_layer_ocbe.use_tap_norm,
    ).to(device=all_layer_ocbe.fuse[0].weight.device, dtype=all_layer_ocbe.fuse[0].weight.dtype)
    for target, source in zip(fixed.tap_norms, all_layer_ocbe.tap_norms[int(boundary):]):
        target.load_state_dict(source.state_dict())
    with torch.no_grad():
        start = int(boundary) * all_layer_ocbe.embed_dim
        fixed.fuse[0].weight.copy_(all_layer_ocbe.fuse[0].weight[:, start:])
        fixed.fuse[0].bias.copy_(all_layer_ocbe.fuse[0].bias)
        fixed.fuse[2].weight.copy_(all_layer_ocbe.fuse[2].weight)
        fixed.fuse[2].bias.copy_(all_layer_ocbe.fuse[2].bias)
    return fixed


def build_fixed_suffix_student(all_layer_student, boundary):
    """Copy matching absolute decoder blocks and heads into a standalone suffix Student."""
    layers = all_layer_student.n_taps
    if all_layer_student.tap_align != 'absolute':
        raise ValueError('boundary supernet requires absolute Student alignment')
    if all_layer_student.tap_indices != list(range(layers)):
        raise ValueError(f'expected consecutive all-layer taps, got {all_layer_student.tap_indices}')
    if not 0 <= int(boundary) < layers:
        raise ValueError(f'boundary must lie in [0,{layers - 1}], got {boundary}')
    heads = all_layer_student.blocks[0].attn.num_heads
    fixed = build_student(
        n_taps=layers - int(boundary), code_dim=all_layer_student.proj_in.in_features,
        embed_dim=all_layer_student.embed_dim, depth=all_layer_student.depth, heads=heads,
        tap_indices=list(range(int(boundary), layers)), tap_align='absolute',
        teacher_depth=all_layer_student.teacher_depth,
    ).to(device=all_layer_student.proj_in.weight.device, dtype=all_layer_student.proj_in.weight.dtype)
    fixed.proj_in.load_state_dict(all_layer_student.proj_in.state_dict())
    fixed.blocks.load_state_dict(all_layer_student.blocks.state_dict())
    for target, source in zip(fixed.heads, all_layer_student.heads[int(boundary):]):
        target.load_state_dict(source.state_dict())
    return fixed


def all_boundary_topk_scores(ocbe, student, taps, topk=5):
    """Evaluate every exact suffix efficiently with one batched Student forward.

    Returns:
      ``boundary_scores``: ``(B,L)``, final detector score for every boundary.
      ``all_layer_scores``: ``(B,L)``, per-layer scores under boundary L0, suitable as DLBM input.
    """
    layers = _validate_probe(ocbe, student, taps, boundary=0)
    batch = taps[0].shape[0]
    codes = torch.cat([
        suffix_code_from_all_layer_ocbe(ocbe, taps, boundary) for boundary in range(layers)
    ], dim=0)
    reconstructions = student(codes)
    boundary_scores = []
    all_layer_scores = None
    for boundary in range(layers):
        begin, end = boundary * batch, (boundary + 1) * batch
        selected_reconstructions = [
            reconstructions[layer][begin:end] for layer in range(boundary, layers)
        ]
        layer_scores = per_layer_topk_dist(
            taps[boundary:], selected_reconstructions, k=topk,
        )
        if boundary == 0:
            all_layer_scores = layer_scores
        boundary_scores.append(layer_scores.mean(dim=1))
    return torch.stack(boundary_scores, dim=1), all_layer_scores
