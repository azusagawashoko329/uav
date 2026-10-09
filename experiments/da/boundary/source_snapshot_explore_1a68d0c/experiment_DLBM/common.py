"""Shared utilities for the DLBM probe -> boundary -> fresh-final pipeline."""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
ALL_SCENES = tuple(SCENES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_scenes(value):
    scenes = [s.strip() for s in value.split(',') if s.strip()]
    unknown = [s for s in scenes if s not in SCENES]
    if unknown:
        raise ValueError(f'unknown scene keys: {unknown}; choices={list(SCENES)}')
    if not scenes:
        raise ValueError('at least one scene is required')
    return scenes


def cross_frame_cutpaste(batch, min_area=0.02, max_area=0.15):
    """Create a label-free local counterfactual using only normal frames from the same scene.

    Pixel values are already normalized, so copying a rectangle preserves the source pixels exactly.
    For a one-frame batch the source is a different spatial crop of the same frame.
    """
    if batch.ndim != 4:
        raise ValueError(f'expected BCHW batch, got {tuple(batch.shape)}')
    if not 0 < min_area <= max_area < 1:
        raise ValueError('expected 0 < min_area <= max_area < 1')
    b, _, height, width = batch.shape
    source = batch.roll(1, dims=0)
    output = batch.clone()
    for i in range(b):
        area = float(torch.empty(1).uniform_(min_area, max_area)) * height * width
        aspect = math.exp(float(torch.empty(1).uniform_(math.log(0.5), math.log(2.0))))
        patch_h = max(1, min(height - 1, round(math.sqrt(area / aspect))))
        patch_w = max(1, min(width - 1, round(math.sqrt(area * aspect))))
        sy = int(torch.randint(0, height - patch_h + 1, (1,)))
        sx = int(torch.randint(0, width - patch_w + 1, (1,)))
        dy = int(torch.randint(0, height - patch_h + 1, (1,)))
        dx = int(torch.randint(0, width - patch_w + 1, (1,)))
        output[i, :, dy:dy + patch_h, dx:dx + patch_w] = source[
            i, :, sy:sy + patch_h, sx:sx + patch_w]
    return output


def multi_mode_pseudo_stress(batch, max_blocks=2, return_mask=False):
    """Train-only local stress family used by the historical effective Gate."""
    if batch.ndim != 4:
        raise ValueError(f'expected BCHW batch, got {tuple(batch.shape)}')
    output = batch.clone()
    _, channels, height, width = output.shape
    masks = output.new_zeros((len(output), height, width))
    maximum = max(8, min(height, width) // 4)
    for index in range(len(output)):
        for _ in range(random.randint(1, int(max_blocks))):
            size = min(random.choice([8, 16, 24, 32, 48]), maximum)
            target_y = random.randint(0, height - size)
            target_x = random.randint(0, width - size)
            mode = random.choices(
                ['cutpaste', 'noise', 'black', 'white'],
                weights=[0.6, 0.2, 0.1, 0.1],
            )[0]
            if mode == 'cutpaste':
                source_y = random.randint(0, height - size)
                source_x = random.randint(0, width - size)
                source = output[
                    index, :, source_y:source_y + size, source_x:source_x + size,
                ].clone()
                if random.random() < 0.5:
                    source = torch.flip(source, dims=[random.choice([1, 2])])
            elif mode == 'noise':
                source = output[
                    index, :, target_y:target_y + size, target_x:target_x + size,
                ] + torch.randn(channels, size, size, device=output.device) * 0.8
            else:
                source = torch.full(
                    (channels, size, size), -2.0 if mode == 'black' else 2.0,
                    device=output.device,
                )
            coordinate = torch.linspace(-1, 1, size, device=output.device)
            yy, xx = torch.meshgrid(coordinate, coordinate, indexing='ij')
            mask = (1.0 - torch.sqrt(yy.square() + xx.square()).clamp(max=1.0)).unsqueeze(0)
            target = output[
                index, :, target_y:target_y + size, target_x:target_x + size,
            ]
            output[index, :, target_y:target_y + size, target_x:target_x + size] = (
                mask * source + (1.0 - mask) * target
            )
            masks[index, target_y:target_y + size, target_x:target_x + size] = torch.maximum(
                masks[index, target_y:target_y + size, target_x:target_x + size],
                mask.squeeze(0),
            )
    return (output, masks) if return_mask else output


def policy_from_logits(logits, temperature=1.0):
    if logits.ndim != 1:
        raise ValueError(f'expected one logit per layer, got {tuple(logits.shape)}')
    probabilities = (logits / temperature).softmax(dim=-1)
    gates = probabilities.cumsum(dim=-1)
    boundary = int(probabilities.argmax().item())
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    entropy = float((entropy / math.log(len(probabilities))).item()) if len(probabilities) > 1 else 0.0
    return {
        'boundary': boundary,
        'selected_taps': list(range(boundary, len(probabilities))),
        'probabilities': [float(v) for v in probabilities.cpu()],
        'soft_gates': [float(v) for v in gates.cpu()],
        'expected_boundary': float(
            (probabilities * torch.arange(len(probabilities), device=probabilities.device)).sum().item()),
        'confidence': float(probabilities.max().item()),
        'normalized_entropy': entropy,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def probe_path(probe_dir, scene):
    probe_dir = Path(probe_dir)
    candidates = (probe_dir / f'{scene}.pth', probe_dir / scene / f'{scene}.pth')
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'missing probe checkpoint; tried: {list(candidates)}')


def ensure_all_layer_probe(checkpoint, scene):
    cfg = checkpoint.get('cfg', {})
    taps = list(cfg.get('taps', []))
    if taps != list(range(12)):
        raise ValueError(f'{scene} probe must use taps 0..11, got {taps}')
    if cfg.get('tap_align', 'relative') != 'absolute':
        raise ValueError(f'{scene} probe must use tap_align=absolute for boundary comparability')
    return cfg
