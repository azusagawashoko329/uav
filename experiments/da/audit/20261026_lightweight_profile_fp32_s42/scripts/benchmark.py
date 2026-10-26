"""Matched inference audit of existing architectures; no training or AUC selection."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import build_appearance_teacher, build_bottleneck, build_student
from model.loss import anomaly_score, anomaly_readouts
from model.student_adapter import FourBlockTwelveHeadDecoder
from model.student_structural import FourBlockGlobalLocalDecoder
from model.bottleneck_depthrank import build_depthrank2_bottleneck
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.bottleneck_dgt import build_depth_group_gated_tensor_bottleneck
from model.bottleneck_dgt_variants import build_dgt_variant


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def encode_clean(self, taps):
    normalized = [norm(tap) for norm, tap in zip(self.tap_norms, taps)]
    q = self.effective_depth_basis()
    groups = []
    for start in range(0, self.n_taps, 3):
        first, middle, last = normalized[start:start + 3]
        mode0 = q[0, 0] * first + q[0, 1] * middle + q[0, 2] * last
        mode1 = q[1, 0] * first + q[1, 1] * middle + q[1, 2] * last
        group = torch.cat((mode0, mode1), dim=-1)
        value = F.gelu(self.value_proj(group))
        gate = 1.0 + torch.tanh(self.gate_proj(group))
        groups.append(value * gate)
    return torch.stack(groups, dim=2)


def cross_clean(self, taps):
    hidden = self.encode_groups(taps)
    x = hidden + self.depth_pos.view(1, 1, self.n_groups, -1)
    q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    attn = (torch.einsum('bnid,bnjd->bnij', q, k) / self.attention_dim ** 0.5).softmax(-1)
    mixed = hidden + self.attn_out(torch.einsum('bnij,bnjd->bnid', attn, v))
    weights = torch.einsum('md,bngd->bnmg', self.pool_queries, mixed).softmax(-1)
    pooled = torch.einsum('bnmg,bngd->bnmd', weights, mixed)
    return self.pool_out(pooled.reshape(pooled.shape[0], pooled.shape[1], -1))


def bilinear_clean(self, taps):
    hidden = self.encode_groups(taps)
    deltas = []
    for g in range(self.n_groups - 1):
        left = F.gelu(self.pair_a(hidden[:, :, g, :]))
        right = F.gelu(self.pair_b(hidden[:, :, g + 1, :]))
        deltas.append(self.pair_up(left * right))
    mixed = hidden.clone()
    for g, delta in enumerate(deltas):
        mixed[:, :, g, :] = mixed[:, :, g, :] + delta
        mixed[:, :, g + 1, :] = mixed[:, :, g + 1, :] + delta
    return self.out_proj(mixed.reshape(mixed.shape[0], mixed.shape[1], -1))


def residual_clean(self, taps):
    hidden = self.encode_groups(taps)
    terms = []
    for g in range(self.n_groups - 1):
        diff = hidden[:, :, g + 1, :] - hidden[:, :, g, :]
        level = hidden[:, :, g + 1, :] + hidden[:, :, g, :]
        terms.append(F.gelu(self.pair_a(diff)) * F.gelu(self.pair_b(level)))
    residual = self.residual_out(torch.cat(terms, dim=-1))
    base = self.out_proj(hidden.reshape(hidden.shape[0], hidden.shape[1], -1))
    return base + residual


def clean_copy(model, kind):
    clean = copy.deepcopy(model)
    if hasattr(clean, 'encode_groups'):
        clean.encode_groups = MethodType(encode_clean, clean)
        fn = {'crossattn_pool': cross_clean, 'bilinear': bilinear_clean,
              'residual_depth': residual_clean}.get(kind)
        if fn:
            clean.forward = MethodType(fn, clean)
    return clean


def build_pair(spec):
    kind = spec['ocbe']
    n = len(spec['taps'])
    if kind == 'dense':
        ocbe = build_bottleneck(n_taps=n, fuse_hidden=spec['hidden'])
    elif kind == 'rank2_fixed':
        ocbe = build_depthrank2_bottleneck()
    elif kind == 'rank2_random':
        ocbe = build_learnable_depthrank2_bottleneck(basis_init='random_orthonormal', basis_seed=42)
    elif kind == 'dgt_graph':
        ocbe = build_depth_group_gated_tensor_bottleneck()
    else:
        ocbe = build_dgt_variant(kind)
    if spec['student'] == 'original12':
        student = build_student(n_taps=4, depth=12)
    elif spec['student'].startswith('original4'):
        student = FourBlockTwelveHeadDecoder(adapter=spec.get('adapter') or 'none')
    else:
        student = FourBlockGlobalLocalDecoder(dim=spec['dim'])
    checkpoint = ROOT / spec['experiment'] / 'ckpt/bike_final.pth'
    weight_source = {'kind': 'seed42_architecture_initialization',
                     'note': 'Original trained checkpoint unavailable locally; fixed dense graph has no data-dependent routing.'}
    if checkpoint.is_file():
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ocbe.load_state_dict(data['ocbe_state_dict'], strict=True)
        student.load_state_dict(data['student_state_dict'], strict=True)
        weight_source = {'kind': 'trained_bike_checkpoint', 'path': str(checkpoint.relative_to(ROOT)),
                         'sha256': digest(checkpoint), 'seed': data['seed']}
    return ocbe.eval(), student.eval(), weight_source


def size_metrics(ocbe, student, teacher):
    state = OrderedDict()
    for prefix, model in [('ocbe', ocbe), ('student', student)]:
        state.update((prefix + '.' + k, v.detach().cpu()) for k, v in model.state_dict().items())
    raw = sum(v.numel() * v.element_size() for v in state.values())
    buf = io.BytesIO()
    torch.save(state, buf)
    serial = buf.tell()
    buf.close()
    for k, v in teacher.state_dict().items():
        state['teacher.' + k] = v.detach().cpu()
    buf = io.BytesIO()
    torch.save(state, buf)
    full_serial = buf.tell()
    buf.close()
    return {'core_state_tensor_bytes': raw, 'core_pth_bytes': serial,
            'full_pth_bytes': full_serial,
            'full_state_tensor_bytes': sum(v.numel() * v.element_size() for v in state.values()),
            'note': 'Actual torch.save state_dict byte length measured in memory; excludes optimizer and metadata; no untrained checkpoint exported.'}


def major_flops(fn):
    # no inference_mode: PyTorch 2.1 otherwise bypasses Python dispatch. Enabling
    # grad also prevents fused native MHA from hiding its projection GEMMs.
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        result = fn()
    del result
    return {'flops': int(counter.get_total_flops()),
            'operators': {str(k): int(v) for k, v in counter.flop_counts['Global'].items()}}


def expected_student_flops(student, patches=256):
    # Independent equation including MHA's functional (not Module) QKV weights.
    total = 0
    for module in student.modules():
        if isinstance(module, torch.nn.Linear):
            # MultiheadAttention.out_proj is implemented functionally; count
            # the whole MHA below and exclude this child to avoid double count.
            if any(module is a.out_proj for a in student.modules()
                   if isinstance(a, torch.nn.MultiheadAttention)):
                continue
            total += 2 * patches * module.in_features * module.out_features
        elif isinstance(module, torch.nn.MultiheadAttention):
            d = module.embed_dim
            total += 2 * patches * 4 * d * d + 4 * patches * patches * d
        elif isinstance(module, torch.nn.Conv2d):
            total += 2 * patches * module.out_channels * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    return total


def measure(fn, device, warmup, samples, repeats, batch):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        sample_rounds = []
        for _ in range(repeats):
            times = []
            for _ in range(samples):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
            sample_rounds.append(times)
        # Separate queued throughput, no synchronization between individual calls.
        rates = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(samples):
                fn()
            torch.cuda.synchronize(device)
            rates.append(batch * samples / (time.perf_counter() - start))
    values = np.array(sample_rounds).ravel()
    return {'latency_p50_ms': float(np.median(values)),
            'latency_p95_ms': float(np.percentile(values, 95)),
            'latency_mean_ms': float(values.mean()),
            'latency_std_ms': float(values.std()),
            'serial_fps_from_p50': 1000 * batch / float(np.median(values)),
            'queued_throughput_fps': float(np.median(rates)),
            'round_p50_ms': [float(np.median(v)) for v in sample_rounds],
            'throughput_rounds_fps': rates, 'batch': batch, 'samples_per_round': samples,
            'rounds': repeats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    config = read_json(EXP / 'config.json')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    started = stamp()
    env = {'torch': torch.__version__, 'python': platform.python_version(),
           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
           'cuda_ordinal': args.gpu, 'cuda_device_name': torch.cuda.get_device_name(device),
           'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.used', '--format=csv'], text=True),
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'started_at': started, 'dtype': 'float32', 'tf32': False,
           'cpu_threads': torch.get_num_threads()}
    print(json.dumps({'event': 'started', **env}), flush=True)
    teacher = build_appearance_teacher(layers=list(range(12)), pretrained=True).to(device).eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    result = {'environment': env, 'teacher_parameters': teacher_params,
              'measurement_config': {k: config[k] for k in ['batches', 'warmup', 'samples', 'repeats']},
              'teacher': {}, 'variants': []}
    image1 = torch.randn(1, 3, 256, 256, device=device)
    teacher_flops = major_flops(lambda: teacher(image1))
    n, d = 261, 768
    expected_teacher = 2 * 256 * d * 3 * 16 * 16 + 12 * (2 * n * 12 * d * d + 4 * n * n * d)
    if teacher_flops['flops'] != expected_teacher:
        raise ValueError(('Teacher FLOP equation mismatch', teacher_flops, expected_teacher))
    result['teacher_flops'] = teacher_flops
    images = {batch: torch.randn(batch, 3, 256, 256, device=device) for batch in config['batches']}
    for batch, image in images.items():
        result['teacher'][str(batch)] = measure(lambda: teacher(image), device, config['warmup'], config['samples'], config['repeats'], batch)
    path = EXP / f'results/gpu{args.gpu}.json'
    for spec in config['variants']:
        if args.only and spec['id'] not in args.only:
            continue
        torch.manual_seed(42)
        ocbe, student, weights = build_pair(spec)
        sizes = size_metrics(ocbe, student, teacher)
        ocbe = ocbe.to(device)
        clean = clean_copy(ocbe, spec['ocbe']).to(device).eval()
        student = student.to(device).eval()
        row = {'id': spec['id'], 'label': spec['label'], 'experiment': spec['experiment'],
               'weight_source': weights, 'model_size': sizes,
               'ocbe_parameters': sum(p.numel() for p in ocbe.parameters()),
               'student_parameters': sum(p.numel() for p in student.parameters()),
               'batches': {}}
        row['core_parameters'] = row['ocbe_parameters'] + row['student_parameters']
        row['full_parameters'] = row['core_parameters'] + teacher_params
        if row['core_parameters'] != spec['expected_core_parameters']:
            raise ValueError(('Parameter mismatch', spec['id'], row['core_parameters']))
        for batch, image in images.items():
            with torch.inference_mode():
                all_taps = teacher(image)['patches']
                taps = [all_taps[i] for i in spec['taps']]
                code = clean(taps)
                raw_code = ocbe(taps)
                torch.testing.assert_close(code, raw_code, atol=1e-6, rtol=1e-5)
                rec = student(code)
                torch.testing.assert_close(anomaly_score(taps, rec), anomaly_readouts(taps, rec)['topk'])
                diff = float((code - raw_code).abs().max().cpu())
            if batch == 1:
                # New tensors outside inference_mode are needed for autograd-safe
                # operation dispatch during FLOP counting, never during timing.
                profile_taps = [t.clone() for t in taps]
                profile_code = code.clone()
                row['ocbe_flops'] = major_flops(lambda: clean(profile_taps))
                row['student_flops'] = major_flops(lambda: student(profile_code))
                if row['student_flops']['flops'] != expected_student_flops(student):
                    raise ValueError(('Student FLOP equation mismatch', spec['id'], row['student_flops']))
                row['core_flops'] = row['ocbe_flops']['flops'] + row['student_flops']['flops']
                row['full_flops'] = row['core_flops'] + teacher_flops['flops']
                del profile_taps, profile_code
            def full_score():
                ts = teacher(image)['patches']
                ts = [ts[i] for i in spec['taps']]
                return anomaly_score(ts, student(clean(ts)))
            functions = OrderedDict([
                ('ocbe', lambda: clean(taps)),
                ('student', lambda: student(code)),
                ('core', lambda: student(clean(taps))),
                ('core_score', lambda: anomaly_score(taps, student(clean(taps)))),
                ('full_score', full_score),
            ])
            if hasattr(ocbe, 'encode_groups'):
                functions['core_score_with_diagnostics'] = lambda: anomaly_score(taps, student(ocbe(taps)))
            timed = {}
            for scope, fn in functions.items():
                timed[scope] = measure(fn, device, config['warmup'], config['samples'], config['repeats'], batch)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                full_score()
            torch.cuda.synchronize(device)
            timed['clean_output_max_abs_diff'] = diff
            timed['peak_full_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            row['batches'][str(batch)] = timed
            print(json.dumps({'event': 'measured', 'variant': spec['id'], 'batch': batch,
                              'core_gflops': row['core_flops'] / 1e9,
                              'core_ms': timed['core_score']['latency_p50_ms'],
                              'full_ms': timed['full_score']['latency_p50_ms']}), flush=True)
            del all_taps, taps, code, raw_code, rec
        result['variants'].append(row)
        result['last_updated_at'] = stamp()
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        del ocbe, clean, student
        gc.collect()
        torch.cuda.empty_cache()
    result['finished_at'] = stamp()
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'event': 'completed', 'gpu': args.gpu, 'path': str(path)}), flush=True)


if __name__ == '__main__':
    main()
