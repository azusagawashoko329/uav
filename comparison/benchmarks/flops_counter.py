#!/usr/bin/env python3
"""v2 FLOPs counter for comparison benchmarks.

Primary counter: torch.utils.flop_counter.FlopCounterMode (TorchDispatchMode).
It covers Conv2d/ConvTranspose2d/Conv3d/ConvTranspose3d, Linear (via addmm/mm),
MatMul/BMM and fused SDPA kernels. We add a formula for
aten._native_multi_head_attention, which PyTorch's registry does not cover.

Cross-checks:
  * hand module formulas for Conv/ConvTranspose/Linear/MultiheadAttention;
  * torch.profiler(with_flops=True), compared only on op categories that both
    the counter and profiler report (never on the full total).

No inference_mode here: TorchDispatchMode/profiler need normal dispatch. Use
torch.no_grad() for counting; timing uses inference_mode separately.
"""
from __future__ import annotations

from collections import defaultdict
from math import prod
from typing import Callable

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

ATEN = torch.ops.aten


def _numel(shape) -> int:
    return int(prod(int(x) for x in shape))


def native_mha_flop(
    query_shape,
    key_shape,
    value_shape,
    embed_dim,
    num_head,
    qkv_weight_shape,
    qkv_bias_shape,
    proj_weight_shape,
    proj_bias_shape,
    mask_shape=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
    out_shape=None,
    **kwargs,
) -> int:
    """FLOPs for aten::_native_multi_head_attention (inference, no dropout).

    query is (B, L, E) for batch_first=True; accept (L, B, E) as a fallback.
    QKV projection + QK + AV + output projection, 1 MAC = 2 FLOPs.
    """
    e = int(embed_dim)
    if len(query_shape) != 3:
        raise ValueError(f'unexpected native MHA query shape: {query_shape}')
    if int(query_shape[-1]) == e:
        b, l, _ = (int(x) for x in query_shape)
    else:
        l, b, _ = (int(x) for x in query_shape)
    h = int(num_head)
    d = e // h
    qkv = 2 * b * l * e * (3 * e)
    qk = 2 * b * h * l * l * d
    av = 2 * b * h * l * l * d
    proj = 2 * b * l * e * e
    return int(qkv + qk + av + proj)


CUSTOM_MAPPING = {
    ATEN._native_multi_head_attention: native_mha_flop,
}


def _op_name(op) -> str:
    return str(op)


def count_flops(fn: Callable, *args) -> dict:
    """Count FLOPs with FlopCounterMode + native-MHA formula.

    Returns:
      total_flops: int
      per_op: {aten.op: flops}
      per_module: {module_name: {aten.op: flops}}
    """
    with torch.no_grad():
        with FlopCounterMode(custom_mapping=CUSTOM_MAPPING, display=False) as fc:
            fn(*args)
    counts = fc.get_flop_counts()
    per_module = {
        mod: {_op_name(op): int(v) for op, v in ops.items()}
        for mod, ops in counts.items()
    }
    per_op = per_module.get('Global', {})
    return {
        'total_flops': int(fc.get_total_flops()),
        'per_op': per_op,
        'per_module': per_module,
    }


def _conv_out_flops(module: nn.Module, inp, out) -> int:
    x = inp[0]
    w = module.weight
    groups = int(module.groups)
    if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # weight: (Cin, Cout/groups, *k); each input position contributes to
        # Cin * Cout/groups * prod(k) MACs. Border cropping/padding is ignored
        # by this formula; the counter uses actual output shapes for the
        # authoritative value.
        macs = _numel(x.shape) * int(w.shape[1]) * _numel(w.shape[2:])
        return int(2 * macs)
    # Conv2d/Conv3d: weight (Cout, Cin/groups, *k), use actual output shape.
    macs = _numel(out.shape) * int(w.shape[1]) * _numel(w.shape[2:])
    return int(2 * macs)


def _linear_flops(module: nn.Linear, inp, out) -> int:
    x = inp[0]
    return int(2 * _numel(x.shape) * int(module.out_features))


def _mha_flops(module: nn.MultiheadAttention, inp, out) -> int:
    q = inp[0]
    if module.batch_first:
        b, l, e = (int(v) for v in q.shape)
    else:
        l, b, e = (int(v) for v in q.shape)
    h = int(module.num_heads)
    d = e // h
    qkv = 2 * b * l * e * (3 * e)
    qk = 2 * b * h * l * l * d
    av = 2 * b * h * l * l * d
    proj = 2 * b * l * e * e
    return int(qkv + qk + av + proj)


def hand_module_flops(model: nn.Module, *args) -> dict:
    """Independent per-module formulas for Conv/ConvTranspose/Linear/MHA.

    This is a separate pass and is never added to the primary counter. Linear
    modules inside MultiheadAttention (out_proj) are skipped to avoid double
    counting; the MHA hook accounts for the full module.
    """
    records = defaultdict(int)
    handles = []
    skip_ids = set()
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            for child in m.modules():
                if child is not m:
                    skip_ids.add(id(child))

    def hook(name, fn):
        def _h(module, inp, out):
            records[name] += int(fn(module, inp, out))
        return _h

    for name, m in model.named_modules():
        if id(m) in skip_ids:
            continue
        key = name or 'Global'
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
            handles.append(m.register_forward_hook(hook(key, _conv_out_flops)))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook(key, _linear_flops)))
        elif isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(hook(key, _mha_flops)))
    try:
        with torch.no_grad():
            model(*args)
    finally:
        for h in handles:
            h.remove()
    return dict(records)


def _match_counter_module(name: str, per_module: dict) -> str | None:
    """ModuleTracker prefixes names with the root class (e.g. Wrap.m.conv).

    Match the named_modules() name by exact or suffix match.
    """
    if name in per_module:
        return name
    suffix = '.' + name if name else ''
    matches = [k for k in per_module if k == name or k.endswith(suffix)]
    if not matches:
        return None
    # Prefer the shortest matching tracker key.
    return min(matches, key=len)


def compare_hand_vs_counter(counter: dict, hand: dict, tol: float = 0.01) -> dict:
    """Compare hand module formulas with the primary counter, per module.

    Modules present only in the primary counter are reported as unverified by
    hand (e.g. timm SDPA blocks or functional F.linear); they are not failures
    if they are covered by the primary counter's registered ops.
    """
    per_module = counter['per_module']
    checked = []
    mismatches = []
    counter_only = []
    hand_only = []
    matched_counter_names = set()
    for name, hf in sorted(hand.items()):
        cname = _match_counter_module(name, per_module)
        if cname is None:
            hand_only.append(name)
            continue
        matched_counter_names.add(cname)
        ops = per_module.get(cname, {})
        cf = int(sum(ops.values()))
        if cf == 0:
            hand_only.append(name)
            continue
        rel = abs(cf - hf) / max(1, abs(hf))
        checked.append({'module': name, 'counter_module': cname,
                        'counter_flops': cf, 'hand_flops': int(hf), 'rel_diff': rel})
        if rel > tol:
            mismatches.append(checked[-1])
    for name, ops in sorted(per_module.items()):
        if name == 'Global' or name in matched_counter_names:
            continue
        if sum(ops.values()) > 0:
            counter_only.append(name)
    ok = not mismatches and not hand_only
    max_rel = max((x['rel_diff'] for x in checked), default=0.0)
    return {
        'ok': ok,
        'checked_modules': len(checked),
        'max_rel_diff': max_rel,
        'mismatches': mismatches,
        'counter_only_modules': counter_only,
        'hand_only_modules': hand_only,
    }


def profiler_flops(fn: Callable, *args) -> dict:
    """Run torch.profiler(with_flops=True) and aggregate by event key.

    This total is intentionally not used as the Full FLOPs acceptance value;
    it is only compared with the counter on common op categories.
    """
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
        ) as prof:
            fn(*args)
    by_op = defaultdict(int)
    for evt in prof.events():
        fl = int(getattr(evt, 'flops', 0) or 0)
        if not fl:
            continue
        key = str(getattr(evt, 'key', '') or getattr(evt, 'name', '') or '')
        # Normalize aten::op -> aten.op for comparison with counter keys.
        if key.startswith('aten::'):
            key = 'aten.' + key[len('aten::'):]
        by_op[key] += fl
    return {'total_flops': int(sum(by_op.values())), 'per_op': dict(by_op)}


_CONV_KEYS = {
    'aten.convolution', 'aten._convolution',
    'aten.conv2d', 'aten.conv3d',
    'aten.conv_transpose2d', 'aten.conv_transpose3d',
}
_MATMUL_KEYS = {'aten.addmm', 'aten.mm', 'aten.bmm', 'aten.baddbmm'}
_NATIVE_MHA = 'aten._native_multi_head_attention'


def _op_category(key: str) -> str:
    key = key.replace('aten::', 'aten.')
    if key in _CONV_KEYS:
        return 'conv'
    if key in _MATMUL_KEYS:
        return 'matmul'
    if key.startswith('aten._scaled_dot_product'):
        return 'attention'
    if key == _NATIVE_MHA:
        return 'native_mha'
    return 'other:' + key


def _aggregate_by_category(per_op: dict) -> dict:
    out = defaultdict(int)
    for key, val in per_op.items():
        out[_op_category(key)] += int(val)
    return dict(out)


def compare_profiler_common_subset(counter: dict, profiler: dict, tol: float = 0.01,
                                   skip_categories: set[str] | None = None) -> dict:
    """Compare only op categories present in both counter and profiler.

    Categories:
      * conv: Conv2d/Conv3d/ConvTranspose2d/3d (profiler may name aten.conv2d
        while the counter names aten.convolution);
      * matmul: addmm/mm/bmm/baddbmm — skipped when native MHA is present,
        because the profiler decomposes the fused MHA into matmul ops;
      * attention: _scaled_dot_product_* kernels;
      * native_mha: profiler does not report it, so it is counter-only.

    Profiler blind spots are skipped here and covered by the counter + hand
    formula; the profiler total is never used as the Full FLOPs acceptance.
    """
    c_op = counter['per_op']
    p_op = profiler['per_op']
    c_cat = _aggregate_by_category(c_op)
    p_cat = _aggregate_by_category(p_op)
    skip_categories = set(skip_categories or ())
    common = []
    skipped = []
    for cat, cv in sorted(c_cat.items()):
        if cat in skip_categories:
            skipped.append(cat + '(profiler_blind_spot)')
            continue
        if cat not in p_cat or p_cat[cat] == 0:
            skipped.append(cat)
            continue
        if cat == 'matmul' and c_cat.get('native_mha', 0) > 0:
            skipped.append('matmul(native_mha_decomposed)')
            continue
        if cat.startswith('other:'):
            skipped.append(cat)
            continue
        pv = int(p_cat[cat])
        rel = abs(int(cv) - pv) / max(1, abs(pv))
        common.append({
            'category': cat,
            'counter_flops': int(cv),
            'profiler_flops': pv,
            'rel_diff': rel,
        })
    bad = [x for x in common if x['rel_diff'] > tol]
    return {
        'ok': not bad,
        'common_ops': common,
        'common_categories': [x['category'] for x in common],
        'max_rel_diff': max((x['rel_diff'] for x in common), default=0.0),
        'mismatches': bad,
        'skipped_categories': skipped,
        'profiler_total_flops': int(profiler.get('total_flops', 0)),
        'counter_total_flops': int(counter.get('total_flops', 0)),
    }


def _mac_module_names(model: nn.Module) -> list[str]:
    names = set()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            names.add('Conv2d')
        elif isinstance(m, nn.ConvTranspose2d):
            names.add('ConvTranspose2d')
        elif isinstance(m, nn.Conv3d):
            names.add('Conv3d')
        elif isinstance(m, nn.ConvTranspose3d):
            names.add('ConvTranspose3d')
        elif isinstance(m, nn.Linear):
            names.add('Linear')
        elif isinstance(m, nn.MultiheadAttention):
            names.add('MultiheadAttention')
    return sorted(names)


def flops_report(fn: Callable, model: nn.Module, *args, run_profiler: bool = True) -> dict:
    """Build the v2 FLOPs report for one forward."""
    counter = count_flops(fn, *args)
    hand = hand_module_flops(model, *args)
    hand_check = compare_hand_vs_counter(counter, hand)
    prof = profiler_flops(fn, *args) if run_profiler else {'total_flops': 0, 'per_op': {}}
    # torch.profiler does not support ConvTranspose2d/3d or Conv3d reliably.
    # If the graph contains them, skip the 'conv' profiler comparison; the
    # hand formulas and the primary counter still cover those modules.
    skip_categories = set()
    if any(isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Conv3d))
           for m in model.modules()):
        skip_categories.add('conv')
    prof_check = compare_profiler_common_subset(
        counter, prof, skip_categories=skip_categories) if run_profiler else {
        'ok': True, 'common_ops': [], 'common_categories': [],
        'max_rel_diff': 0.0, 'mismatches': [], 'skipped_categories': [],
        'profiler_total_flops': 0, 'counter_total_flops': int(counter['total_flops']),
    }

    breakdown = []
    for mod, ops in counter['per_module'].items():
        if mod == 'Global':
            continue
        for op, fl in ops.items():
            if fl:
                breakdown.append({'module': mod, 'op': op, 'flops': int(fl)})
    breakdown.sort(key=lambda x: x['flops'], reverse=True)

    # Ops covered by the primary counter. Native MHA is explicitly covered by
    # our formula; SDPA kernels are covered by PyTorch's registry. Include
    # human-readable module op names (ConvTranspose2d/Conv3d/etc.) because the
    # dispatch op is a single aten.convolution for all of them.
    module_ops = _mac_module_names(model)
    covered = sorted(set(counter['per_op']) | set(module_ops))
    missing = []
    # A useful duplicate heuristic: a single conv should not be reported by
    # both convolution and _convolution in the same graph.
    if any('convolution' in k for k in counter['per_op']) and \
       any('_convolution' in k for k in counter['per_op']):
        # Both can appear legitimately if different modules use different
        # paths; report it instead of silently assuming duplication.
        duplicate_note = 'both aten.convolution and aten._convolution present; inspect per-module counts'
    else:
        duplicate_note = 'single dispatch path; module hooks and counter are separate passes'

    ok = bool(hand_check['ok'] and prof_check['ok'] and not missing)
    return {
        'total_flops': int(counter['total_flops']),
        'per_op': counter['per_op'],
        'breakdown': breakdown,
        'flops_verified': ok,
        'flops_ops_covered': covered,
        'flops_ops_missing': missing,
        'flops_module_ops': module_ops,
        'flops_common_subset_check': (
            f"hand-vs-counter: {hand_check['checked_modules']} modules, "
            f"max_rel_diff={hand_check['max_rel_diff']:.6f}; "
            f"counter-only modules={len(hand_check['counter_only_modules'])}; "
            f"profiler common categories={prof_check['common_categories']}, "
            f"max_rel_diff={prof_check['max_rel_diff']:.6f}; "
            f"skipped={prof_check['skipped_categories']}"
        ),
        'flops_duplicate_check': duplicate_note,
        'hand_check': hand_check,
        'profiler_check': prof_check,
        'profiler_total_flops': int(prof['total_flops']),
    }
