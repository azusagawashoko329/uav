"""Inference-only fused shared 2x3 depth projection; no model or loss changes.

Imported only when explicitly preparing fused CUDA FP32 inference. Triton is an
optional execution dependency; the normal training and cached-loop paths do not
import it. The output layout matches concatenated [group, direction, channel].
"""
import torch
import triton
import triton.language as tl

# Launchers contain compiled code, never tensor pointers or learned Q values.
# Keys include the CUDA device and every compile-time shape constant.
_launchers = {}


@triton.jit
def _project_triplets(x, q, out, total: tl.constexpr, channels: tl.constexpr,
                      groups: tl.constexpr, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < total
    channel = offset % channels
    direction = (offset // channels) % 2
    group = (offset // (channels * 2)) % groups
    patch = offset // (channels * 2 * groups)
    first = patch * groups * 3 * channels + group * 3 * channels + channel
    a = tl.load(x + first, valid, other=0)
    b = tl.load(x + first + channels, valid, other=0)
    c = tl.load(x + first + 2 * channels, valid, other=0)
    w0 = tl.load(q + direction * 3)
    w1 = tl.load(q + direction * 3 + 1)
    w2 = tl.load(q + direction * 3 + 2)
    # Explicit round-to-nearest instructions preserve the original separate
    # multiply/add order, including on Triton 2.1 without an FMA fusion flag.
    term0 = tl.math.mul_rn(w0, a)
    term1 = tl.math.mul_rn(w1, b)
    term2 = tl.math.mul_rn(w2, c)
    result = tl.math.add_rn(tl.math.add_rn(term0, term1), term2)
    tl.store(out + offset, result, valid)


def project_depth_rank2(normalized_taps, q):
    """Pack twelve tap tensors and project all patches/channels/groups at once."""
    if torch.is_grad_enabled():
        raise RuntimeError("Fused depth projection is inference-only")
    batch, patches, channels = normalized_taps[0].shape
    groups = len(normalized_taps) // 3
    # Flattened [tap, channel] has exactly the same storage order as stacking
    # along the tap axis, without the additional stack/shape-dispatch wrapper.
    packed = torch.cat(normalized_taps, dim=-1)
    output = torch.empty((batch, patches, groups * 2 * channels),
                         device=packed.device, dtype=packed.dtype)
    total = output.numel()
    device = packed.device.index
    if torch.cuda.current_device() == device:
        _launch(packed, q, output, total, channels, groups, device)
    else:
        with torch.cuda.device(device):
            _launch(packed, q, output, total, channels, groups, device)
    return output


def _launch(packed, q, output, total, channels, groups, device):
    # Reuse the compiled kernel's launcher instead of repeating JIT tensor
    # specialization, device selection and signature construction every frame.
    key = (device, total, channels, groups)
    launcher = _launchers.get(key)
    if launcher is None:
        grid = (triton.cdiv(total, 256), 1, 1)
        compiled = _project_triplets[grid](packed, q, output, total, channels,
                                         groups, 256, num_warps=4)
        _launchers[key] = compiled[grid]
    else:
        # The compiled launcher accepts runtime tensor arguments only;
        # total/channels/groups/block were already specialized at compilation.
        launcher(packed, q, output)
