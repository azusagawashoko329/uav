"""Four-stage reverse decoder with optional zero-initialized inter-stage adapters."""
import torch
from torch import nn

from .student import Block


class ChannelAdapter(nn.Module):
    def __init__(self, dim=768, hidden=96):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class SpatialAdapter(ChannelAdapter):
    def __init__(self, dim=768, hidden=96):
        super().__init__(dim, hidden)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)

    def forward(self, x):
        b, p, _ = x.shape
        side = int(p ** 0.5)
        if side * side != p:
            raise ValueError(f"spatial adapter requires square patch grid, got {p}")
        h = self.act(self.down(self.norm(x)))
        h = h.transpose(1, 2).reshape(b, -1, side, side)
        h = self.act(self.spatial(h)).flatten(2).transpose(1, 2)
        return x + self.up(h)


class FourBlockTwelveHeadDecoder(nn.Module):
    """Teacher groups 0..3 map to student stages 3..0, three heads per stage."""
    def __init__(self, adapter="none", dim=768, code_dim=64, hidden=96):
        super().__init__()
        if adapter not in ("none", "channel", "spatial"):
            raise ValueError(adapter)
        self.proj_in = nn.Linear(code_dim, dim)
        self.blocks = nn.ModuleList(Block(dim, 8) for _ in range(4))
        # Keep shared parameter initialization identical across adapter variants.
        self.heads = nn.ModuleList(nn.Linear(dim, dim) for _ in range(12))
        cls = {"none": nn.Identity, "channel": ChannelAdapter,
               "spatial": SpatialAdapter}[adapter]
        self.adapters = nn.ModuleList(cls() if adapter == "none" else cls(dim, hidden)
                                      for _ in range(3))

    def forward(self, code):
        x = self.proj_in(code)
        out = [None] * 12
        for stage, block in enumerate(self.blocks):
            x = block(x)
            group = 3 - stage
            for li in range(group * 3, group * 3 + 3):
                out[li] = self.heads[li](x)
            if stage < 3:
                x = self.adapters[stage](x)
        return out
