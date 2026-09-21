"""DLBM v007: learned opening-transition statistic.

The network learns the same primitive exposed by the manual Gate: whether an
opened suffix has a structured same-frame relation to the deep base and whether
opening changes pairwise frame ordering.  It never receives scene boundaries or
the manual rank/tail/z-score formulae.
"""
import torch
import torch.nn as nn


class LearnedOpeningTransition(nn.Module):
    def __init__(self, num_layers=12, topk=5, model_dim=32, heads=4, dropout=0.0):
        super().__init__()
        if num_layers < 2 or topk < 1 or model_dim % heads:
            raise ValueError('invalid layer/top-k/model dimensions')
        self.num_layers = int(num_layers)
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(4, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.role_embedding = nn.Parameter(torch.empty(2, model_dim))
        self.cross_relation = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True,
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim),
        )
        # Normal/Pseudo are fused by a learned head; no fixed branch delta is used.
        self.realness_head = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, model_dim),
            nn.GELU(), nn.Linear(model_dim, 1),
        )
        # Dense RankNet supervision learns the two score axes whose relative ordering
        # is the primitive used by the manual Gate.  The heads see learned transition
        # embeddings rather than hand-computed scene statistics.
        self.base_order_head = nn.Linear(model_dim, 1)
        self.open_order_head = nn.Linear(model_dim, 1)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def _topk_tokens(self, maps):
        if maps.ndim != 3 or maps.shape[1] != self.num_layers:
            raise ValueError(
                f'expected maps (frames,{self.num_layers},patches), got {tuple(maps.shape)}')
        if maps.shape[0] < 2 or maps.shape[2] < self.topk:
            raise ValueError('at least two frames and top-k patches are required')
        if not torch.isfinite(maps).all():
            raise FloatingPointError('non-finite discrepancy maps')
        side = int(round(maps.shape[2] ** 0.5))
        if side * side != maps.shape[2]:
            raise ValueError(f'patch count must form a square grid, got {maps.shape[2]}')
        values, indices = maps.topk(self.topk, dim=-1)
        denom = float(max(1, side - 1))
        x = (indices % side).to(values.dtype) / denom
        y = torch.div(indices, side, rounding_mode='floor').to(values.dtype) / denom
        rank = torch.arange(self.topk, device=maps.device, dtype=values.dtype)
        rank = rank.view(1, 1, -1).expand_as(values) / float(max(1, self.topk - 1))
        return self.token_encoder(torch.stack((values, x, y, rank), dim=-1))

    def encode_branch(self, maps, boundary, opened_permutation=None):
        """Encode base L11 versus the actual opened layers ``boundary..L10`` per frame."""
        boundary = int(boundary)
        if not 0 <= boundary < self.num_layers - 1:
            raise ValueError(f'opening boundary must lie in [0,{self.num_layers - 2}]')
        tokens = self._topk_tokens(maps)
        frames = tokens.shape[0]
        base = (
            tokens[:, -1]
            + self.layer_embedding[-1].view(1, 1, -1)
            + self.role_embedding[1].view(1, 1, -1)
        )
        opened = tokens[:, boundary:-1]
        if opened_permutation is not None:
            permutation = torch.as_tensor(
                opened_permutation, dtype=torch.long, device=maps.device,
            )
            if permutation.shape != (frames,):
                raise ValueError('opened permutation must contain one index per frame')
            opened = opened.index_select(0, permutation)
        layer = self.layer_embedding[boundary:-1].view(
            1, self.num_layers - boundary - 1, 1, self.model_dim,
        )
        opened = opened + layer + self.role_embedding[0].view(1, 1, 1, -1)
        opened = opened.reshape(frames, -1, self.model_dim)
        related, _ = self.cross_relation(opened, base, base, need_weights=False)
        opened_summary = (opened + related).mean(dim=1)
        base_summary = base.mean(dim=1)
        result = self.transition(torch.cat((base_summary, opened_summary), dim=-1))
        if not torch.isfinite(result).all():
            raise FloatingPointError('non-finite transition embedding')
        return result

    def encode(self, normal_maps, pseudo_maps, boundary, opened_permutation=None):
        if normal_maps.shape != pseudo_maps.shape:
            raise ValueError('normal/pseudo map shape mismatch')
        normal = self.encode_branch(normal_maps, boundary, opened_permutation)
        pseudo = self.encode_branch(pseudo_maps, boundary, opened_permutation)
        return normal, pseudo

    def realness_logits(self, normal_embedding, pseudo_embedding):
        if normal_embedding.shape != pseudo_embedding.shape:
            raise ValueError('normal/pseudo embedding shape mismatch')
        return self.realness_head(
            torch.cat((normal_embedding, pseudo_embedding), dim=-1),
        ).squeeze(-1)

    def order_scores(self, embedding):
        return (self.base_order_head(embedding).squeeze(-1),
                self.open_order_head(embedding).squeeze(-1))


def boundary_from_evidence(evidence):
    """Return the legal suffix with maximum held-out learned evidence; L11 is zero baseline."""
    if evidence.ndim != 1 or evidence.numel() < 1 or not torch.isfinite(evidence).all():
        raise ValueError('expected finite boundary evidence vector')
    complete = torch.cat((evidence, evidence.new_zeros(1)))
    return int(complete.argmax().item()), complete
