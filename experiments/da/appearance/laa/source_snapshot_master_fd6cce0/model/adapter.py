"""Layer-Attention Adapter (LAA) for the reverse-distillation appearance pipeline.

Replaces the hand-picked teacher taps [5,7,9,11] with a LEARNED per-layer weight `w = softmax(logits)`
over ALL 12 DINOv3 blocks (per-scene, since we train one model per scene). The teacher↔student reverse
one-to-one alignment is UNCHANGED — the student still reconstructs the RAW frozen teacher feature at
every layer; `w` sets HOW MUCH each layer's aligned top-k distance counts (LOSS + SCORE; see
model/loss.adapter_objective / anomaly_score_weighted). `w` is trained ONLY by the pseudo-anomaly
selection objective (detached on the reconstruction path), so it can't drift to easy-to-rebuild layers.

Two variants for where `w` is used:
  * variant B (input_agg=False, DEFAULT): OCBE input = the plain all-12 concat (its own per-tap LN
    handles scale, stationary, information-rich); `w` only re-weights the LOSS and the SCORE.
  * variant A (input_agg=True): `w` ALSO aggregates the OCBE input -> a single fused view
    sum_l w_l * LN_l(A_l) (B,P,768), so OCBE n_taps=1. Tests whether a selective INPUT adds anything on
    top of a selective LOSS. `w` is DETACHED in aggregate() too (selection trained only by pseudo-anomaly,
    never by reconstruction). NOTE B ckpts have no tap_norms -> input_agg gates them so load_state_dict
    stays compatible.
"""
import torch
import torch.nn as nn


class LayerAttnAdapter(nn.Module):
    def __init__(self, n_layers=12, embed_dim=768, input_agg=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_agg = input_agg
        # the selection knob: one logit per teacher layer -> softmax -> w. Init uniform (logits=0).
        self.logits = nn.Parameter(torch.zeros(n_layers))
        # per-layer LN only needed for variant A's input aggregation (B uses OCBE's own per-tap LN).
        if input_agg:
            self.tap_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

    @property
    def w(self):
        """(n_layers,) softmax selection weights. Differentiable; model/loss detaches it on the
        reconstruction path and trains it only via the pseudo-anomaly selection term."""
        return torch.softmax(self.logits, dim=0)

    def aggregate(self, taps):
        """variant A only: OCBE input = sum_l w_l * LN_l(A_l) -> (B,P,embed_dim). `w` is DETACHED here so
        the normal-reconstruction gradient through the input can't train the selection logits."""
        assert self.input_agg, "aggregate() requires input_agg=True (variant A)"
        normed = torch.stack([nrm(t) for nrm, t in zip(self.tap_norms, taps)], dim=0)  # (L,B,P,C)
        return (self.w.detach().view(-1, 1, 1, 1) * normed).sum(0)                       # (B,P,C)


def build_adapter(n_layers=12, embed_dim=768, input_agg=False):
    return LayerAttnAdapter(n_layers=n_layers, embed_dim=embed_dim, input_agg=input_agg)


if __name__ == '__main__':
    from bottleneck import build_bottleneck
    from student import build_student
    from loss import per_layer_topk_dist, adapter_objective, anomaly_score_weighted

    L, B, P, code_dim, depth, k = 12, 2, 196, 64, 12, 5
    taps = [torch.randn(B, P, 768) for _ in range(L)]
    for agg in (False, True):
        adapter = build_adapter(n_layers=L, input_agg=agg)
        ocbe = build_bottleneck(n_taps=1 if agg else L, code_dim=code_dim)
        student = build_student(n_taps=L, code_dim=code_dim, depth=depth)
        code = ocbe([adapter.aggregate(taps)]) if agg else ocbe(taps)
        recon = student(code)
        t_norm = per_layer_topk_dist(taps, recon, k=k)
        loss, logs = adapter_objective(adapter.w, t_norm, t_norm)
        n_ad = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f'variant {"A" if agg else "B"} | adapter params={n_ad} | OCBE in n_taps={1 if agg else L} '
              f'-> {len(recon)} recons of {tuple(recon[0].shape)} | loss={loss.item():.4f}')
