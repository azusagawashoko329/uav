"""Topk reconstruction loss + per-frame anomaly score for the reverse-distillation pipeline.

The teacher tap feature A_k and the student reconstruction A'_k are compared per patch by COSINE
distance (magnitude-invariant). The Topk operator keeps only the k HARDEST patches (largest distance)
per tap — focusing training on where the student fails (= hard-example mining, EfficientAD-style; and
the patch-localized read-out our prior analysis found strongest). Same operator is used at train time
(reduced to a scalar over the batch) and at test time (kept per-frame as the anomaly score).

NOTE: this is the Topk objective only — there is NO CLS/token (LL) term in this pipeline; the 5 prefix
tokens are not distilled. train-k and score-k may differ (k-mismatch is a known knob).
"""
import torch
import torch.nn.functional as F


def _per_patch_cosine_dist(A, Ahat):
    """A, Ahat: (B, P, C) -> (B, P) cosine distance per patch (1 - cos)."""
    return 1.0 - F.cosine_similarity(A, Ahat, dim=-1)


def topk_loss(teacher_taps, student_recons, k=5):
    """Training loss on NORMAL data: per tap, mean of the k hardest patches; averaged over taps & batch."""
    loss = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        loss = loss + d.topk(k, dim=-1).values.mean(-1).mean()    # top-k patches -> mean -> batch mean
    return loss / len(teacher_taps)


def full_loss(teacher_taps, student_recons):
    """no-topk ablation = MADViT-style FLATTEN-then-cosine: flatten each tap's (B,P,D) feature map to a
    single (B, P*D) vector and take ONE cosine distance (teacher vs student) — NOT a per-patch mean.
    Averaged over taps & batch. Matched score = the 'flat' read-out (same flatten-cosine)."""
    loss = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = 1.0 - F.cosine_similarity(A.reshape(A.size(0), -1), Ahat.reshape(Ahat.size(0), -1), dim=1)  # (B,)
        loss = loss + d.mean()                                    # batch mean
    return loss / len(teacher_taps)


@torch.no_grad()
def anomaly_score(teacher_taps, student_recons, k=5):
    """Test-time PER-FRAME anomaly score: same Topk operator, kept per sample -> (B,)."""
    score = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        score = score + d.topk(k, dim=-1).values.mean(-1)         # (B,)
    return score / len(teacher_taps)


# ----------------------------------------------------------------------------------------------------
# Layer-Attention Adapter (LAA) variants — top-k stays IDENTICAL (per-layer, over patches); the only
# change is the OUTER reduce over layers: uniform mean over fixed taps  ->  adapter-weighted sum over
# all 12 layers. See model/adapter.py for the design (collapse-safe targets, non-cheating selection).
# ----------------------------------------------------------------------------------------------------
def per_layer_topk_dist(teacher_taps, student_recons, k=5):
    """Per-sample, per-layer top-k cosine distance -> (B, L). This is exactly the existing top-k
    read-out (k hardest patches per layer), just kept SEPARATELY per layer so the adapter can weight
    across layers. Carries gradient (trains student/OCBE/adapter-LN)."""
    cols = []
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        cols.append(d.topk(k, dim=-1).values.mean(-1))            # (B,)
    return torch.stack(cols, dim=1)                               # (B, L)


def adapter_objective(w, t_norm, t_pseudo=None, tau=1.0, lam_sel=1.0, lam_ent=0.01, eps=1e-8, gap_zscore=False):
    """Total LAA training objective. `w`: (L,) softmax selection weights (from adapter, differentiable).
    `t_norm`/`t_pseudo`: (B, L) per-layer top-k distances on normal / pseudo-anomaly frames.

    Three terms, with deliberate gradient routing:
      L_normal : w-weighted reconstruction on normal frames. `w` is DETACHED -> trains ONLY OCBE +
                 student + adapter-LN, never the selection logits (no easy-layer gaming).
      L_select : drives `w` toward layers with the largest normal-vs-pseudo gap g_l = t_pseudo - t_norm
                 (KL(target||w), target = softmax(g/tau)). g is DETACHED -> trains ONLY the logits.
      L_ent    : -lam_ent * H(w) keeps `w` from collapsing onto a single layer.
    If `t_pseudo` is None (skeleton / no-pseudo run) only L_normal is used and `w` stays at init."""
    L_normal = (w.detach() * t_norm).sum(1).mean()
    loss = L_normal
    logs = {'L_normal': float(L_normal)}
    if t_pseudo is not None:
        g = (t_pseudo.mean(0) - t_norm.mean(0)).detach()          # (L,) per-layer discriminability
        # z-score-standardizing g spreads w (0.001 -> 0.01..0.1) = aggressive per-layer selection. Default OFF
        # (plain softmax -> near-uniform w, the better model on Drone-Anomaly; zscore HURT 85.78->83.73, worst
        # on appearance-null vehicle). --gap_zscore re-enables it (TODO3 "放开非均匀" exploration; see note/LAA.md).
        if gap_zscore:
            g = (g - g.mean()) / (g.std() + eps)
        target = torch.softmax(g / tau, dim=0)                    # (L,) desired selection
        L_select = (target * (torch.log(target + eps) - torch.log(w + eps))).sum()   # KL(target||w)
        L_ent = (w * torch.log(w + eps)).sum()                    # = -H(w); minimizing -> toward uniform
        loss = loss + lam_sel * L_select + lam_ent * L_ent
        logs.update(L_select=float(L_select), H=float(-L_ent))
    return loss, logs


@torch.no_grad()
def anomaly_score_weighted(teacher_taps, student_recons, w, k=5):
    """Test-time per-frame score for the adapter pipeline: adapter-weighted sum of the per-layer top-k
    distances -> (B,). Same top-k, same `w` as training; no pseudo-anomaly at test."""
    t = per_layer_topk_dist(teacher_taps, student_recons, k=k)    # (B, L)
    return (w.detach().view(1, -1) * t).sum(1)                    # (B,)


@torch.no_grad()
def anomaly_readouts_weighted(teacher_taps, student_recons, w, k=5):
    """LAA counterpart of anomaly_readouts: SAME four read-outs, but combined over layers by the learned
    weights `w` (sum_l w_l * .) instead of a uniform mean. 'topk' is the headline (= anomaly_score_weighted,
    training-consistent); 'max'/'mean'/'flat' are diagnostic variants. Returns dict of (B,) tensors."""
    w = w.detach().reshape(-1)
    topk = mx = mn = fl = 0.0
    for li, (A, Ahat) in enumerate(zip(teacher_taps, student_recons)):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        topk = topk + w[li] * d.topk(k, dim=-1).values.mean(-1)
        mx = mx + w[li] * d.max(dim=-1).values
        mn = mn + w[li] * d.mean(dim=-1)
        fl = fl + w[li] * (1.0 - F.cosine_similarity(A.reshape(A.size(0), -1), Ahat.reshape(Ahat.size(0), -1), dim=1))
    return {'topk': topk, 'max': mx, 'mean': mn, 'flat': fl}


@torch.no_grad()
def anomaly_readouts(teacher_taps, student_recons, k=5):
    """Per-frame scores under several read-outs of the reconstruction discrepancy (each averaged over
    taps). 'topk' (k) is the training-consistent score for the top-k objective; 'max'/'mean' are
    per-patch variants; 'flat' = MADViT-style flatten-then-cosine (the matched score for the
    --loss full ablation). Returns dict of (B,) tensors."""
    topk = mx = mn = fl = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        topk = topk + d.topk(k, dim=-1).values.mean(-1)
        mx = mx + d.max(dim=-1).values
        mn = mn + d.mean(dim=-1)
        fl = fl + (1.0 - F.cosine_similarity(A.reshape(A.size(0), -1), Ahat.reshape(Ahat.size(0), -1), dim=1))
    n = len(teacher_taps)
    return {'topk': topk / n, 'max': mx / n, 'mean': mn / n, 'flat': fl / n}
