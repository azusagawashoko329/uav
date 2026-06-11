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


@torch.no_grad()
def anomaly_score(teacher_taps, student_recons, k=5):
    """Test-time PER-FRAME anomaly score: same Topk operator, kept per sample -> (B,)."""
    score = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        score = score + d.topk(k, dim=-1).values.mean(-1)         # (B,)
    return score / len(teacher_taps)


@torch.no_grad()
def anomaly_readouts(teacher_taps, student_recons, k=5):
    """Per-frame scores under 3 spatial read-outs of the SAME per-patch reconstruction discrepancy
    (each averaged over taps). 'topk' (k) is the training-consistent score (== anomaly_score);
    'max'/'mean' are analysis variants of the identical discrepancy. Returns dict of (B,) tensors."""
    topk = mx = mn = 0.0
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)                       # (B, P)
        topk = topk + d.topk(k, dim=-1).values.mean(-1)
        mx = mx + d.max(dim=-1).values
        mn = mn + d.mean(dim=-1)
    n = len(teacher_taps)
    return {'topk': topk / n, 'max': mx / n, 'mean': mn / n}
