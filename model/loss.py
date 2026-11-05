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


def matched_score(teacher_taps, student_recons, k=5, alpha=0.5):
    """Per-sample score matching the training objective in scripts/uit_appearance_runner.loss_for.

    V0 (4 taps): mean over taps of the per-tap top-k mean.
    V1-V5 (12 taps): four adjacent triplets; each triplet score is
        0.5 * mean_tap(per-tap top-k mean) + 0.5 * max_tap(per-tap top-k mean),
    then mean over the four groups.  Returns (B,).
    """
    per = []
    for A, Ahat in zip(teacher_taps, student_recons):
        d = _per_patch_cosine_dist(A, Ahat)
        per.append(d.topk(k, dim=-1).values.mean(-1))
    n = len(per)
    if n == 4:
        return torch.stack(per, dim=1).mean(dim=1)
    if n % 3 == 0:
        groups = torch.stack(per, dim=1).reshape(per[0].size(0), n // 3, 3)
        avg = groups.mean(-1)
        worst = groups.max(-1).values
        return (avg + alpha * (worst - avg)).mean(dim=1)
    raise ValueError(f'matched_score expects 4 or 3k taps, got {n}')


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
