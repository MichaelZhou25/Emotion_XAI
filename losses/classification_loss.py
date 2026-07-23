import torch
import torch.nn.functional as F


def ce_loss(logits, labels, label_smoothing=0.0):
    return F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))


def balanced_ce_loss(logits, labels, label_smoothing=0.0, max_weight=3.0):
    counts = torch.bincount(labels, minlength=logits.shape[1]).to(logits.dtype)
    present = counts > 0
    if present.sum() <= 1:
        return ce_loss(logits, labels, label_smoothing)
    weights = torch.ones_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights = weights.clamp(max=float(max_weight))
    weights = weights / weights[present].mean().clamp_min(1e-8)
    return F.cross_entropy(
        logits,
        labels,
        weight=weights,
        label_smoothing=float(label_smoothing),
    )
