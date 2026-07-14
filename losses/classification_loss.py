import torch.nn.functional as F


def ce_loss(logits, labels, label_smoothing=0.0):
    return F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
