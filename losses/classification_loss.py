import torch.nn.functional as F


def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels)
