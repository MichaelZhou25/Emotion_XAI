import torch


def masked_mean(x, mask, dim=None, eps=1e-8):
    mask = mask.to(dtype=x.dtype, device=x.device)
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))
