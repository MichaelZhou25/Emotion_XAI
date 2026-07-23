import torch
import torch.nn.functional as F


def bernoulli_edge_code_logits(edge_logits, codebook, temperature=1.0):
    """Decode independent edge logits with a fixed class-edge codebook."""
    codebook = torch.as_tensor(
        codebook,
        dtype=edge_logits.dtype,
        device=edge_logits.device,
    )
    if edge_logits.ndim != 2 or codebook.ndim != 2:
        raise ValueError('edge_logits and codebook must both be rank-2 tensors')
    if edge_logits.shape[1] != codebook.shape[1]:
        raise ValueError('edge_logits and codebook must contain the same number of edges')

    log_active = F.logsigmoid(edge_logits)
    log_inactive = F.logsigmoid(-edge_logits)
    class_log_likelihood = (
        torch.matmul(log_active, codebook.t())
        + torch.matmul(log_inactive, (1.0 - codebook).t())
    )
    return class_log_likelihood / max(float(temperature), 1e-6)
