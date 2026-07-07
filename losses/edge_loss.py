import torch
import torch.nn.functional as F


def edge_bce_loss(edge_weights, labels, graph):
    targets = torch.tensor([graph['edge_targets'][int(y)] for y in labels.detach().cpu().tolist()],
                           dtype=edge_weights.dtype, device=edge_weights.device)
    return F.binary_cross_entropy(edge_weights, targets)
