import torch
import torch.nn.functional as F


def concept_mse_loss(concept_scores, labels, graph):
    targets = torch.tensor([graph['concept_targets'][int(y)] for y in labels.detach().cpu().tolist()],
                           dtype=concept_scores.dtype, device=concept_scores.device)
    mask = graph['concept_mask_tensor'].to(concept_scores.device).to(concept_scores.dtype).view(1, -1)
    loss = (concept_scores - targets).pow(2) * mask
    return loss.sum() / (mask.sum() * concept_scores.shape[0] + 1e-8)
