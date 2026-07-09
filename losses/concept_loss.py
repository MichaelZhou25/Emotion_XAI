import torch
import torch.nn.functional as F


def concept_mse_loss(concept_scores, labels, graph):
    targets = torch.tensor([graph['concept_targets'][int(y)] for y in labels.detach().cpu().tolist()],
                           dtype=concept_scores.dtype, device=concept_scores.device)
    mask = graph['concept_mask_tensor'].to(concept_scores.device).to(concept_scores.dtype).view(1, -1)
    if targets.shape[1] != concept_scores.shape[1]:
        target_dim = concept_scores.shape[1]
        src_dim = targets.shape[1]
        if src_dim < target_dim:
            pad = target_dim - src_dim
            targets = F.pad(targets, (0, pad), value=0.0)
            mask = F.pad(mask, (0, pad), value=0.0)
        else:
            targets = targets[:, :target_dim]
            mask = mask[:, :target_dim]
    loss = (concept_scores - targets).pow(2) * mask
    return loss.sum() / (mask.sum() * concept_scores.shape[0] + 1e-8)
