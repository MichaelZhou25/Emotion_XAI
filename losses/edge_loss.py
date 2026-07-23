import torch
import torch.nn.functional as F


def _class_balanced_sample_weights(labels, num_classes, max_weight, dtype):
    counts = torch.bincount(labels, minlength=num_classes).to(dtype)
    present = counts > 0
    if present.sum() <= 1:
        return None
    class_weights = torch.ones_like(counts)
    class_weights[present] = counts[present].sum() / (present.sum() * counts[present])
    class_weights = class_weights.clamp(max=float(max_weight))
    class_weights = class_weights / class_weights[present].mean().clamp_min(1e-8)
    return class_weights.index_select(0, labels)


def edge_bce_loss(
    edge_weights,
    labels,
    graph,
    edge_scores=None,
    positive_weight=None,
    relation_balance=False,
    relation_balance_max_weight=4.0,
    class_balance=False,
    class_balance_max_weight=3.0,
):
    targets = torch.tensor([graph['edge_targets'][int(y)] for y in labels.detach().cpu().tolist()],
                           dtype=edge_weights.dtype, device=edge_weights.device)
    if edge_scores is not None:
        pos_weight = None
        if relation_balance:
            positive = targets.sum(dim=0)
            negative = targets.shape[0] - positive
            present = positive > 0
            pos_weight = torch.ones_like(positive)
            pos_weight[present] = negative[present] / positive[present].clamp_min(1.0)
            max_weight = float(relation_balance_max_weight)
            pos_weight = pos_weight.clamp(min=1.0 / max_weight, max=max_weight)
        elif positive_weight is not None:
            pos_weight = torch.as_tensor(
                positive_weight,
                dtype=edge_scores.dtype,
                device=edge_scores.device,
            )
        per_edge = F.binary_cross_entropy_with_logits(
            edge_scores,
            targets,
            pos_weight=pos_weight,
            reduction='none',
        )
    else:
        per_edge = F.binary_cross_entropy(edge_weights, targets, reduction='none')

    per_sample = per_edge.mean(dim=1)
    if class_balance:
        sample_weights = _class_balanced_sample_weights(
            labels,
            graph['num_classes'],
            class_balance_max_weight,
            per_sample.dtype,
        )
        if sample_weights is not None:
            return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return per_sample.mean()
