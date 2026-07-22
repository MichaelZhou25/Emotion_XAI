import torch
import torch.nn.functional as F


def hierarchical_root_loss(outputs, labels, graph, label_smoothing=0.0):
    root_logits = outputs.get('root_logits')
    target_map = graph.get('root_target_by_class')
    if root_logits is None or target_map is None:
        return outputs['logits_final'].new_tensor(0.0)
    mapping = torch.tensor(target_map, dtype=torch.long, device=labels.device)
    root_targets = mapping.index_select(0, labels)
    return F.cross_entropy(
        root_logits,
        root_targets,
        label_smoothing=float(label_smoothing),
    )


def _balanced_class_weights(targets, num_classes, max_weight):
    counts = torch.bincount(targets, minlength=num_classes).float()
    present = counts > 0
    if present.sum() <= 1:
        return None
    weights = torch.ones_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights = weights.clamp(max=float(max_weight))
    return weights / weights[present].mean().clamp_min(1e-8)


def hierarchical_child_loss(
    outputs,
    labels,
    graph,
    label_smoothing=0.0,
    balance=True,
    max_weight=3.0,
):
    child_logits = outputs.get('child_logits')
    child_classes = graph.get('child_class_indices', [])
    if child_logits is None or not child_classes:
        return outputs['logits_final'].new_tensor(0.0)

    target_map = torch.full(
        (graph['num_classes'],),
        -1,
        dtype=torch.long,
        device=labels.device,
    )
    for child_target, class_index in enumerate(child_classes):
        target_map[int(class_index)] = child_target
    mapped_targets = target_map.index_select(0, labels)
    child_mask = mapped_targets >= 0
    if not child_mask.any().item():
        return child_logits.sum() * 0.0

    child_targets = mapped_targets[child_mask]
    class_weights = None
    if balance:
        class_weights = _balanced_class_weights(
            child_targets,
            child_logits.shape[1],
            max_weight=max_weight,
        )
    return F.cross_entropy(
        child_logits[child_mask],
        child_targets,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
