from itertools import combinations

import torch
import torch.nn.functional as F

from losses.classification_loss import balanced_ce_loss
from losses.edge_loss import edge_bce_loss
from losses.hyperbolic_contrastive_loss import (
    hyperbolic_hierarchy_prototype_nce_loss,
    hyperbolic_subject_centroid_loss,
)
from models.edge_code import bernoulli_edge_code_logits
from utils.schedules import scheduled_value


def _edge_targets(labels, graph, dtype, device):
    return torch.tensor(
        [graph['edge_targets'][int(label)] for label in labels.detach().cpu().tolist()],
        dtype=dtype,
        device=device,
    )


def sibling_edge_contrastive_loss(edge_logits, labels, graph, min_parent_depth=0):
    """Contrast outgoing sibling edges only when the target path enters that node."""
    targets = _edge_targets(labels, graph, edge_logits.dtype, edge_logits.device)
    node_to_index = {name: index for index, name in enumerate(graph['nodes'])}
    outgoing = {}
    for edge_index, (parent, _) in enumerate(graph['edges']):
        outgoing.setdefault(node_to_index[parent], []).append(edge_index)

    group_losses = []
    node_depth = graph.get('node_depth', [0] * len(graph['nodes']))
    for parent_index, edge_indices in outgoing.items():
        if len(edge_indices) <= 1:
            continue
        if node_depth[parent_index] < int(min_parent_depth):
            continue
        indices = torch.tensor(edge_indices, dtype=torch.long, device=edge_logits.device)
        group_targets = targets.index_select(1, indices)
        valid = group_targets.sum(dim=1) == 1
        if valid.any().item():
            group_losses.append(
                F.cross_entropy(
                    edge_logits[valid].index_select(1, indices),
                    group_targets[valid].argmax(dim=1),
                )
            )
    if not group_losses:
        return edge_logits.new_tensor(0.0)
    return torch.stack(group_losses).mean()


def root_decision_loss(edge_logits, labels, graph, label_smoothing=0.0):
    """Classify neutral/positive/negative using only root outgoing edges."""
    root_name = graph.get('root_node')
    root_targets = graph.get('root_target_by_class')
    if root_name is None or root_targets is None:
        return edge_logits.new_tensor(0.0)

    root_edge_indices = [
        edge_index
        for edge_index, (parent, _) in enumerate(graph['edges'])
        if parent == root_name
    ]
    num_groups = max(int(target) for target in root_targets) + 1
    if not root_edge_indices or num_groups <= 1:
        return edge_logits.new_tensor(0.0)

    path_matrix = torch.as_tensor(
        graph['path_matrix'],
        dtype=edge_logits.dtype,
        device=edge_logits.device,
    )
    group_codes = []
    for group_index in range(num_groups):
        class_index = next(
            index
            for index, target in enumerate(root_targets)
            if int(target) == group_index
        )
        group_codes.append(path_matrix[class_index, root_edge_indices])
    codebook = torch.stack(group_codes, dim=0)
    root_logits = bernoulli_edge_code_logits(
        edge_logits[:, root_edge_indices],
        codebook,
    )
    targets = torch.as_tensor(
        root_targets,
        dtype=torch.long,
        device=labels.device,
    ).index_select(0, labels)
    return F.cross_entropy(
        root_logits,
        targets,
        label_smoothing=float(label_smoothing),
    )


def tree_validity_loss(edge_logits, graph):
    """Penalize mutually active siblings and children active without a parent."""
    probabilities = torch.sigmoid(edge_logits)
    node_to_index = {name: index for index, name in enumerate(graph['nodes'])}
    outgoing = {}
    incoming = {}
    for edge_index, (parent, child) in enumerate(graph['edges']):
        parent_index = node_to_index[parent]
        child_index = node_to_index[child]
        outgoing.setdefault(parent_index, []).append(edge_index)
        incoming[child_index] = edge_index

    sibling_terms = []
    for edge_indices in outgoing.values():
        for left, right in combinations(edge_indices, 2):
            sibling_terms.append(probabilities[:, left] * probabilities[:, right])

    implication_terms = []
    for edge_index, (parent, _) in enumerate(graph['edges']):
        parent_index = node_to_index[parent]
        if parent_index in incoming:
            implication_terms.append(
                F.relu(probabilities[:, edge_index] - probabilities[:, incoming[parent_index]])
            )

    zero = probabilities.new_tensor(0.0)
    sibling_loss = torch.stack(sibling_terms, dim=1).mean() if sibling_terms else zero
    implication_loss = torch.stack(implication_terms, dim=1).mean() if implication_terms else zero
    return sibling_loss + implication_loss


def compute_hyperbolic_edge_only_loss(
    outputs,
    labels,
    graph,
    cfg,
    subject_ids=None,
    epoch=None,
):
    weights = cfg.get('loss', {})
    edge_logits = outputs['edge_logits']
    edge_objective = weights.get('edge_objective', 'bce')
    edge_bce = edge_bce_loss(
        outputs['edge_weights'],
        labels,
        graph,
        edge_scores=edge_logits,
        positive_weight=weights.get('edge_positive_weight', 1.0),
        relation_balance=weights.get('edge_relation_balance', False),
        relation_balance_max_weight=weights.get('edge_relation_balance_max_weight', 4.0),
        class_balance=weights.get('edge_class_balance', False),
        class_balance_max_weight=weights.get('edge_class_balance_max_weight', 3.0),
        loss_weights=weights.get('edge_loss_weights'),
    )
    edge_code_ce = F.cross_entropy(
        outputs['edge_code_logits'],
        labels,
        label_smoothing=float(weights.get('edge_label_smoothing', 0.0)),
    )
    if edge_objective == 'bce':
        edge_loss = edge_bce
    elif edge_objective == 'code_ce':
        edge_loss = edge_code_ce
    elif edge_objective == 'hybrid':
        edge_loss = edge_bce + float(weights.get('edge_code_ce_weight', 0.2)) * edge_code_ce
    elif edge_objective == 'sibling_contrast':
        edge_loss = edge_bce + float(weights.get('edge_sibling_weight', 0.2)) * (
            sibling_edge_contrastive_loss(
                edge_logits,
                labels,
                graph,
                min_parent_depth=weights.get('edge_sibling_min_parent_depth', 0),
            )
        )
    elif edge_objective == 'node_edge_graph':
        label_smoothing = float(weights.get('edge_label_smoothing', 0.0))
        if weights.get('edge_graph_class_balance', False):
            graph_ce = balanced_ce_loss(
                outputs['edge_graph_logits'],
                labels,
                label_smoothing=label_smoothing,
                max_weight=weights.get('edge_graph_class_balance_max_weight', 3.0),
            )
        else:
            graph_ce = F.cross_entropy(
                outputs['edge_graph_logits'],
                labels,
                label_smoothing=label_smoothing,
            )
        root_ce = root_decision_loss(
            edge_logits,
            labels,
            graph,
            label_smoothing=label_smoothing,
        )
        child_ce = sibling_edge_contrastive_loss(
            edge_logits,
            labels,
            graph,
            min_parent_depth=1,
        )
        edge_loss = (
            graph_ce
            + float(weights.get('edge_bce_weight', 0.2)) * edge_bce
            + float(weights.get('edge_root_ce_weight', 0.0)) * root_ce
            + float(weights.get('edge_child_ce_weight', 0.0)) * child_ce
        )
    else:
        raise ValueError(f'Unknown edge_objective: {edge_objective}')

    hpcl = hyperbolic_hierarchy_prototype_nce_loss(
        outputs,
        labels,
        graph,
        temperature=weights.get('hp_temperature', 0.2),
        include_root=weights.get('hp_include_root_ancestor', False),
        leaf_weight=weights.get('hp_leaf_positive_weight', 1.0),
        ancestor_weight=weights.get('hp_ancestor_positive_weight', 0.3),
    )
    subject_centroid_weight = scheduled_value(
        weights.get('hp_subject_centroid_weight', 0.0),
        epoch,
        weights.get('hp_subject_centroid_schedule', 'constant'),
        weights.get('hp_subject_centroid_warmup_epochs', 0),
        weights.get('hp_subject_centroid_ramp_epochs', 0),
    )
    if subject_ids is not None and subject_centroid_weight > 0:
        hpcl = hpcl + subject_centroid_weight * hyperbolic_subject_centroid_loss(
            outputs,
            labels,
            subject_ids,
            graph,
            base_margin=weights.get('hp_subject_hierarchy_margin_base', 0.2),
            max_margin=weights.get('hp_subject_hierarchy_margin_max', 0.6),
            neutral_boost=weights.get('hp_subject_neutral_margin_boost', 0.1),
            branch_boost=weights.get('hp_subject_branch_margin_boost', 0.05),
            same_branch_scale=weights.get('hp_subject_same_branch_margin_scale', 0.5),
            pull_weight=weights.get('hp_subject_pull_weight', 1.0),
            margin_weight=weights.get('hp_subject_margin_weight', 0.5),
            compactness_weight=weights.get('hp_subject_compactness_weight', 0.1),
            pair_weight_base=weights.get('hp_subject_pair_weight_base', 1.0),
            neutral_positive_pair_weight=weights.get(
                'hp_subject_neutral_positive_pair_weight'
            ),
            neutral_negative_pair_weight=weights.get(
                'hp_subject_neutral_negative_pair_weight'
            ),
            positive_negative_pair_weight=weights.get(
                'hp_subject_positive_negative_pair_weight'
            ),
            cross_branch_pair_weight=weights.get(
                'hp_subject_cross_branch_pair_weight'
            ),
            same_branch_pair_weight=weights.get(
                'hp_subject_same_branch_pair_weight'
            ),
            same_negative_pair_weight=weights.get(
                'hp_subject_same_negative_pair_weight'
            ),
        )
    losses = {
        'edge': edge_loss,
        'tree': tree_validity_loss(edge_logits, graph),
        'hpcl': hpcl,
    }
    if (
        cfg.get('model', {}).get('use_domain_adversarial', False)
        and subject_ids is not None
        and 'domain_logits' in outputs
    ):
        losses['domain'] = F.cross_entropy(outputs['domain_logits'], subject_ids)
    else:
        losses['domain'] = edge_logits.new_tensor(0.0)

    losses['total'] = (
        float(weights.get('lambda_edge', 1.0)) * losses['edge']
        + float(weights.get('lambda_tree', 0.05)) * losses['tree']
        + float(weights.get('lambda_hpcl', 0.02)) * losses['hpcl']
        + float(weights.get('lambda_domain', 0.05)) * losses['domain']
    )
    return losses['total'], losses
