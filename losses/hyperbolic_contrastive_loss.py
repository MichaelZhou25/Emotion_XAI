import torch
import torch.nn.functional as F

from models.hyperbolic_prototype import expmap0, poincare_distance


def _node_maps(graph):
    nodes = graph['nodes']
    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    parents = {}
    neighbors = {idx: [] for idx in range(len(nodes))}
    for parent, child in graph.get('edges', []):
        parent_idx = node_to_idx[parent]
        child_idx = node_to_idx[child]
        parents[child_idx] = parent_idx
        neighbors[parent_idx].append(child_idx)
        neighbors[child_idx].append(parent_idx)
    return node_to_idx, parents, neighbors


def _class_prototypes(prototypes, graph, device):
    class_indices = torch.tensor(
        graph.get('class_node_indices', list(range(graph['num_classes']))),
        dtype=torch.long,
        device=device,
    )
    return prototypes.index_select(0, class_indices)


def _class_node_indices(graph):
    return [int(i) for i in graph.get('class_node_indices', list(range(graph['num_classes'])))]


def _ancestor_indices(graph, node_idx, include_root=False):
    _, parents, _ = _node_maps(graph)
    depths = graph.get('node_depth', [0] * len(graph['nodes']))
    ancestors = []
    current = int(node_idx)
    while current in parents:
        current = parents[current]
        if include_root or depths[current] > 0:
            ancestors.append(current)
    return ancestors


def _class_positive_node_mask(graph, device, include_root=False):
    class_nodes = _class_node_indices(graph)
    mask = torch.zeros(graph['num_classes'], graph['num_nodes'], dtype=torch.bool, device=device)
    for class_idx, node_idx in enumerate(class_nodes):
        positive_nodes = [node_idx] + _ancestor_indices(graph, node_idx, include_root=include_root)
        mask[class_idx, positive_nodes] = True
    return mask


def _class_positive_node_weights(
    graph,
    device,
    include_root=False,
    leaf_weight=1.0,
    ancestor_weight=0.3,
):
    class_nodes = _class_node_indices(graph)
    weights = torch.zeros(graph['num_classes'], graph['num_nodes'], dtype=torch.float32, device=device)
    for class_idx, node_idx in enumerate(class_nodes):
        weights[class_idx, node_idx] = float(leaf_weight)
        for ancestor_idx in _ancestor_indices(graph, node_idx, include_root=include_root):
            weights[class_idx, ancestor_idx] = float(ancestor_weight)
    return weights


def _tree_distance_matrix(graph, device):
    class_nodes = _class_node_indices(graph)
    _, _, neighbors = _node_maps(graph)
    distances = []
    for start in class_nodes:
        queue = [(start, 0)]
        seen = {start}
        node_dist = {}
        while queue:
            node_idx, dist = queue.pop(0)
            node_dist[node_idx] = dist
            for nxt in neighbors[node_idx]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        distances.append([node_dist[end] for end in class_nodes])
    return torch.tensor(distances, dtype=torch.float32, device=device)


def _class_depths_and_branches(graph, device):
    class_nodes = _class_node_indices(graph)
    _, parents, _ = _node_maps(graph)
    depths = graph.get('node_depth', [0] * len(graph['nodes']))
    class_depths = []
    top_branches = []
    for node_idx in class_nodes:
        current = int(node_idx)
        class_depths.append(float(depths[current]))
        if depths[current] == 0:
            top_branches.append(-1)
            continue
        while current in parents and depths[parents[current]] > 0:
            current = parents[current]
        top_branches.append(current)
    return (
        torch.tensor(class_depths, dtype=torch.float32, device=device),
        torch.tensor(top_branches, dtype=torch.long, device=device),
    )


def _class_valence(graph, device):
    concept_matrix = graph.get('concept_matrix')
    if not concept_matrix:
        return torch.zeros(graph['num_classes'], dtype=torch.float32, device=device)

    concept = torch.tensor(concept_matrix, dtype=torch.float32, device=device)
    if concept.ndim < 2 or concept.shape[1] == 0:
        return torch.zeros(graph['num_classes'], dtype=torch.float32, device=device)

    if concept.shape[0] == graph['num_nodes']:
        class_indices = torch.tensor(_class_node_indices(graph), dtype=torch.long, device=device)
        return concept.index_select(0, class_indices)[:, 0]
    return concept[:graph['num_classes'], 0]


def _hierarchy_margin_matrix(
    graph,
    device,
    base_margin=0.2,
    max_margin=0.6,
    neutral_boost=0.2,
    branch_boost=0.1,
    same_branch_scale=0.5,
):
    tree_dist = _tree_distance_matrix(graph, device)
    class_depths, top_branches = _class_depths_and_branches(graph, device)
    margin = float(base_margin) * tree_dist

    non_root = class_depths > 0
    same_branch = (
        (top_branches.view(-1, 1) == top_branches.view(1, -1))
        & non_root.view(-1, 1)
        & non_root.view(1, -1)
    )
    margin = torch.where(same_branch, margin * float(same_branch_scale), margin)

    neutral_pair = (class_depths.view(-1, 1) == 0) ^ (class_depths.view(1, -1) == 0)
    cross_branch = (
        (top_branches.view(-1, 1) != top_branches.view(1, -1))
        & non_root.view(-1, 1)
        & non_root.view(1, -1)
    )
    margin = margin + float(neutral_boost) * neutral_pair.float()
    margin = margin + float(branch_boost) * cross_branch.float()
    margin = margin.clamp(max=float(max_margin))
    margin.fill_diagonal_(0.0)
    return margin


def _hierarchy_pair_weight_matrix(
    graph,
    device,
    base_weight=1.0,
    neutral_pair_weight=None,
    neutral_positive_pair_weight=None,
    neutral_negative_pair_weight=None,
    cross_branch_pair_weight=None,
    positive_negative_pair_weight=None,
    same_branch_pair_weight=None,
    same_negative_pair_weight=None,
):
    num_classes = graph['num_classes']
    weights = torch.full((num_classes, num_classes), float(base_weight), dtype=torch.float32, device=device)
    class_depths, top_branches = _class_depths_and_branches(graph, device)
    valence = _class_valence(graph, device)

    non_root = class_depths > 0
    neutral = (class_depths == 0) | (valence.abs() < 1e-6)
    positive = valence > 0
    negative = valence < 0

    neutral_pair = neutral.view(-1, 1) ^ neutral.view(1, -1)
    neutral_positive = (
        (neutral.view(-1, 1) & positive.view(1, -1))
        | (positive.view(-1, 1) & neutral.view(1, -1))
    )
    neutral_negative = (
        (neutral.view(-1, 1) & negative.view(1, -1))
        | (negative.view(-1, 1) & neutral.view(1, -1))
    )
    cross_branch = (
        (top_branches.view(-1, 1) != top_branches.view(1, -1))
        & non_root.view(-1, 1)
        & non_root.view(1, -1)
    )
    positive_negative = (
        (positive.view(-1, 1) & negative.view(1, -1))
        | (negative.view(-1, 1) & positive.view(1, -1))
    )
    same_branch = (
        (top_branches.view(-1, 1) == top_branches.view(1, -1))
        & non_root.view(-1, 1)
        & non_root.view(1, -1)
    )
    same_negative = negative.view(-1, 1) & negative.view(1, -1) & same_branch

    if neutral_pair_weight is not None:
        weights = torch.where(neutral_pair, weights.new_full(weights.shape, float(neutral_pair_weight)), weights)
    if same_branch_pair_weight is not None:
        weights = torch.where(same_branch, weights.new_full(weights.shape, float(same_branch_pair_weight)), weights)
    if cross_branch_pair_weight is not None:
        weights = torch.where(cross_branch, weights.new_full(weights.shape, float(cross_branch_pair_weight)), weights)
    if positive_negative_pair_weight is not None:
        weights = torch.where(positive_negative, weights.new_full(weights.shape, float(positive_negative_pair_weight)), weights)
    if neutral_positive_pair_weight is not None:
        weights = torch.where(neutral_positive, weights.new_full(weights.shape, float(neutral_positive_pair_weight)), weights)
    if neutral_negative_pair_weight is not None:
        weights = torch.where(neutral_negative, weights.new_full(weights.shape, float(neutral_negative_pair_weight)), weights)
    if same_negative_pair_weight is not None:
        weights = torch.where(same_negative, weights.new_full(weights.shape, float(same_negative_pair_weight)), weights)

    weights.fill_diagonal_(0.0)
    return weights


def hyperbolic_prototype_nce_loss(outputs, labels, graph, temperature=0.2):
    z = outputs['proto_embedding']
    prototypes = _class_prototypes(outputs['prototypes'], graph, z.device)
    dist = poincare_distance(z, prototypes)
    logits = -dist / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, labels)


def hyperbolic_hierarchy_prototype_nce_loss(
    outputs,
    labels,
    graph,
    temperature=0.2,
    include_root=False,
    leaf_weight=1.0,
    ancestor_weight=0.3,
):
    z = outputs['proto_embedding']
    prototypes = outputs['prototypes']
    dist = poincare_distance(z, prototypes)
    logits = -dist / max(float(temperature), 1e-6)
    positive_weights = _class_positive_node_weights(
        graph,
        z.device,
        include_root=include_root,
        leaf_weight=leaf_weight,
        ancestor_weight=ancestor_weight,
    ).index_select(0, labels)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_weight_sum = positive_weights.sum(dim=1).clamp_min(1e-8)
    return -(positive_weights * log_prob).sum(dim=1).div(positive_weight_sum).mean()


def hyperbolic_supcon_loss(outputs, labels, temperature=0.2):
    z = outputs['proto_embedding']
    batch_size = z.shape[0]
    if batch_size <= 1:
        return z.new_tensor(0.0)

    dist = poincare_distance(z, z)
    logits = -dist / max(float(temperature), 1e-6)
    eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    valid_mask = ~eye
    positive_mask = (labels.view(-1, 1) == labels.view(1, -1)) & valid_mask
    positive_count = positive_mask.sum(dim=1)
    anchor_mask = positive_count > 0
    if not anchor_mask.any().item():
        return z.new_tensor(0.0)

    logits = logits - logits.masked_fill(~valid_mask, -1e4).max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * valid_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -pos_log_prob[anchor_mask].mean()


def hyperbolic_subject_centroid_loss(
    outputs,
    labels,
    subject_ids,
    graph,
    base_margin=0.2,
    max_margin=0.6,
    neutral_boost=0.2,
    branch_boost=0.1,
    same_branch_scale=0.5,
    pull_weight=1.0,
    margin_weight=0.5,
    compactness_weight=0.1,
    pair_weight_base=1.0,
    neutral_positive_pair_weight=None,
    neutral_negative_pair_weight=None,
    positive_negative_pair_weight=None,
    cross_branch_pair_weight=None,
    same_branch_pair_weight=None,
    same_negative_pair_weight=None,
):
    z = outputs['proto_embedding']
    tangent = outputs['proto_tangent']
    prototypes = _class_prototypes(outputs['prototypes'], graph, z.device)
    margin_matrix = _hierarchy_margin_matrix(
        graph,
        z.device,
        base_margin=base_margin,
        max_margin=max_margin,
        neutral_boost=neutral_boost,
        branch_boost=branch_boost,
        same_branch_scale=same_branch_scale,
    )
    pair_weights = _hierarchy_pair_weight_matrix(
        graph,
        z.device,
        base_weight=pair_weight_base,
        neutral_positive_pair_weight=neutral_positive_pair_weight,
        neutral_negative_pair_weight=neutral_negative_pair_weight,
        positive_negative_pair_weight=positive_negative_pair_weight,
        cross_branch_pair_weight=cross_branch_pair_weight,
        same_branch_pair_weight=same_branch_pair_weight,
        same_negative_pair_weight=same_negative_pair_weight,
    )

    if labels.numel() == 0:
        return z.new_tensor(0.0)

    num_classes = graph['num_classes']
    group_ids = subject_ids.long() * num_classes + labels.long()
    unique_groups, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    num_groups = unique_groups.numel()
    counts = torch.bincount(inverse, minlength=num_groups).to(tangent.dtype).clamp_min(1.0)

    centroid_tangent = tangent.new_zeros((num_groups, tangent.shape[-1]))
    centroid_tangent.index_add_(0, inverse, tangent)
    centroid_tangent = centroid_tangent / counts.unsqueeze(1)
    centroids = expmap0(centroid_tangent)
    group_classes = unique_groups.remainder(num_classes).long()

    centroid_dist = poincare_distance(centroids, prototypes)
    positive_dist = centroid_dist.gather(1, group_classes.unsqueeze(1)).squeeze(1)
    group_margins = margin_matrix.index_select(0, group_classes)
    group_weights = pair_weights.index_select(0, group_classes)
    negative_mask = torch.ones_like(group_weights, dtype=torch.bool)
    negative_mask.scatter_(1, group_classes.unsqueeze(1), False)
    active_weights = group_weights * negative_mask.float()
    ranking = F.relu(group_margins + positive_dist.unsqueeze(1) - centroid_dist)
    ranking = (ranking * active_weights).sum(dim=1) / active_weights.sum(dim=1).clamp_min(1e-8)

    sample_to_centroid = poincare_distance(z, centroids).gather(1, inverse.unsqueeze(1)).squeeze(1)
    compactness = z.new_zeros(num_groups)
    compactness.index_add_(0, inverse, sample_to_centroid.pow(2))
    compactness = compactness / counts

    group_loss = (
        float(pull_weight) * positive_dist.pow(2)
        + float(margin_weight) * ranking
        + float(compactness_weight) * compactness
    )
    return group_loss.mean()


def hyperbolic_hierarchy_margin_loss(
    outputs,
    labels,
    graph,
    base_margin=0.2,
    max_margin=0.6,
    neutral_boost=0.2,
    branch_boost=0.1,
    same_branch_scale=0.5,
    prototype_weight=0.5,
    pair_weight_base=1.0,
    neutral_pair_weight=None,
    neutral_positive_pair_weight=None,
    neutral_negative_pair_weight=None,
    cross_branch_pair_weight=None,
    positive_negative_pair_weight=None,
    same_branch_pair_weight=None,
    same_negative_pair_weight=None,
    pair_reduction='sample',
):
    z = outputs['proto_embedding']
    prototypes = _class_prototypes(outputs['prototypes'], graph, z.device)
    dist = poincare_distance(z, prototypes)
    margin_matrix = _hierarchy_margin_matrix(
        graph,
        z.device,
        base_margin=base_margin,
        max_margin=max_margin,
        neutral_boost=neutral_boost,
        branch_boost=branch_boost,
        same_branch_scale=same_branch_scale,
    )
    pair_weights = _hierarchy_pair_weight_matrix(
        graph,
        z.device,
        base_weight=pair_weight_base,
        neutral_pair_weight=neutral_pair_weight,
        neutral_positive_pair_weight=neutral_positive_pair_weight,
        neutral_negative_pair_weight=neutral_negative_pair_weight,
        cross_branch_pair_weight=cross_branch_pair_weight,
        positive_negative_pair_weight=positive_negative_pair_weight,
        same_branch_pair_weight=same_branch_pair_weight,
        same_negative_pair_weight=same_negative_pair_weight,
    )

    if pair_reduction == 'pair_mean':
        pair_losses = []
        pair_loss_weights = []
        num_classes = graph['num_classes']
        for a in range(num_classes):
            for b in range(a + 1, num_classes):
                pair_weight = pair_weights[a, b]
                if pair_weight.detach().item() <= 0:
                    continue

                margin = margin_matrix[a, b]
                terms = []
                mask_a = labels == a
                if mask_a.any().item():
                    gap_a = dist[mask_a, b] - dist[mask_a, a]
                    terms.append(F.relu(margin - gap_a).mean())

                mask_b = labels == b
                if mask_b.any().item():
                    gap_b = dist[mask_b, a] - dist[mask_b, b]
                    terms.append(F.relu(margin - gap_b).mean())

                proto_gap = poincare_distance(prototypes[a:a + 1], prototypes[b:b + 1]).squeeze()
                terms.append(float(prototype_weight) * F.relu(margin - proto_gap))

                pair_losses.append(pair_weight * torch.stack(terms).mean())
                pair_loss_weights.append(pair_weight)

        if not pair_losses:
            return z.new_tensor(0.0)
        return torch.stack(pair_losses).sum() / torch.stack(pair_loss_weights).sum().clamp_min(1e-8)
    if pair_reduction != 'sample':
        raise ValueError(f'Unknown hierarchy margin pair_reduction: {pair_reduction}')

    pos_dist = dist.gather(1, labels.view(-1, 1))
    sample_margins = margin_matrix.index_select(0, labels)
    sample_weights = pair_weights.index_select(0, labels)
    class_mask = torch.ones_like(sample_margins, dtype=torch.bool)
    class_mask.scatter_(1, labels.view(-1, 1), False)
    sample_loss = F.relu(sample_margins + pos_dist - dist)
    sample_weight_mask = sample_weights * class_mask.float()
    sample_weight_sum = sample_weight_mask.sum()
    if sample_weight_sum.detach().item() > 0:
        sample_loss = (sample_loss * sample_weight_mask).sum() / sample_weight_sum.clamp_min(1e-8)
    else:
        sample_loss = z.new_tensor(0.0)

    proto_dist = poincare_distance(prototypes, prototypes)
    pair_mask = torch.triu(torch.ones_like(proto_dist, dtype=torch.bool), diagonal=1)
    proto_penalty = F.relu(margin_matrix - proto_dist)[pair_mask]
    proto_weights = pair_weights[pair_mask]
    proto_weight_sum = proto_weights.sum()
    if proto_weight_sum.detach().item() > 0:
        proto_loss = (proto_penalty * proto_weights).sum() / proto_weight_sum.clamp_min(1e-8)
    else:
        proto_loss = z.new_tensor(0.0)
    return sample_loss + float(prototype_weight) * proto_loss


def hyperbolic_class_margin_loss(outputs, labels, graph, class_pairs=None, margin=0.4, prototype_weight=0.5):
    z = outputs['proto_embedding']
    prototypes = _class_prototypes(outputs['prototypes'], graph, z.device)
    if class_pairs is None:
        class_pairs = [(0, graph['num_classes'] - 1)]

    dist = poincare_distance(z, prototypes)
    losses = []
    for a, b in class_pairs:
        a = int(a)
        b = int(b)
        if a >= prototypes.shape[0] or b >= prototypes.shape[0]:
            continue

        mask_a = labels == a
        if mask_a.any().item():
            gap_a = dist[mask_a, b] - dist[mask_a, a]
            losses.append(F.relu(float(margin) - gap_a).mean())

        mask_b = labels == b
        if mask_b.any().item():
            gap_b = dist[mask_b, a] - dist[mask_b, b]
            losses.append(F.relu(float(margin) - gap_b).mean())

        proto_gap = poincare_distance(prototypes[a:a + 1], prototypes[b:b + 1]).squeeze()
        losses.append(float(prototype_weight) * F.relu(float(margin) - proto_gap))

    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()
