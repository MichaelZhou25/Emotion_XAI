import torch
import torch.nn.functional as F


def radius_hierarchy_loss(prototypes, graph, margin=0.05):
    node_to_idx = {n: i for i, n in enumerate(graph['nodes'])}
    norms = prototypes.norm(dim=-1)
    loss = prototypes.new_tensor(0.0)
    count = 0
    for parent, child in graph['edges']:
        pi = node_to_idx[parent]
        ci = node_to_idx[child]
        loss = loss + F.relu(margin - (norms[ci] - norms[pi]))
        count += 1
    return loss / max(count, 1)
