import numpy as np


def default_pairs(num_channels):
    return [(i, num_channels - 1 - i) for i in range(num_channels // 2)]


def hemispheric_from_edge_channel(edge_channel_importance, pairs=None):
    # edge_channel_importance: [N,E,C]
    C = edge_channel_importance.shape[-1]
    pairs = pairs or default_pairs(C)
    diffs = []
    for l, r in pairs:
        diffs.append(edge_channel_importance[..., l] - edge_channel_importance[..., r])
    return np.stack(diffs, axis=-1)  # [N,E,P]
