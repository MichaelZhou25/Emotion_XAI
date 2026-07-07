import numpy as np


def summarize_edge_attention(edge_attention):
    # edge_attention: [N,E,T,C,B]
    return {
        'edge_time_importance': edge_attention.sum(axis=(3,4)),       # [N,E,T]
        'edge_channel_importance': edge_attention.sum(axis=(2,4)),    # [N,E,C]
        'edge_band_importance': edge_attention.sum(axis=(2,3)),       # [N,E,B]
        'edge_mean_map': edge_attention.mean(axis=0),                 # [E,T,C,B]
    }
