SEED_GRAPH = {
    'name': 'seed_valence_graph',
    'dataset': 'SEED',
    'nodes': ['neutral', 'positive_valence', 'negative_valence'],
    'classes': ['neutral', 'positive', 'negative'],
    'edges': [
        ('neutral', 'positive_valence'),
        ('neutral', 'negative_valence'),
    ],
    'path_matrix': [
        [0, 0],
        [1, 0],
        [0, 1],
    ],
    'edge_targets': {
        0: [0, 0],
        1: [1, 0],
        2: [0, 1],
    },
    'concept_matrix': [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    'concept_targets': {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [-1.0, 0.0, 0.0],
    },
    'concept_mask': [1.0, 0.0, 0.0],
    'semantic_coords': [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    'node_depth': [0, 1, 1],
}
