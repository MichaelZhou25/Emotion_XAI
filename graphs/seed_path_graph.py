SEED_PATH_GRAPH = {
    'name': 'seed_hyperbolic_path_graph',
    'dataset': 'SEED',
    'root_node': 'emotion_root',
    'nodes': ['emotion_root', 'neutral', 'positive', 'negative'],
    'classes': ['neutral', 'positive', 'negative'],
    'class_node_indices': [1, 2, 3],
    'edges': [
        ('emotion_root', 'neutral'),
        ('emotion_root', 'positive'),
        ('emotion_root', 'negative'),
    ],
    'path_matrix': [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    'edge_targets': {
        0: [1, 0, 0],
        1: [0, 1, 0],
        2: [0, 0, 1],
    },
    'concept_matrix': [
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0],
    ],
    'concept_targets': {
        0: [0.0, -1.0, 0.0],
        1: [1.0, 0.5, 0.0],
        2: [-1.0, 0.5, 0.0],
    },
    'concept_mask': [1.0, 1.0, 0.0],
    'semantic_coords': [
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0],
    ],
    'node_depth': [0, 1, 1, 1],
}
