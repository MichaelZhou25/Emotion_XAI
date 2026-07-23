from copy import deepcopy


SEED_NEUTRAL_CENTERED_GRAPH = {
    'name': 'seed_neutral_centered_path_graph',
    'dataset': 'SEED',
    'path_scoring': 'energy',
    'root_node': 'neutral',
    'nodes': ['neutral', 'positive_valence', 'negative_valence'],
    'classes': ['neutral', 'positive', 'negative'],
    'class_node_indices': [0, 1, 2],
    'fixed_node_indices': [0],
    'stop_class_index': 0,
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
    'root_group_names': ['neutral', 'positive_valence', 'negative_valence'],
    'root_class_groups': [[0], [1], [2]],
    'root_target_by_class': [0, 1, 2],
    'child_class_indices': [],
    'concept_matrix': [
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
        [1.0, 0.25, 0.0],
        [-1.0, 0.25, 0.0],
    ],
    'node_depth': [0, 1, 1],
}


SEEDIV_NEUTRAL_CENTERED_GRAPH = {
    'name': 'seediv_neutral_centered_path_graph',
    'dataset': 'SEED-IV',
    'path_scoring': 'energy',
    'root_node': 'neutral',
    'nodes': [
        'neutral',
        'positive_valence',
        'negative_valence',
        'happy',
        'sad',
        'fear',
    ],
    'classes': ['neutral', 'happy', 'sad', 'fear'],
    'class_node_indices': [0, 3, 4, 5],
    'fixed_node_indices': [0],
    'stop_class_index': 0,
    'edges': [
        ('neutral', 'positive_valence'),
        ('neutral', 'negative_valence'),
        ('positive_valence', 'happy'),
        ('negative_valence', 'sad'),
        ('negative_valence', 'fear'),
    ],
    'path_matrix': [
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ],
    'edge_targets': {
        0: [0, 0, 0, 0, 0],
        1: [1, 0, 1, 0, 0],
        2: [0, 1, 0, 1, 0],
        3: [0, 1, 0, 0, 1],
    },
    'root_group_names': ['neutral', 'positive_valence', 'negative_valence'],
    'root_class_groups': [[0], [1], [2, 3]],
    'root_target_by_class': [0, 1, 2, 2],
    'child_group_name': 'negative_valence',
    'child_class_indices': [2, 3],
    'concept_matrix': [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
        [-1.0, -0.5, -0.3],
        [-1.0, 1.0, -0.7],
    ],
    'concept_targets': {
        0: [0.0, -1.0, 0.0],
        1: [1.0, 1.0, 0.5],
        2: [-1.0, -0.5, -0.3],
        3: [-1.0, 1.0, -0.7],
    },
    'concept_mask': [1.0, 1.0, 1.0],
    'semantic_coords': [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
        [-1.0, -0.5, -0.3],
        [-1.0, 1.0, -0.7],
    ],
    'node_depth': [0, 1, 1, 2, 2, 2],
}


SEED_NEUTRAL_CENTERED_EDGE_GRAPH = deepcopy(SEED_NEUTRAL_CENTERED_GRAPH)
SEED_NEUTRAL_CENTERED_EDGE_GRAPH.update({
    'name': 'seed_neutral_centered_edge_graph',
    'path_scoring': 'edge_code',
})


SEEDIV_NEUTRAL_CENTERED_EDGE_GRAPH = deepcopy(SEEDIV_NEUTRAL_CENTERED_GRAPH)
SEEDIV_NEUTRAL_CENTERED_EDGE_GRAPH.update({
    'name': 'seediv_neutral_centered_edge_graph',
    'path_scoring': 'edge_code',
})
