from copy import deepcopy


SEEDV_GRAPH = {
    'name': 'seedv_affective_hierarchy',
    'dataset': 'SEED-V',
    'nodes': [
        'neutral',
        'positive_valence',
        'negative_valence',
        'happy',
        'sad',
        'fear',
        'disgust',
    ],
    'classes': ['neutral', 'happy', 'sad', 'fear', 'disgust'],
    'class_node_indices': [0, 3, 4, 5, 6],
    'edges': [
        ('neutral', 'positive_valence'),
        ('neutral', 'negative_valence'),
        ('positive_valence', 'happy'),
        ('negative_valence', 'sad'),
        ('negative_valence', 'fear'),
        ('negative_valence', 'disgust'),
    ],
    'path_matrix': [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
    ],
    'edge_targets': {
        0: [0, 0, 0, 0, 0, 0],
        1: [1, 0, 1, 0, 0, 0],
        2: [0, 1, 0, 1, 0, 0],
        3: [0, 1, 0, 0, 1, 0],
        4: [0, 1, 0, 0, 0, 1],
    },
    # Axes follow the project's SEED-IV convention:
    # valence, activation, and control/threat-related affect.
    'concept_matrix': [
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 0.5],
        [-1.0, -0.5, -0.3],
        [-1.0, 1.0, -0.7],
        [-1.0, 0.5, 0.7],
    ],
    'concept_targets': {
        0: [0.0, -1.0, 0.0],
        1: [1.0, 1.0, 0.5],
        2: [-1.0, -0.5, -0.3],
        3: [-1.0, 1.0, -0.7],
        4: [-1.0, 0.5, 0.7],
    },
    'concept_mask': [1.0, 1.0, 1.0],
    'semantic_coords': [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
        [-1.0, -0.5, -0.3],
        [-1.0, 1.0, -0.7],
        [-1.0, 0.5, 0.7],
    ],
    'node_depth': [0, 1, 1, 2, 2, 2, 2],
}


SEEDV_NEUTRAL_CENTERED_GRAPH = deepcopy(SEEDV_GRAPH)
SEEDV_NEUTRAL_CENTERED_GRAPH.update({
    'name': 'seedv_neutral_centered_path_graph',
    'path_scoring': 'energy',
    'root_node': 'neutral',
    'fixed_node_indices': [0],
    'stop_class_index': 0,
    'root_group_names': [
        'neutral',
        'positive_valence',
        'negative_valence',
    ],
    'root_class_groups': [[0], [1], [2, 3, 4]],
    'root_target_by_class': [0, 1, 2, 2, 2],
    'child_group_name': 'negative_valence',
    'child_class_indices': [2, 3, 4],
})


SEEDV_NEUTRAL_CENTERED_EDGE_GRAPH = deepcopy(SEEDV_NEUTRAL_CENTERED_GRAPH)
SEEDV_NEUTRAL_CENTERED_EDGE_GRAPH.update({
    'name': 'seedv_neutral_centered_edge_graph',
    'path_scoring': 'edge_code',
})
