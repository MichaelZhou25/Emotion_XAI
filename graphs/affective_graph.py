from copy import deepcopy
import torch
from graphs.seed_graph import SEED_GRAPH
from graphs.seediv_graph import SEEDIV_GRAPH
from graphs.seed_path_graph import SEED_PATH_GRAPH
from graphs.seediv_path_graph import SEEDIV_PATH_GRAPH
from graphs.neutral_centered_graph import (
    SEED_NEUTRAL_CENTERED_GRAPH,
    SEEDIV_NEUTRAL_CENTERED_GRAPH,
)


def get_affective_graph(dataset_name, graph_name=None):
    name = dataset_name.upper().replace('_', '-')
    requested = (graph_name or '').lower().replace('-', '_')
    if requested in {'seed_neutral_centered_path_graph', 'seed_neutral_centered_graph'}:
        g = deepcopy(SEED_NEUTRAL_CENTERED_GRAPH)
    elif requested in {'seediv_neutral_centered_path_graph', 'seediv_neutral_centered_graph'}:
        g = deepcopy(SEEDIV_NEUTRAL_CENTERED_GRAPH)
    elif requested in {'seed_hyperbolic_path_graph', 'seed_path_graph'}:
        g = deepcopy(SEED_PATH_GRAPH)
    elif requested in {'seediv_hyperbolic_path_graph', 'seediv_path_graph'}:
        g = deepcopy(SEEDIV_PATH_GRAPH)
    elif name == 'SEED':
        g = deepcopy(SEED_GRAPH)
    elif name in ['SEED-IV', 'SEEDIV']:
        g = deepcopy(SEEDIV_GRAPH)
    else:
        raise ValueError(f'Unsupported dataset graph: {dataset_name}')
    g['num_classes'] = len(g['classes'])
    g['num_nodes'] = len(g['nodes'])
    g['num_edges'] = len(g['edges'])
    g['path_matrix_tensor'] = torch.tensor(g['path_matrix'], dtype=torch.float32)
    g['concept_matrix_tensor'] = torch.tensor(g['concept_matrix'], dtype=torch.float32)
    g['concept_mask_tensor'] = torch.tensor(g['concept_mask'], dtype=torch.float32)
    if 'class_node_indices' not in g:
        g['class_node_indices'] = list(range(g['num_classes']))
    return g
