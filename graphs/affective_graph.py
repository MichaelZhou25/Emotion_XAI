from copy import deepcopy
import torch
from graphs.seed_graph import SEED_GRAPH
from graphs.seediv_graph import SEEDIV_GRAPH


def get_affective_graph(dataset_name):
    name = dataset_name.upper().replace('_', '-')
    if name == 'SEED':
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
