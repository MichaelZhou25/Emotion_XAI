import argparse
import os
from pathlib import Path
import sys

if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from data.dataloader import build_fold_loaders, load_feature_store
from data.splits import legacy_loso_split
from graphs.affective_graph import get_affective_graph
from models import build_model
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate fixed edge/prototype energy mixtures from saved LOSO checkpoints.',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--alphas', default='0,0.05,0.1,0.2,0.5,1.0')
    return parser.parse_args()


@torch.no_grad()
def collect_logits(model, loader, device):
    edge_logits = []
    prototype_logits = []
    labels = []
    model.eval()
    for batch in loader:
        outputs = model(batch['x'].to(device))
        edge_logits.append(outputs['edge_code_logits'].cpu())
        prototype_logits.append(outputs['logits_proto'].cpu())
        labels.append(batch['y'])
    return torch.cat(edge_logits), torch.cat(prototype_logits), torch.cat(labels)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(int(cfg.get('train', {}).get('num_threads', 1)))
    store = load_feature_store(cfg)
    graph = get_affective_graph(cfg['dataset']['name'], cfg['graph']['name'])
    device = torch.device(
        'cuda'
        if torch.cuda.is_available() and cfg.get('train', {}).get('device', 'auto') != 'cpu'
        else 'cpu'
    )
    alphas = [float(value) for value in args.alphas.split(',')]
    fold_scores = {alpha: [] for alpha in alphas}
    scale_ratios = []

    for target in store.subjects:
        train_subjects, _, test_subjects = legacy_loso_split(store.subjects, target)
        _, _, test_loader = build_fold_loaders(
            store,
            train_subjects,
            [],
            test_subjects,
            cfg,
        )
        model = build_model(cfg, graph).to(device)
        checkpoint_path = (
            Path(args.results_dir)
            / f'target_subject_{target:02d}'
            / 'best_model.pth'
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        edge, prototype, labels = collect_logits(model, test_loader, device)
        edge_scale = edge.std(dim=1).mean().item()
        prototype_scale = prototype.std(dim=1).mean().item()
        scale_ratios.append(prototype_scale / max(edge_scale, 1e-8))

        values = []
        for alpha in alphas:
            accuracy = ((edge + alpha * prototype).argmax(dim=1) == labels).float().mean().item()
            fold_scores[alpha].append(accuracy)
            values.append(f'a={alpha:g}:{accuracy:.4f}')
        print(f'target={target:02d} ' + ' '.join(values))

    print(f'prototype_to_edge_logit_scale={np.mean(scale_ratios):.4f}')
    for alpha in alphas:
        scores = np.asarray(fold_scores[alpha], dtype=np.float64)
        print(f'alpha={alpha:g} mean={scores.mean():.6f} std={scores.std():.6f}')


if __name__ == '__main__':
    main()
