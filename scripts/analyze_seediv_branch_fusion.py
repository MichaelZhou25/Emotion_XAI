import argparse
import json
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
import torch.nn.functional as F

from data.dataloader import build_fold_loaders, load_feature_store
from data.splits import legacy_loso_split
from graphs.affective_graph import get_affective_graph
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.metrics import classification_metrics


RUNS = {
    (1, 'base'): (
        'configs/seediv_hyperbolic_node_edge_v8_full100.yaml',
        'results/seed_iv/hyperbolic_node_edge/v8_full100',
    ),
    (2, 'base'): (
        'configs/seediv_hyperbolic_node_edge_v8_session2_full100.yaml',
        'results/seed_iv/hyperbolic_node_edge/v8_session2_full100',
    ),
    (3, 'base'): (
        'configs/seediv_hyperbolic_node_edge_v8_session3_full100.yaml',
        'results/seed_iv/hyperbolic_node_edge/v8_session3_full100',
    ),
    (1, 'relation'): (
        'configs/tuning/seediv_edge_relation_balance_s1_full100.yaml',
        'results/tuning/seediv_edge_full15/relation_balance/s1',
    ),
    (2, 'relation'): (
        'configs/tuning/seediv_edge_relation_balance_s2_full100.yaml',
        'results/tuning/seediv_edge_full15/relation_balance/s2',
    ),
    (3, 'relation'): (
        'configs/tuning/seediv_edge_relation_balance_s3_full100.yaml',
        'results/tuning/seediv_edge_full15/relation_balance/s3',
    ),
}


def collect_outputs(model, loader, device):
    edge_logits = []
    prototype_logits = []
    final_logits = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch['x'].to(device))
            edge_logits.append(outputs['edge_code_logits'].cpu().numpy())
            prototype_logits.append(
                outputs['edge_endpoint_logits'].cpu().numpy()
            )
            final_logits.append(outputs['logits_final'].cpu().numpy())
            labels.append(batch['y'].numpy())
    return (
        np.concatenate(edge_logits),
        np.concatenate(prototype_logits),
        np.concatenate(final_logits),
        np.concatenate(labels),
    )


def normalized(logits):
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / np.maximum(
        centered.std(axis=1, keepdims=True),
        1e-6,
    )


def softmax(logits):
    values = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def candidate_predictions(edge, prototype):
    raw_weights = np.linspace(0.0, 2.0, 41)
    probability_weights = np.linspace(0.0, 1.0, 21)
    for weight in raw_weights:
        yield (
            f'raw:{weight:.2f}',
            (edge + weight * prototype).argmax(axis=1),
        )
    edge_normalized = normalized(edge)
    prototype_normalized = normalized(prototype)
    for weight in raw_weights:
        yield (
            f'normalized:{weight:.2f}',
            (
                edge_normalized + weight * prototype_normalized
            ).argmax(axis=1),
        )
    edge_probability = softmax(edge)
    prototype_probability = softmax(prototype)
    for weight in probability_weights:
        yield (
            f'probability:{weight:.2f}',
            (
                (1.0 - weight) * edge_probability
                + weight * prototype_probability
            ).argmax(axis=1),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        '--variant',
        choices=('base', 'relation'),
        required=True,
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    config_path, result_dirs = RUNS[(args.session, args.variant)]
    cfg = load_config(config_path)
    if cfg.get('train', {}).get('num_threads'):
        torch.set_num_threads(int(cfg['train']['num_threads']))
    device = torch.device(
        'cuda'
        if torch.cuda.is_available() and args.device != 'cpu'
        else 'cpu'
    )
    graph = get_affective_graph(
        cfg['dataset']['name'],
        cfg.get('graph', {}).get('name'),
    )
    store = load_feature_store(cfg)
    subjects = store.subjects
    outputs_by_subject = {}
    for target in subjects:
        train_subjects, _, test_subjects = legacy_loso_split(subjects, target)
        _, _, test_loader = build_fold_loaders(
            store,
            train_subjects,
            [],
            test_subjects,
            cfg,
        )
        model = build_model(cfg, graph).to(device)
        candidate_dirs = (
            result_dirs if isinstance(result_dirs, tuple) else (result_dirs,)
        )
        checkpoint = next(
            (
                Path(result_dir)
                / f'target_subject_{target:02d}'
                / 'best_model.pth'
                for result_dir in candidate_dirs
                if (
                    Path(result_dir)
                    / f'target_subject_{target:02d}'
                    / 'best_model.pth'
                ).exists()
            ),
            None,
        )
        if checkpoint is None:
            raise FileNotFoundError(
                f'No checkpoint for target subject {target} in '
                f'{candidate_dirs}'
            )
        load_checkpoint(checkpoint, model, device)
        outputs_by_subject[target] = collect_outputs(
            model,
            test_loader,
            device,
        )
        del model

    candidate_subject_metrics = {}
    for subject in subjects:
        edge, prototype, _, labels = outputs_by_subject[subject]
        for name, prediction in candidate_predictions(edge, prototype):
            candidate_subject_metrics.setdefault(name, {})[subject] = (
                classification_metrics(labels, prediction)
            )

    candidates = []
    for name, subject_metrics in candidate_subject_metrics.items():
        candidates.append({
            'fusion': name,
            'acc': float(np.mean([
                subject_metrics[subject]['acc'] for subject in subjects
            ])),
            'macro_f1': float(np.mean([
                subject_metrics[subject]['macro_f1'] for subject in subjects
            ])),
            'per_subject_acc': [
                subject_metrics[subject]['acc'] for subject in subjects
            ],
        })
    candidates.sort(
        key=lambda row: (row['acc'], row['macro_f1']),
        reverse=True,
    )

    selected_folds = []
    for held_out_index, target in enumerate(subjects):
        ranked = []
        for row in candidates:
            source_acc = np.mean([
                value
                for index, value in enumerate(row['per_subject_acc'])
                if index != held_out_index
            ])
            ranked.append((source_acc, row['macro_f1'], row))
        selected = max(ranked, key=lambda item: (item[0], item[1]))[2]
        selected_folds.append({
            'target_subject': target,
            'fusion': selected['fusion'],
            'acc': selected['per_subject_acc'][held_out_index],
        })

    branch_diagnostics = []
    for subject in subjects:
        edge, prototype, final, labels = outputs_by_subject[subject]
        edge_prediction = edge.argmax(axis=1)
        prototype_prediction = prototype.argmax(axis=1)
        final_prediction = final.argmax(axis=1)
        agreement = edge_prediction == prototype_prediction
        branch_diagnostics.append({
            'target_subject': int(subject),
            'edge_acc': float(np.mean(edge_prediction == labels)),
            'prototype_acc': float(np.mean(prototype_prediction == labels)),
            'final_acc': float(np.mean(final_prediction == labels)),
            'agreement_rate': float(np.mean(agreement)),
            'agreement_acc': (
                float(np.mean(edge_prediction[agreement] == labels[agreement]))
                if agreement.any()
                else None
            ),
        })

    output = {
        'note': (
            'Diagnostic on legacy-LOSO checkpoints. Global fusion selection '
            'uses target labels and is optimistic; leave-one-subject-out '
            'selection chooses fusion using the other 14 target subjects.'
        ),
        'session': args.session,
        'variant': args.variant,
        'checkpoint_result_dirs': list(candidate_dirs),
        'branch_diagnostics': {
            'mean_edge_acc': float(np.mean([
                row['edge_acc'] for row in branch_diagnostics
            ])),
            'mean_prototype_acc': float(np.mean([
                row['prototype_acc'] for row in branch_diagnostics
            ])),
            'mean_final_acc': float(np.mean([
                row['final_acc'] for row in branch_diagnostics
            ])),
            'mean_agreement_rate': float(np.mean([
                row['agreement_rate'] for row in branch_diagnostics
            ])),
            'mean_agreement_acc': float(np.mean([
                row['agreement_acc']
                for row in branch_diagnostics
                if row['agreement_acc'] is not None
            ])),
            'folds': branch_diagnostics,
        },
        'candidates_ranked': candidates,
        'leave_one_subject_out_selection': {
            'mean_acc': float(np.mean([
                fold['acc'] for fold in selected_folds
            ])),
            'folds': selected_folds,
        },
    }
    output_path = Path(
        args.output
        or (
            'results/diagnostics/'
            f'seediv_s{args.session}_{args.variant}_branch_fusion.json'
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    current = next(
        row for row in candidates if row['fusion'] == 'raw:0.50'
    )
    print(json.dumps({
        'current_raw_0.50': {
            key: current[key] for key in ('acc', 'macro_f1')
        },
        'top_candidates': [
            {
                key: row[key]
                for key in ('fusion', 'acc', 'macro_f1')
            }
            for row in candidates[:10]
        ],
        'leave_one_subject_out_mean_acc':
            output['leave_one_subject_out_selection']['mean_acc'],
        'branch_diagnostics': {
            key: value
            for key, value in output['branch_diagnostics'].items()
            if key != 'folds'
        },
        'output': str(output_path),
    }, indent=2))


if __name__ == '__main__':
    main()
