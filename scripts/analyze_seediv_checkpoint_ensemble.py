import argparse
import itertools
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


SESSION_CANDIDATES = {
    1: {
        'v8': (
            'configs/seediv_hyperbolic_node_edge_v8_full100.yaml',
            'results/seed_iv/hyperbolic_node_edge/v8_full100',
        ),
        'relation': (
            'configs/tuning/seediv_edge_relation_balance_s1_full100.yaml',
            'results/tuning/seediv_edge_full15/relation_balance/s1',
        ),
        'centroid': (
            'configs/tuning/seediv_hpcl_subject_centroid_s1_phase40.yaml',
            'results/tuning/seediv_hpcl_centroid_phase40/s1',
        ),
        'endpoint025': (
            'configs/tuning/seediv_v8_phase3_endpoint025_s1_full100.yaml',
            'results/tuning/v8_full15/seediv/endpoint025_s1',
        ),
        'endpoint_gate': (
            'configs/tuning/seediv_endpoint_gate_s1_full100.yaml',
            'results/tuning/seediv_gate_full15/gate_only/s1',
        ),
        'target_uda': (
            'configs/seediv_target_domain_uda_session1_phase40.yaml',
            (
                'results/tuning/seediv_target_domain_uda_ramp10_s1_probe',
                'results/tuning/seediv_target_domain_uda_ramp10_s1_remaining9',
            ),
        ),
    },
    2: {
        'v8': (
            'configs/seediv_hyperbolic_node_edge_v8_session2_full100.yaml',
            'results/seed_iv/hyperbolic_node_edge/v8_session2_full100',
        ),
        'relation': (
            'configs/tuning/seediv_edge_relation_balance_s2_full100.yaml',
            'results/tuning/seediv_edge_full15/relation_balance/s2',
        ),
        'centroid': (
            'configs/tuning/seediv_hpcl_subject_centroid_s2_phase40.yaml',
            'results/tuning/seediv_hpcl_centroid_phase40/s2',
        ),
        'target_uda': (
            'configs/seediv_target_domain_uda_session2_phase40.yaml',
            (
                'results/tuning/seediv_target_domain_uda_s2_probe',
                'results/tuning/seediv_target_domain_uda_s2_remaining9',
            ),
        ),
    },
    3: {
        'v8': (
            'configs/seediv_hyperbolic_node_edge_v8_session3_full100.yaml',
            'results/seed_iv/hyperbolic_node_edge/v8_session3_full100',
        ),
        'relation': (
            'configs/tuning/seediv_edge_relation_balance_s3_full100.yaml',
            'results/tuning/seediv_edge_full15/relation_balance/s3',
        ),
        'centroid': (
            'configs/tuning/seediv_hpcl_subject_centroid_s3_phase40.yaml',
            'results/tuning/seediv_hpcl_centroid_phase40/s3',
        ),
        'target_uda': (
            'configs/seediv_target_domain_uda_session3_phase40.yaml',
            (
                'results/tuning/seediv_target_domain_uda_s3_probe',
                'results/tuning/seediv_target_domain_uda_s3_remaining9',
            ),
        ),
    },
}


def predict(model, loader, device):
    probabilities = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch['x'].to(device))['logits_final']
            probabilities.append(F.softmax(logits, dim=-1).cpu().numpy())
            labels.append(batch['y'].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def summarize(probabilities, labels_by_subject, names):
    subject_metrics = []
    for subject in sorted(labels_by_subject):
        probability = np.mean(
            [probabilities[name][subject] for name in names],
            axis=0,
        )
        prediction = probability.argmax(axis=1)
        subject_metrics.append(
            classification_metrics(labels_by_subject[subject], prediction)
        )
    return {
        key: float(np.mean([metrics[key] for metrics in subject_metrics]))
        for key in subject_metrics[0]
    } | {
        'acc_std': float(np.std([metrics['acc'] for metrics in subject_metrics])),
        'macro_f1_std': float(
            np.std([metrics['macro_f1'] for metrics in subject_metrics])
        ),
        'per_subject_acc': [
            float(metrics['acc']) for metrics in subject_metrics
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        '--output',
        default=None,
    )
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    candidates = SESSION_CANDIDATES[args.session]

    device = torch.device(
        'cuda'
        if torch.cuda.is_available() and args.device != 'cpu'
        else 'cpu'
    )
    configs = {
        name: load_config(config_path)
        for name, (config_path, _) in candidates.items()
    }
    base_cfg = configs['v8']
    if base_cfg.get('train', {}).get('num_threads'):
        torch.set_num_threads(int(base_cfg['train']['num_threads']))
    store = load_feature_store(base_cfg)
    subjects = store.subjects

    probabilities = {name: {} for name in candidates}
    labels_by_subject = {}
    for target in subjects:
        train_subjects, _, test_subjects = legacy_loso_split(subjects, target)
        _, _, test_loader = build_fold_loaders(
            store,
            train_subjects,
            [],
            test_subjects,
            base_cfg,
        )
        reference_labels = None
        for name, (_, result_dirs) in candidates.items():
            cfg = configs[name]
            graph = get_affective_graph(
                cfg['dataset']['name'],
                cfg.get('graph', {}).get('name'),
            )
            model = build_model(cfg, graph).to(device)
            if isinstance(result_dirs, str):
                result_dirs = (result_dirs,)
            checkpoints = [
                Path(result_dir)
                / f'target_subject_{target:02d}'
                / 'best_model.pth'
                for result_dir in result_dirs
            ]
            checkpoint = next(
                (path for path in checkpoints if path.exists()),
                None,
            )
            if checkpoint is None:
                raise FileNotFoundError(
                    ', '.join(str(path) for path in checkpoints)
                )
            load_checkpoint(checkpoint, model, device)
            probability, labels = predict(model, test_loader, device)
            if reference_labels is not None and not np.array_equal(
                labels,
                reference_labels,
            ):
                raise RuntimeError(
                    f'Label order mismatch for {name}, subject {target}.'
                )
            probabilities[name][target] = probability
            reference_labels = labels
            del model
        labels_by_subject[target] = reference_labels
        print(f'predicted target_subject_{target:02d}', flush=True)

    names = tuple(candidates)
    subsets = []
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            metrics = summarize(
                probabilities,
                labels_by_subject,
                subset,
            )
            subsets.append({'models': list(subset), **metrics})
    subsets.sort(key=lambda row: (row['acc'], row['macro_f1']), reverse=True)

    leave_one_subject_out = []
    for held_out_index, target in enumerate(subjects):
        eligible = []
        for row in subsets:
            train_acc = float(np.mean([
                acc
                for index, acc in enumerate(row['per_subject_acc'])
                if index != held_out_index
            ]))
            eligible.append((train_acc, row['macro_f1'], row))
        selected = max(eligible, key=lambda item: (item[0], item[1]))[2]
        held_out_acc = selected['per_subject_acc'][held_out_index]
        leave_one_subject_out.append({
            'target_subject': int(target),
            'selected_models': selected['models'],
            'acc': held_out_acc,
        })

    oracle_subject_acc = []
    oracle_sample_acc = []
    for subject_index, target in enumerate(subjects):
        oracle_subject_acc.append(max(
            row['per_subject_acc'][subject_index]
            for row in subsets
            if len(row['models']) == 1
        ))
        correct = np.stack([
            probabilities[name][target].argmax(axis=1)
            == labels_by_subject[target]
            for name in names
        ])
        oracle_sample_acc.append(float(correct.any(axis=0).mean()))

    output = {
        'note': (
            'Diagnostic only: every checkpoint was selected by target-test '
            'accuracy under legacy LOSO. Selecting an ensemble from these '
            'same target results is additionally optimistic.'
        ),
        'device': str(device),
        'subjects': subjects,
        'subsets_ranked': subsets,
        'leave_one_subject_out_selection': {
            'mean_acc': float(np.mean([
                row['acc'] for row in leave_one_subject_out
            ])),
            'folds': leave_one_subject_out,
        },
        'oracle_subject_selection_mean_acc': float(
            np.mean(oracle_subject_acc)
        ),
        'oracle_sample_mean_acc': float(np.mean(oracle_sample_acc)),
    }
    output_path = Path(
        args.output
        or f'results/diagnostics/seediv_s{args.session}_checkpoint_ensemble.json'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'top_subsets': [
            {
                key: row[key]
                for key in ('models', 'acc', 'macro_f1', 'acc_std')
            }
            for row in subsets[:10]
        ],
        'leave_one_subject_out_mean_acc':
            output['leave_one_subject_out_selection']['mean_acc'],
        'oracle_subject_selection_mean_acc':
            output['oracle_subject_selection_mean_acc'],
        'oracle_sample_mean_acc': output['oracle_sample_mean_acc'],
        'output': str(output_path),
    }, indent=2))


if __name__ == '__main__':
    main()
