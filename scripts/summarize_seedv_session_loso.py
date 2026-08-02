import argparse
import json
from pathlib import Path

import numpy as np


NUM_CLASSES = 5
CLASS_NAMES = ['neutral', 'happy', 'sad', 'fear', 'disgust']
METRIC_KEYS = [
    'acc',
    'balanced_acc',
    'macro_f1',
    'weighted_f1',
    'kappa',
    'loss',
    'best_val_acc',
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize three session-wise SEED-V strict LOSO runs'
    )
    parser.add_argument(
        '--root',
        default='results/seed_v/strict_dg_loso/v8_full100',
    )
    args = parser.parse_args()
    root = Path(args.root)

    all_folds = []
    session_results = {}
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for session in (1, 2, 3):
        session_dir = root / f'sessions_{session}'
        summary = load_json(session_dir / 'summary.json')
        folds = [
            load_json(path)
            for path in sorted(
                session_dir.glob('target_subject_*/test_result.json')
            )
        ]
        if len(folds) != 16:
            raise ValueError(
                f'Session {session}: expected 16 folds, found {len(folds)}'
            )
        for fold in folds:
            fold_confusion = np.asarray(
                fold['confusion'],
                dtype=np.int64,
            )
            if fold_confusion.shape != (NUM_CLASSES, NUM_CLASSES):
                raise ValueError(
                    f'Session {session}: invalid confusion shape '
                    f'{fold_confusion.shape}'
                )
            confusion += fold_confusion
        all_folds.extend(folds)
        session_results[str(session)] = {
            key: summary[key]
            for key in METRIC_KEYS
        }

    metrics = {}
    for key in METRIC_KEYS:
        values = np.asarray([fold[key] for fold in all_folds], dtype=float)
        metrics[key] = {
            'mean': float(values.mean()),
            'std_across_48_folds': float(values.std()),
            'mean_of_session_means': float(
                np.mean(
                    [
                        session_results[str(session)][key]['mean']
                        for session in (1, 2, 3)
                    ]
                )
            ),
        }

    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        true_count = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        per_class[name] = {
            'support': true_count,
            'predicted': predicted_count,
            'precision': (
                float(true_positive / predicted_count)
                if predicted_count
                else 0.0
            ),
            'recall': (
                float(true_positive / true_count)
                if true_count
                else 0.0
            ),
        }

    result = {
        'protocol': 'session-wise strict_dg_loso',
        'sessions': session_results,
        'overall': {
            'num_sessions': 3,
            'num_folds': len(all_folds),
            'chance_accuracy': 0.2,
            'metrics': metrics,
            'pooled_sample_accuracy': float(
                np.trace(confusion) / confusion.sum()
            ),
            'best_fold_accuracy': float(
                max(fold['acc'] for fold in all_folds)
            ),
            'worst_fold_accuracy': float(
                min(fold['acc'] for fold in all_folds)
            ),
        },
        'classes': CLASS_NAMES,
        'per_class': per_class,
        'confusion': confusion.tolist(),
    }
    output = root / 'session_average_summary.json'
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(output)


if __name__ == '__main__':
    main()
