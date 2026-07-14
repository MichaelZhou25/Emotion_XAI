import argparse
import json
from pathlib import Path

import numpy as np


def load_fold_results(input_dirs):
    folds = {}
    sources = {}
    for input_dir in map(Path, input_dirs):
        for result_path in sorted(input_dir.glob('target_subject_*/test_result.json')):
            target = int(result_path.parent.name.rsplit('_', 1)[-1])
            with result_path.open('r', encoding='utf-8') as handle:
                metrics = json.load(handle)
            if target in folds and folds[target] != metrics:
                raise ValueError(f'Conflicting result for target subject {target}: {sources[target]} vs {result_path}')
            folds[target] = metrics
            sources[target] = str(result_path)
    return folds, sources


def summarize(folds, expected_subjects):
    missing = sorted(set(expected_subjects) - set(folds))
    unexpected = sorted(set(folds) - set(expected_subjects))
    if missing or unexpected:
        raise ValueError(f'Fold mismatch: missing={missing}, unexpected={unexpected}')

    ordered = [folds[target] for target in expected_subjects]
    numeric_keys = [key for key, value in ordered[0].items() if isinstance(value, (int, float))]
    return {
        key: {
            'mean': float(np.mean([metrics[key] for metrics in ordered])),
            'std': float(np.std([metrics[key] for metrics in ordered])),
        }
        for key in numeric_keys
    }


def main():
    parser = argparse.ArgumentParser(description='Aggregate independently seeded LOSO fold results.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--subjects', nargs='+', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    folds, sources = load_fold_results(args.inputs)
    summary = summarize(folds, args.subjects)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    manifest = {
        'subjects': args.subjects,
        'inputs': args.inputs,
        'fold_sources': {str(target): sources[target] for target in args.subjects},
    }
    with (output_dir / 'fold_sources.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
