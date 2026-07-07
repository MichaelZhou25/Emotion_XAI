import argparse
from pathlib import Path
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Aggregate EAGLE-Net fold results')
    parser.add_argument('--result_dir', required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    files = sorted(result_dir.glob('target_subject_*/test_result.json'))
    if not files:
        raise FileNotFoundError(f'No test_result.json found under {result_dir}')

    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        rows.append(d)
    keys = sorted([k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))])
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std()), 'values': vals.tolist()}
    print(json.dumps(summary, indent=2))
    (result_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
