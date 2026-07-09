import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONFIGS = [
    'configs/ablation/seed_hemi_mv_direct_only_probe.yaml',
    'configs/ablation/seed_hemi_mv_direct_proto_probe.yaml',
    'configs/ablation/seed_hemi_mv_direct_proto_no_hemi_probe.yaml',
]


def config_slug(path):
    return Path(path).stem


def write_status(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Run stage-1 HemiMV ablation probes sequentially.')
    parser.add_argument('--configs', nargs='*', default=DEFAULT_CONFIGS)
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir or f'run_logs/stage1_ablation_{stamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / 'status.json'
    env = os.environ.copy()
    if os.name == 'nt':
        env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    status = {
        'started_at': dt.datetime.now().isoformat(timespec='seconds'),
        'configs': [],
        'completed': False,
    }
    write_status(status_path, status)

    for cfg in args.configs:
        slug = config_slug(cfg)
        entry = {
            'config': cfg,
            'started_at': dt.datetime.now().isoformat(timespec='seconds'),
            'stdout': str(log_dir / f'{slug}.out.log'),
            'stderr': str(log_dir / f'{slug}.err.log'),
            'returncode': None,
        }
        status['configs'].append(entry)
        write_status(status_path, status)

        start = time.time()
        with open(entry['stdout'], 'w', encoding='utf-8') as out, open(entry['stderr'], 'w', encoding='utf-8') as err:
            proc = subprocess.run(
                [sys.executable, 'train.py', '--config', cfg],
                stdout=out,
                stderr=err,
                env=env,
                check=False,
            )
        entry['returncode'] = proc.returncode
        entry['finished_at'] = dt.datetime.now().isoformat(timespec='seconds')
        entry['duration_sec'] = round(time.time() - start, 2)
        write_status(status_path, status)
        if proc.returncode != 0:
            status['completed'] = False
            status['failed_config'] = cfg
            status['finished_at'] = dt.datetime.now().isoformat(timespec='seconds')
            write_status(status_path, status)
            return proc.returncode

    status['completed'] = True
    status['finished_at'] = dt.datetime.now().isoformat(timespec='seconds')
    write_status(status_path, status)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
