import os
import argparse

if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from utils.config import load_config
from utils.seed import set_seed
from data.dataloader import load_feature_store
from protocols.strict_dg_loso import StrictDGLOSO
from protocols.legacy_loso import LegacyLOSO


def main():
    parser = argparse.ArgumentParser(description='Train EAGLE-Net')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    import torch
    if cfg.get('train', {}).get('num_threads'):
        torch.set_num_threads(int(cfg['train']['num_threads']))
    set_seed(cfg['train'].get('seed', 2026))
    store = load_feature_store(cfg)

    protocol = cfg['protocol']['name']
    if protocol == 'strict_dg_loso':
        runner = StrictDGLOSO(cfg, store)
    elif protocol == 'legacy_loso':
        runner = LegacyLOSO(cfg, store)
    else:
        raise ValueError(f'Unknown protocol: {protocol}. Only strict_dg_loso and legacy_loso are supported.')

    runner.run()


if __name__ == '__main__':
    main()
