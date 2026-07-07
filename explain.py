import argparse
from pathlib import Path
import json
import torch
from utils.config import load_config
from data.dataloader import load_feature_store, build_loader
from graphs.affective_graph import get_affective_graph
from models import build_model
from xai.extract_outputs import extract_model_outputs
from xai.edge_maps import summarize_edge_attention
from xai.deletion_insertion import deletion_insertion_curve


def main():
    parser = argparse.ArgumentParser(description='Export EAGLE-Net XAI outputs for a checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subjects', nargs='+', type=int, required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--run_deletion', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get('train', {}).get('num_threads'):
        torch.set_num_threads(int(cfg['train']['num_threads']))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = load_feature_store(cfg)
    graph = get_affective_graph(cfg['dataset']['name'])
    model = build_model(cfg, graph).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    loader = build_loader(store, args.subjects, cfg, shuffle=False, batch_size=cfg['train']['batch_size'])

    arrays = extract_model_outputs(model, loader, graph, cfg, device=device)
    for name, arr in arrays.items():
        import numpy as np
        np.save(out_dir / f'{name}.npy', arr)

    summary = summarize_edge_attention(arrays['edge_attention'])
    for name, arr in summary.items():
        import numpy as np
        np.save(out_dir / f'{name}.npy', arr)

    if args.run_deletion:
        curve = deletion_insertion_curve(model, loader, graph, cfg, device=device)
        (out_dir / 'deletion_insertion.json').write_text(json.dumps(curve, indent=2), encoding='utf-8')

    print(f'XAI outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
