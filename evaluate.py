import argparse
from pathlib import Path
import json
import torch
from utils.config import load_config
from data.dataloader import load_feature_store, build_loader
from graphs.affective_graph import get_affective_graph
from models import build_model
from trainer.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate one EAGLE-Net checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subjects', nargs='+', type=int, required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get('train', {}).get('num_threads'):
        torch.set_num_threads(int(cfg['train']['num_threads']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = load_feature_store(cfg)
    graph = get_affective_graph(cfg['dataset']['name'])
    model = build_model(cfg, graph).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    loader = build_loader(store, args.subjects, cfg, shuffle=False, batch_size=cfg['train']['batch_size'])
    metrics, _ = evaluate(model, loader, graph, cfg, device=device, return_outputs=False)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
