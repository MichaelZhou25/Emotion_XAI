import torch
from trainer.train_one_epoch import train_one_epoch
from trainer.evaluate import evaluate


def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train'].get('weight_decay', 0.0))


def train_and_validate_one_epoch(model, train_loader, val_loader, optimizer, graph, cfg, device):
    train_metrics = train_one_epoch(model, train_loader, optimizer, graph, cfg, device)
    val_metrics, _ = evaluate(model, val_loader, graph, cfg, device) if val_loader is not None else ({}, None)
    return train_metrics, val_metrics
