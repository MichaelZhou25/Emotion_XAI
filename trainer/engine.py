import math

import torch
from trainer.train_one_epoch import train_one_epoch
from trainer.evaluate import evaluate
from utils.ema import ModelEMA


def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train'].get('weight_decay', 0.0))


def build_lr_scheduler(optimizer, cfg):
    train_cfg = cfg.get('train', {})
    name = train_cfg.get('lr_scheduler', 'none')
    if name in (None, 'none', 'constant'):
        return None
    if name != 'cosine':
        raise ValueError(f'Unknown lr_scheduler: {name}')

    total_epochs = int(train_cfg['epochs'])
    warmup_epochs = int(train_cfg.get('lr_warmup_epochs', 0))
    base_lr = float(train_cfg['lr'])
    min_lr = float(train_cfg.get('min_lr', 0.0))
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_scale(epoch_index):
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return (epoch_index + 1) / float(warmup_epochs)
        decay_steps = max(1, total_epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)


def build_ema(model, cfg):
    ema_cfg = cfg.get('train', {}).get('ema', {})
    if not ema_cfg.get('enabled', False):
        return None
    return ModelEMA(
        model,
        decay=ema_cfg.get('decay', 0.999),
        warmup_updates=ema_cfg.get('warmup_updates', 100),
    )


def train_and_validate_one_epoch(model, train_loader, val_loader, optimizer, graph, cfg, device):
    train_metrics = train_one_epoch(model, train_loader, optimizer, graph, cfg, device)
    val_metrics, _ = evaluate(model, val_loader, graph, cfg, device) if val_loader is not None else ({}, None)
    return train_metrics, val_metrics
