from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics or {}}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    return ckpt
