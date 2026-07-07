import torch
from tqdm import tqdm
from losses.eagle_loss import compute_eagle_loss


def train_one_epoch(model, loader, optimizer, graph, cfg, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    loss_sums = {}
    for batch in tqdm(loader, desc='train', leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss, loss_dict = compute_eagle_loss(outputs, y, graph, cfg)
        loss.backward()
        if cfg['train'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
        optimizer.step()
        bs = y.size(0)
        total += bs
        total_loss += float(loss.detach()) * bs
        pred = outputs['logits_final'].argmax(dim=-1)
        correct += int((pred == y).sum())
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach()) * bs
    metrics = {'loss': total_loss / max(total,1), 'acc': correct / max(total,1)}
    for k, v in loss_sums.items():
        metrics[f'loss_{k}'] = v / max(total,1)
    return metrics
