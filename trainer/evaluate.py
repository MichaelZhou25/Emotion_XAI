import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from losses.eagle_loss import compute_eagle_loss
from utils.metrics import classification_metrics, confusion


@torch.no_grad()
def evaluate(model, loader, graph, cfg, device, return_outputs=False):
    model.eval()
    y_true, y_pred = [], []
    probs = []
    loss_total = 0.0
    n = 0
    collected = {k: [] for k in ['edge_attention','edge_weights','proto_distance','concept_scores','logits_final']}
    for batch in tqdm(loader, desc='eval', leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        outputs = model(x)
        loss, _ = compute_eagle_loss(outputs, y, graph, cfg)
        p = F.softmax(outputs['logits_final'], dim=-1)
        pred = p.argmax(dim=-1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs.append(p.cpu().numpy())
        bs = y.size(0)
        loss_total += float(loss.detach()) * bs
        n += bs
        if return_outputs:
            for k in collected:
                collected[k].append(outputs[k].detach().cpu().numpy())
    metrics = classification_metrics(y_true, y_pred)
    metrics['loss'] = loss_total / max(n, 1)
    metrics['confusion'] = confusion(y_true, y_pred)
    if return_outputs:
        arrays = {k: np.concatenate(v, axis=0) for k, v in collected.items() if v}
        arrays['y_true'] = np.asarray(y_true)
        arrays['y_pred'] = np.asarray(y_pred)
        arrays['prob'] = np.concatenate(probs, axis=0)
        return metrics, arrays
    return metrics, None
