import numpy as np
import torch


@torch.no_grad()
def extract_model_outputs(model, loader, graph, cfg, device):
    model.eval()
    keys = ['edge_attention','edge_weights','proto_distance','concept_scores','logits_final']
    collected = {k: [] for k in keys}
    y_true, y_pred = [], []
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        out = model(x)
        pred = out['logits_final'].argmax(dim=-1)
        for k in keys:
            collected[k].append(out[k].detach().cpu().numpy())
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    arrays = {k: np.concatenate(v, axis=0) for k,v in collected.items()}
    arrays['y_true'] = np.concatenate(y_true)
    arrays['y_pred'] = np.concatenate(y_pred)
    return arrays
