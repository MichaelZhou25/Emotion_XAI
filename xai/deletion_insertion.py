import torch
import torch.nn.functional as F


def _edge_importance_for_input(edge_attention, x):
    imp = edge_attention.amax(dim=1)
    if imp.shape[1] == 1 and x.shape[1] > 1:
        imp = imp.expand(-1, x.shape[1], -1, -1)
    if imp.shape[1:] != x.shape[1:]:
        raise ValueError(f'Edge attention shape {tuple(imp.shape[1:])} does not match input shape {tuple(x.shape[1:])}')
    return imp.reshape(x.shape[0], -1)


@torch.no_grad()
def deletion_insertion_curve(model, loader, graph, cfg, device, fractions=(0.0,0.1,0.2,0.3,0.5)):
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        out = model(x)
        p0 = F.softmax(out['logits_final'], dim=-1).gather(1, y.view(-1,1)).squeeze(1)
        # Use max edge attention across edges as token importance.
        imp = _edge_importance_for_input(out['edge_attention'], x)
        flat_x = x.reshape(x.shape[0], -1)
        order = torch.argsort(imp, dim=1, descending=True)
        for frac in fractions:
            k = int(flat_x.shape[1] * frac)
            xd = flat_x.clone()
            if k > 0:
                idx = order[:, :k]
                xd.scatter_(1, idx, 0.0)
            xd = xd.reshape_as(x)
            pd = F.softmax(model(xd)['logits_final'], dim=-1).gather(1, y.view(-1,1)).squeeze(1)
            rows.append({'batch': batch_idx, 'fraction': float(frac),
                         'prob_original': float(p0.mean().cpu()), 'prob_deleted': float(pd.mean().cpu())})
    return rows
