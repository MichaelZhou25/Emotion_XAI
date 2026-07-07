import torch
import torch.nn.functional as F


@torch.no_grad()
def edge_attention_stability(model, loader, device, noise_std=0.02):
    model.eval()
    sims = []
    for batch in loader:
        x = batch['x'].to(device)
        a1 = model(x)['edge_attention'].reshape(x.shape[0], -1)
        a2 = model(x + noise_std * torch.randn_like(x))['edge_attention'].reshape(x.shape[0], -1)
        sim = F.cosine_similarity(a1, a2, dim=-1)
        sims.extend(sim.cpu().numpy().tolist())
    return {'mean_cosine': float(sum(sims)/max(len(sims),1)), 'values': sims}
