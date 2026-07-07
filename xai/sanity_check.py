import torch


def randomize_model_parameters(model):
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    return model


@torch.no_grad()
def sanity_randomized_attention(model, loader, device):
    randomize_model_parameters(model)
    batch = next(iter(loader))
    out = model(batch['x'].to(device))
    return out['edge_attention'].detach().cpu().numpy()
