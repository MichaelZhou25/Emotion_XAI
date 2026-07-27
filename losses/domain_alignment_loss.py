import torch


def linear_mmd_loss(source_features, target_features):
    """Squared distance between source and target feature means."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError('linear_mmd_loss expects [batch, feature] tensors')
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError('source and target feature dimensions must match')
    mean_delta = source_features.mean(dim=0) - target_features.mean(dim=0)
    return mean_delta.square().mean()
