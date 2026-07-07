import re
import numpy as np


def ensure_tcb(feature, num_channels=62, num_bands=5):
    """Convert one trial feature array to [time, channels, bands].

    Supports common SEED extracted-feature shapes:
    - [bands, channels, time]
    - [channels, bands, time]
    - [channels, time, bands]
    - [time, channels, bands]
    - [time, channels * bands]
    """
    arr = np.asarray(feature)
    if arr.ndim == 2:
        if arr.shape[-1] == num_channels * num_bands:
            return arr.reshape(arr.shape[0], num_channels, num_bands)
        if arr.shape[0] == num_channels * num_bands:
            return arr.T.reshape(arr.shape[1], num_channels, num_bands)
    if arr.ndim != 3:
        raise ValueError(f'Unsupported feature shape: {arr.shape}')
    shape = arr.shape
    # [B, C, T]
    if shape[0] == num_bands and shape[1] == num_channels:
        return np.transpose(arr, (2, 1, 0))
    # [C, B, T]
    if shape[0] == num_channels and shape[1] == num_bands:
        return np.transpose(arr, (2, 0, 1))
    # [C, T, B]
    if shape[0] == num_channels and shape[2] == num_bands:
        return np.transpose(arr, (1, 0, 2))
    # [T, C, B]
    if shape[1] == num_channels and shape[2] == num_bands:
        return arr
    # [T, B, C]
    if shape[1] == num_bands and shape[2] == num_channels:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f'Cannot infer [T,C,B] from shape {shape}')


def build_temporal_windows(feature_tcb, label, time_steps=10, stride=1):
    feature_tcb = np.asarray(feature_tcb, dtype=np.float32)
    xs, ys = [], []
    for start in range(0, max(feature_tcb.shape[0] - time_steps + 1, 0), stride):
        xs.append(feature_tcb[start:start + time_steps])
        ys.append(label)
    if not xs:
        return np.empty((0, time_steps) + feature_tcb.shape[1:], dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def trial_key_sort_key(key):
    nums = re.findall(r'\d+', key)
    return int(nums[-1]) if nums else 10**9
