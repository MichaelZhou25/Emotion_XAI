from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.seed import prepare_seed_features
from data.seed_iv import prepare_seediv_features


@dataclass
class FeatureStore:
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    session_id: np.ndarray
    trial_id: np.ndarray

    @property
    def subjects(self):
        return sorted([int(s) for s in np.unique(self.subject_id)])


def make_synthetic_store(cfg):
    ds = cfg['dataset']
    n_sub = ds.get('num_subjects', 15)
    n_cls = ds.get('num_classes', 3)
    T = ds.get('time_steps', 10); C = ds.get('num_channels', 62); B = ds.get('num_bands', 5)
    samples_per_subject = ds.get('synthetic', {}).get('samples_per_subject', 120)
    rng = np.random.default_rng(cfg['train'].get('seed', 2026))
    xs, ys, subs, sess, trials = [], [], [], [], []
    class_patterns = rng.normal(0, 0.6, size=(n_cls, T, C, B)).astype(np.float32)
    for s in range(n_sub):
        subj_shift = rng.normal(0, 0.25, size=(1, C, B)).astype(np.float32)
        for i in range(samples_per_subject):
            y = i % n_cls
            noise = rng.normal(0, 1.0, size=(T, C, B)).astype(np.float32)
            x = class_patterns[y] + subj_shift + noise
            xs.append(x); ys.append(y); subs.append(s); sess.append(0); trials.append(i)
    return FeatureStore(np.stack(xs), np.array(ys), np.array(subs), np.array(sess), np.array(trials))


def load_npz_store(path):
    arr = np.load(path, allow_pickle=False)
    return FeatureStore(arr['x'].astype(np.float32), arr['y'].astype(np.int64),
                        arr['subject_id'].astype(np.int64), arr['session_id'].astype(np.int64),
                        arr['trial_id'].astype(np.int64))


def load_feature_store(cfg):
    ds = cfg['dataset']
    if ds.get('synthetic', {}).get('enabled', False):
        return make_synthetic_store(cfg)
    processed_path = ds.get('processed_path')
    if processed_path and Path(processed_path).exists():
        return load_npz_store(processed_path)
    root = ds.get('root')
    if not root:
        raise FileNotFoundError('No dataset.processed_path found and dataset.root is empty. Set synthetic.enabled=true for smoke tests.')
    processed_path = processed_path or f"data/cache/{ds['name'].lower().replace('-', '')}_{ds.get('input_type','lds_de')}_s{ds.get('session',1)}_T{ds.get('time_steps',10)}.npz"
    name = ds['name'].upper().replace('_', '-')
    if name == 'SEED':
        prepare_seed_features(root, processed_path, ds.get('time_steps',10), ds.get('stride',1), ds.get('session',1), ds.get('input_type','lds_de'), ds.get('num_channels',62), ds.get('num_bands',5))
    elif name in ['SEED-IV', 'SEEDIV']:
        prepare_seediv_features(root, processed_path, ds.get('time_steps',10), ds.get('stride',1), ds.get('session',1), ds.get('input_type','lds_de'), ds.get('num_channels',62), ds.get('num_bands',5))
    else:
        raise ValueError(ds['name'])
    return load_npz_store(processed_path)


class EEGFeatureDataset(Dataset):
    def __init__(self, store, indices, mean=None, std=None, subject_minmax=None):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std
        self.subject_minmax = subject_minmax

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.store.x[i].astype(np.float32)
        if self.subject_minmax is not None:
            sid = int(self.store.subject_id[i])
            x_min, x_max = self.subject_minmax[sid]
            x = (x - x_min) / (x_max - x_min + 1e-6)
        elif self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
        return {
            'x': torch.from_numpy(x).float(),
            'y': torch.tensor(int(self.store.y[i]), dtype=torch.long),
            'subject_id': torch.tensor(int(self.store.subject_id[i]), dtype=torch.long),
            'session_id': torch.tensor(int(self.store.session_id[i]), dtype=torch.long),
            'trial_id': torch.tensor(int(self.store.trial_id[i]), dtype=torch.long),
        }


def indices_for_subjects(store, subjects):
    subjects = np.asarray(list(map(int, subjects)), dtype=np.int64)
    return np.where(np.isin(store.subject_id, subjects))[0]


def compute_train_norm(store, train_subjects):
    idx = indices_for_subjects(store, train_subjects)
    x = store.x[idx]
    # Per channel-band normalization across samples and time.
    mean = x.mean(axis=(0, 1), keepdims=False).astype(np.float32)[None, :, :]  # [1,C,B]
    std = x.std(axis=(0, 1), keepdims=False).astype(np.float32)[None, :, :]
    return mean, std


def compute_subject_minmax(store):
    stats = {}
    for sid in store.subjects:
        x = store.x[store.subject_id == int(sid)]
        stats[int(sid)] = (
            x.min(axis=0, keepdims=False).astype(np.float32),
            x.max(axis=0, keepdims=False).astype(np.float32),
        )
    return stats


def build_loader(store, subjects, cfg, shuffle=False, batch_size=None, mean=None, std=None, subject_minmax=None):
    idx = indices_for_subjects(store, subjects)
    if len(idx) == 0:
        raise ValueError(f'No samples for subjects: {subjects}')
    ds = EEGFeatureDataset(store, idx, mean=mean, std=std, subject_minmax=subject_minmax)
    return DataLoader(ds, batch_size=batch_size or cfg['train']['batch_size'], shuffle=shuffle,
                      num_workers=cfg['train'].get('num_workers', 0), drop_last=False)


def build_fold_loaders(store, train_subjects, val_subjects, test_subjects, cfg):
    mean = std = None
    subject_minmax = None
    normalize = cfg['dataset'].get('normalize', 'train')
    if normalize == 'train':
        mean, std = compute_train_norm(store, train_subjects)
    elif normalize == 'subject_minmax':
        subject_minmax = compute_subject_minmax(store)
    train_loader = build_loader(store, train_subjects, cfg, shuffle=True, mean=mean, std=std, subject_minmax=subject_minmax)
    val_loader = None if not val_subjects else build_loader(store, val_subjects, cfg, shuffle=False, mean=mean, std=std, subject_minmax=subject_minmax)
    test_loader = build_loader(store, test_subjects, cfg, shuffle=False, mean=mean, std=std, subject_minmax=subject_minmax)
    return train_loader, val_loader, test_loader
