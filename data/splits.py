import numpy as np


def split_source_subjects(source_subjects, val_ratio=0.2, seed=2026):
    source_subjects = list(map(int, source_subjects))
    rng = np.random.default_rng(seed)
    shuffled = np.array(source_subjects)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if val_ratio > 0 else 0
    val_subjects = sorted(shuffled[:val_count].tolist())
    train_subjects = sorted(shuffled[val_count:].tolist())
    if not train_subjects:
        raise ValueError('No training subjects left after validation split.')
    return train_subjects, val_subjects


def strict_loso_split(subjects, target_subject, val_ratio=0.2, seed=2026):
    source_subjects = [int(s) for s in subjects if int(s) != int(target_subject)]
    train_subjects, val_subjects = split_source_subjects(source_subjects, val_ratio, seed + int(target_subject))
    return train_subjects, val_subjects, [int(target_subject)]


def legacy_loso_split(subjects, target_subject):
    train_subjects = [int(s) for s in subjects if int(s) != int(target_subject)]
    return train_subjects, [], [int(target_subject)]
