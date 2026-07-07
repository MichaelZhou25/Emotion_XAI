import numpy as np


def sorted_unique_subjects(subject_ids):
    return sorted([int(s) for s in np.unique(subject_ids)])
