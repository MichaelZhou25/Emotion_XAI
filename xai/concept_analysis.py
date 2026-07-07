import numpy as np


def concept_summary(concept_scores, labels=None):
    res = {'mean': concept_scores.mean(axis=0).tolist(), 'std': concept_scores.std(axis=0).tolist()}
    if labels is not None:
        res['by_class'] = {}
        for c in sorted(np.unique(labels).tolist()):
            s = concept_scores[labels == c]
            res['by_class'][int(c)] = {'mean': s.mean(axis=0).tolist(), 'std': s.std(axis=0).tolist()}
    return res
