import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()
