"""Weighted F1, macro F1, per-class F1, and the confusion matrix for the 7 MELD labels."""
from sklearn.metrics import confusion_matrix, f1_score

from meldrt.labels import LABELS

LABEL_IDS = list(range(len(LABELS)))


def f1_summary(y_true, y_pred):
    # labels= is passed so a class nobody predicted still counts (as 0) in the averages
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
    }


def per_class_f1(y_true, y_pred):
    scores = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    return {name: float(score) for name, score in zip(LABELS, scores)}


def confusion(y_true, y_pred):
    """7 x 7 counts: rows are the true label, columns the predicted label."""
    return confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
