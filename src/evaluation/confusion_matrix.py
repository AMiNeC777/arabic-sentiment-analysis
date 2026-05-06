"""ghofyy de moi"""
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)
