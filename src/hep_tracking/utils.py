"""Evaluation metrics and threshold optimization utilities for classification."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
)

def calculate_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate a comprehensive suite of binary classification metrics.

    Evaluates both continuous probability metrics (ROC-AUC, PR-AUC) and 
    discrete binary metrics (F1, Precision, Recall, Accuracy) by converting 
    the probabilities using the provided threshold.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Probability threshold used to convert continuous probabilities 
            into discrete binary predictions. Defaults to 0.5.

    Returns:
        A dictionary containing the calculated scores for ROC-AUC, PR-AUC, 
        F1-Score, Precision, Recall, and Accuracy.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
        "PR-AUC": average_precision_score(y_true, y_pred_proba),
        "F1-Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }

    return metrics

def find_best_f1_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[float, float]:
    """Determine the optimal probability threshold that maximizes the F1 score.

    Computes the precision-recall curve and iterates over the generated thresholds 
    to find the exact probability cutoff that yields the highest F1 score for the 
    given dataset. Divides safely by zero where precision and recall are both zero.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities for the positive class.

    Returns:
        A tuple containing two floats:
            - The optimal probability threshold.
            - The corresponding maximum F1 score achieved at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    
    return thresholds[best_idx], f1_scores[best_idx]