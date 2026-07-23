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
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    
    return thresholds[best_idx], f1_scores[best_idx]