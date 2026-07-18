import numpy as np
from hep_tracking.utils import calculate_classification_metrics


def test_calculate_classification_metrics_perfect_predictions():
    """Dla idealnych predykcji wszystkie metryki powinny wynosić dokładnie 1.0."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.01, 0.02, 0.9, 0.95, 0.99])

    metrics = calculate_classification_metrics(y_true, y_pred_proba)

    assert metrics["ROC-AUC"] == 1.0
    assert metrics["PR-AUC"] == 1.0
    assert metrics["F1-Score"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["Accuracy"] == 1.0


def test_calculate_classification_metrics_threshold_effect():
    """Sprawdza, że zmiana progu decyzyjnego (threshold) wpływa na Precision/Recall,
    ale nie wpływa na ROC-AUC/PR-AUC (które są niezależne od progu)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.3, 0.6, 0.4, 0.7])

    metrics_low = calculate_classification_metrics(y_true, y_pred_proba, threshold=0.35)
    metrics_high = calculate_classification_metrics(y_true, y_pred_proba, threshold=0.65)

    assert metrics_low["ROC-AUC"] == metrics_high["ROC-AUC"]
    assert metrics_low["PR-AUC"] == metrics_high["PR-AUC"]
    assert metrics_low["Recall"] >= metrics_high["Recall"]


def test_calculate_classification_metrics_handles_zero_division():
    """Sprawdza, że przy braku pozytywnych predykcji Precision/Recall
    zwracają 0 zamiast rzucać wyjątkiem (zero_division=0)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.1, 0.1, 0.1])  # wszystko poniżej progu 0.5

    metrics = calculate_classification_metrics(y_true, y_pred_proba)

    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0