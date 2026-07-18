import time

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

def measure_execution_time(target_function, num_runs=3, warmup_runs=1):
    """Measures the minimum execution time of a function over multiple runs.

    Includes a warm-up phase to avoid initialization overhead in the final timing.
    This function is agnostic to the task and can be used for kNN, ANN, or Classifiers.

    :param target_function: A callable function taking no arguments to be timed.
    :type target_function: callable
    :param num_runs: Number of timed executions.
    :type num_runs: int
    :param warmup_runs: Number of un-timed warm-up executions.
    :type warmup_runs: int
    :return: The minimum execution time across all timed runs in seconds.
    :rtype: float
    """
    for _ in range(warmup_runs):
        target_function()

    execution_times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        target_function()
        execution_times.append(time.perf_counter() - start_time)

    return min(execution_times)

def calculate_recall(true_indices: np.ndarray, pred_indices: np.ndarray) -> float:
    """Calculates the recall of an Approximate Nearest Neighbors (ANN) search.

    Args:
        true_indices (np.ndarray): Exact nearest neighbor indices of shape (N, k).
        pred_indices (np.ndarray): Approximate nearest neighbor indices of shape (N, k).

    Returns:
        float: The mean recall score across all queries, ranging from 0.0 to 1.0.
    """
    n_samples, k_neighbors = true_indices.shape

    hits = (true_indices[:, :, None] == pred_indices[:, None, :]).any(axis=2).sum()

    return float(hits) / (n_samples * k_neighbors)

def pad_features(features: np.ndarray, target_dim: int = 8) -> np.ndarray:
    """Pads the feature matrix with zeros to reach a target dimensionality.

    Useful for meeting hardware-specific memory alignment constraints 
    (e.g., FAISS GPU Product Quantization needing m=8). Also ensures the 
    returned array is C-contiguous and of type float32.

    Args:
        features (np.ndarray): Original feature matrix of shape (N, D).
        target_dim (int): The desired number of dimensions. Defaults to 8.

    Returns:
        np.ndarray: A padded, contiguous float32 matrix of shape (N, target_dim).
    """
    current_dim = features.shape[1]
    
    if current_dim >= target_dim:
        return np.ascontiguousarray(features, dtype=np.float32)
        
    pad_width = target_dim - current_dim
    padded_features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    
    return np.ascontiguousarray(padded_features, dtype=np.float32)


def evaluate_ann_model(model_name: str, model_instance, features: np.ndarray, true_indices: np.ndarray, k: int):
    """Builds the ANN index, queries it, and returns performance metrics.

    Prints the progress and timing to the console. The query time is measured 
    strictly for the query phase, excluding index construction time.

    Args:
        model_name (str): Display name of the model for logging.
        model_instance (object): The initialized ANN wrapper instance.
        features (np.ndarray): Feature matrix to build the index and query on.
        true_indices (np.ndarray): Ground truth exact nearest neighbor indices.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        tuple[float, float]: A tuple containing (QPS, Recall).
    """
    print(f"Trenowanie i budowa indeksu: {model_name}...")
    
    model_instance.build(features)
    
    start_time = time.perf_counter()
    _, pred_indices = model_instance.query(features, k)
    query_time = time.perf_counter() - start_time
    
    qps = features.shape[0] / query_time
    recall = calculate_recall(true_indices, pred_indices)
    
    print(f" -> QPS: {qps:,.0f} | Recall: {recall:.4f}\n")
    return qps, recall

def calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculates standard classification metrics for model evaluation.

    :param y_true: Ground truth binary labels.
    :type y_true: numpy.ndarray
    :param y_pred_proba: Predicted probabilities for the positive class.
    :type y_pred_proba: numpy.ndarray
    :param threshold: Probability threshold for converting probabilities to binary predictions.
    :type threshold: float
    :return: Dictionary containing ROC-AUC, PR-AUC, F1, Precision, Recall, and Accuracy.
    :rtype: dict
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