import time

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

import numpy as np

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