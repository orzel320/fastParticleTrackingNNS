"""Benchmarking utilities for Approximate Nearest Neighbor (ANN) models."""

import time
import gc
from typing import Sequence
import numpy as np
import pandas as pd

from hep_tracking.config import KNNModelConfig
from hep_tracking.models import BaseKNN
from hep_tracking.dataset import TrackDataset


class ANNBenchmarkRunner:
    """Runner for evaluating nearest neighbor search models against datasets.

    This class manages the lifecycle of benchmarking, including warmup iterations, 
    timed query runs, recall calculation, and explicit memory management to prevent 
    OOM (Out Of Memory) errors during sequential evaluations.

    Attributes:
        k_neighbors: Number of nearest neighbors to retrieve during queries.
        warmup_runs: Number of un-timed query executions to run before benchmarking.
        num_runs: Number of timed query executions used to determine minimum query time.
    """

    def __init__(self, k_neighbors: int = 5, warmup_runs: int = 1, num_runs: int = 3):
        """Initialize the benchmark runner configuration.

        Args:
            k_neighbors: Number of neighbors to find. Defaults to 5.
            warmup_runs: Number of warmup iterations. Defaults to 1.
            num_runs: Number of measured runs. Defaults to 3.
        """
        self.k_neighbors = k_neighbors
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs

    def _calculate_recall(self, true_indices: np.ndarray, pred_indices: np.ndarray) -> float:
        """Calculate the exact recall between ground truth and predicted indices.

        Args:
            true_indices: Array of shape (n_samples, k) containing correct neighbor indices.
            pred_indices: Array containing model-predicted neighbor indices.

        Returns:
            The fraction of correctly identified nearest neighbors.
        """
        n_samples, k = true_indices.shape
        hits = (true_indices[:, :, None] == pred_indices[:, None, :]).any(axis=2).sum()
        return float(hits) / (n_samples * k)

    def run(
        self, 
        models_configs: Sequence[KNNModelConfig], 
        datasets: dict[str, TrackDataset], 
        ground_truth: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Execute the benchmark suite on the provided models and datasets.

        Iterates through the provided datasets and evaluates each model configuration.
        For each combination, it measures model build time, executes warmup runs, 
        measures query times over multiple runs, and calculates Queries Per Second (QPS) 
        and recall. Memory is explicitly collected between model evaluations, 
        including CuPy memory pools if available, to ensure clean runs.

        Args:
            models_configs: A sequence of configuration objects defining the models 
                to instantiate and benchmark.
            datasets: A dictionary mapping dataset names to dataset instances.
            ground_truth: A dictionary mapping dataset names to their corresponding 
                ground truth index arrays.

        Returns:
            A pandas DataFrame containing performance metrics (Build_Time_s, 
            Query_Time_s, QPS, Recall) for every tested model and dataset combination.
        """
        results = []

        for dataset_name, dataset in datasets.items():
            print(f"--- Dataset: {dataset_name} | Kształt: {dataset.X.shape} ---")
            true_idx = ground_truth[dataset_name]

            for config in models_configs:
                print(f"Ewaluacja modelu: {config.name}")
                
                model: BaseKNN = config.model_factory(**config.model_kwargs)
                
                start_build = time.perf_counter()
                model.fit(dataset.X)
                build_time = time.perf_counter() - start_build

                for _ in range(self.warmup_runs):
                    model.kneighbors(dataset.X, self.k_neighbors)

                query_times = []
                distances, indices = None, None
                for _ in range(self.num_runs):
                    start_query = time.perf_counter()
                    distances, indices = model.kneighbors(dataset.X, self.k_neighbors)
                    query_times.append(time.perf_counter() - start_query)

                min_query_time = min(query_times)
                qps = len(dataset.X) / min_query_time
                recall = self._calculate_recall(true_idx, indices)

                results.append({
                    "Dataset": dataset_name,
                    "Model": config.name,
                    "Build_Time_s": build_time,
                    "Query_Time_s": min_query_time,
                    "QPS": qps,
                    "Recall": recall
                })

                del model
                gc.collect()
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except ImportError:
                    pass

        return pd.DataFrame(results)