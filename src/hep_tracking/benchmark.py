import time
from typing import Sequence
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hep_tracking.config import KNNModelConfig
from hep_tracking.models import BaseKNN
from hep_tracking.dataset import TrackDataset

class ANNBenchmarkRunner:
    def __init__(self, k_neighbors: int = 5, warmup_runs: int = 1, num_runs: int = 3):
        self.k_neighbors = k_neighbors
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs

    def _calculate_recall(self, true_indices: np.ndarray, pred_indices: np.ndarray) -> float:
        n_samples, k = true_indices.shape
        hits = (true_indices[:, :, None] == pred_indices[:, None, :]).any(axis=2).sum()
        return float(hits) / (n_samples * k)

    def run(
        self, 
        models_configs: Sequence[KNNModelConfig], 
        datasets: dict[str, TrackDataset], 
        ground_truth: dict[str, np.ndarray]
    ) -> pd.DataFrame:
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

        return pd.DataFrame(results)