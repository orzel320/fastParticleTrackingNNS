import time
import numpy as np

from hep_tracking.dataset import TrackDataset
from hep_tracking.features import compute_pair_features

class PipelineEvaluator:
    def __init__(self, dataset: TrackDataset, k_neighbors: int, geom_cuts: dict, ml_threshold: float = 0.5):
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.geom_cuts = geom_cuts
        self.ml_threshold = ml_threshold
        
        self.true_pairs_count = self._calculate_total_true_pairs()
        
    def _calculate_total_true_pairs(self) -> int:
        valid_labels = self.dataset.y[self.dataset.y != -1]
        _, counts = np.unique(valid_labels, return_counts=True)
        return int(np.sum(counts * (counts - 1)))
        
    def _evaluate_predictions(self, query_idx: np.ndarray, neighbor_idx: np.ndarray, mask: np.ndarray) -> dict:
        final_q = query_idx[mask]
        final_n = neighbor_idx[mask]
        
        proposed_count = len(final_q)
        if proposed_count == 0:
            return {"Purity": 0.0, "Efficiency": 0.0, "Proposed_Pairs": 0}
            
        labels_q = self.dataset.y[final_q]
        labels_n = self.dataset.y[final_n]
        
        is_true_positive = (labels_q == labels_n) & (labels_q != -1)
        tp_count = np.sum(is_true_positive)
        
        purity = tp_count / proposed_count
        efficiency = tp_count / self.true_pairs_count if self.true_pairs_count > 0 else 0.0
        
        return {"Purity": purity, "Efficiency": efficiency, "Proposed_Pairs": proposed_count}

    def run_geometric_pipeline(self, retriever) -> dict:
        t0 = time.perf_counter()
        retriever.fit(self.dataset.X)
        _, indices = retriever.kneighbors(self.dataset.X, self.k_neighbors)
        
        n_queries = len(self.dataset.X)
        query_idx = np.repeat(np.arange(n_queries), self.k_neighbors)
        neighbor_idx = indices.flatten()
        
        valid_pairs = query_idx != neighbor_idx
        query_idx = query_idx[valid_pairs]
        neighbor_idx = neighbor_idx[valid_pairs]
        t_retrieval = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        features = compute_pair_features(self.dataset.X[query_idx], self.dataset.X[neighbor_idx])
        t_features = time.perf_counter() - t1
        
        t2 = time.perf_counter()
        delta_r = np.abs(features[:, 3])
        dot_prod = features[:, 6]
        pass_cuts = (delta_r <= self.geom_cuts["max_delta_r"]) & (dot_prod >= self.geom_cuts["min_dot_product"])
        t_filter = time.perf_counter() - t2
        
        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_cuts)
        
        return {
            "Time_Retrieval_s": t_retrieval,
            "Time_Features_s": t_features,
            "Time_Filter_s": t_filter,
            "Time_Total_s": t_retrieval + t_features + t_filter,
            **metrics
        }

    def run_ml_pipeline(self, retriever, classifier) -> dict:
        t0 = time.perf_counter()
        retriever.fit(self.dataset.X)
        _, indices = retriever.kneighbors(self.dataset.X, self.k_neighbors)
        
        n_queries = len(self.dataset.X)
        query_idx = np.repeat(np.arange(n_queries), self.k_neighbors)
        neighbor_idx = indices.flatten()
        
        valid_pairs = query_idx != neighbor_idx
        query_idx = query_idx[valid_pairs]
        neighbor_idx = neighbor_idx[valid_pairs]
        t_retrieval = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        features = compute_pair_features(self.dataset.X[query_idx], self.dataset.X[neighbor_idx])
        t_features = time.perf_counter() - t1
        
        t2 = time.perf_counter()
        probs = classifier.predict_proba(features)[:, 1]
        pass_ml = probs >= self.ml_threshold
        t_filter = time.perf_counter() - t2
        
        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_ml)
        
        return {
            "Time_Retrieval_s": t_retrieval,
            "Time_Features_s": t_features,
            "Time_Filter_s": t_filter,
            "Time_Total_s": t_retrieval + t_features + t_filter,
            **metrics
        }
        
    def run_all_pairs_pipeline(self, classifier) -> dict:
        t0 = time.perf_counter()
        n_points = len(self.dataset.X)
        q_grid, n_grid = np.meshgrid(np.arange(n_points), np.arange(n_points), indexing='ij')
        
        query_idx = q_grid.flatten()
        neighbor_idx = n_grid.flatten()
        
        valid = query_idx != neighbor_idx
        query_idx = query_idx[valid]
        neighbor_idx = neighbor_idx[valid]
        t_retrieval = time.perf_counter() - t0 
        
        t1 = time.perf_counter()
        features = compute_pair_features(self.dataset.X[query_idx], self.dataset.X[neighbor_idx])
        t_features = time.perf_counter() - t1
        
        t2 = time.perf_counter()
        probs = classifier.predict_proba(features)[:, 1]
        pass_ml = probs >= self.ml_threshold
        t_filter = time.perf_counter() - t2
        
        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_ml)
        
        return {
            "Time_Retrieval_s": t_retrieval,
            "Time_Features_s": t_features,
            "Time_Filter_s": t_filter,
            "Time_Total_s": t_retrieval + t_features + t_filter,
            **metrics
        }