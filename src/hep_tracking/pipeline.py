"""Pipeline evaluation strategies for high-energy physics track building."""

import time
import numpy as np

from hep_tracking.dataset import TrackDataset
from hep_tracking.features import compute_pair_features

class PipelineEvaluator:
    """Evaluates different candidate generation and filtering pipelines.

    This class provides standardized methods for running and timing various 
    track-building pipelines (e.g., geometric cuts, machine learning, and brute-force). 
    It automatically calculates the theoretical maximum number of valid pairs 
    in the dataset to establish baseline efficiency.

    Attributes:
        dataset: The dataset containing hit features and ground truth labels.
        k_neighbors: The number of nearest neighbors to retrieve during the 
            candidate generation phase.
        geom_cuts: Dictionary containing threshold values for geometric filtering 
            (e.g., 'max_delta_r', 'min_dot_product').
        ml_threshold: Probability cutoff for the machine learning classifier.
        true_pairs_count: Precalculated total number of valid signal pairs 
            possible within the dataset.
    """

    def __init__(self, dataset: TrackDataset, k_neighbors: int, geom_cuts: dict, ml_threshold: float = 0.5, warmup_runs: int = 0, num_runs: int = 1):
        """Initialize the pipeline evaluator.

        Args:
            dataset: The target tracking dataset to evaluate.
            k_neighbors: Number of nearest neighbors for initial retrieval.
            geom_cuts: Dictionary defining the hard physical limits for pairs.
            ml_threshold: Probability threshold for positive pair classification. 
                Defaults to 0.5.
        """
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.geom_cuts = geom_cuts
        self.ml_threshold = ml_threshold
        self.warmup_runs = warmup_runs
        self.num_runs = max(1, num_runs)
        
        self.true_pairs_count = self._calculate_total_true_pairs()
        
    def _calculate_total_true_pairs(self) -> int:
        """Calculate the theoretical maximum number of valid hit pairs in the dataset.

        Returns:
            The total number of valid pairs (ignoring noise hits labeled as -1).
        """
        valid_labels = self.dataset.y[self.dataset.y != -1]
        _, counts = np.unique(valid_labels, return_counts=True)
        return int(np.sum(counts * (counts - 1)))
        
    def _evaluate_predictions(self, query_idx: np.ndarray, neighbor_idx: np.ndarray, mask: np.ndarray) -> dict:
        """Calculate purity and efficiency metrics for a proposed set of pairs.

        Args:
            query_idx: Array of indices representing the initial hit in the pair.
            neighbor_idx: Array of indices representing the proposed neighbor hit.
            mask: Boolean array indicating which pairs passed the filtering stage.

        Returns:
            A dictionary containing the Purity, Efficiency, and total count of 
            Proposed_Pairs.
        """
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

    def run_geometric_pipeline(self, retriever, X_retrieval=None) -> dict:
        X_search = X_retrieval if X_retrieval is not None else self.dataset.X
        
        def _execute():
            t0 = time.perf_counter()
            retriever.fit(X_search)
            _, indices = retriever.kneighbors(X_search, self.k_neighbors)
            
            n_queries = len(X_search)
            query_idx = np.repeat(np.arange(n_queries), self.k_neighbors)
            neighbor_idx = indices.flatten()
            
            valid_pairs = query_idx != neighbor_idx
            query_idx_filtered = query_idx[valid_pairs]
            neighbor_idx_filtered = neighbor_idx[valid_pairs]
            t_retrieval = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            features = compute_pair_features(self.dataset.X[query_idx_filtered], self.dataset.X[neighbor_idx_filtered])
            t_features = time.perf_counter() - t1
            
            t2 = time.perf_counter()
            delta_r = np.abs(features[:, 3])
            dot_prod = features[:, 6]
            pass_cuts = (delta_r <= self.geom_cuts["max_delta_r"]) & (dot_prod >= self.geom_cuts["min_dot_product"])
            t_filter = time.perf_counter() - t2
            
            return t_retrieval, t_features, t_filter, query_idx_filtered, neighbor_idx_filtered, pass_cuts

        for _ in range(self.warmup_runs):
            _execute()

        total_t_retrieval = 0.0
        total_t_features = 0.0
        total_t_filter = 0.0
        
        for _ in range(self.num_runs):
            t_ret, t_feat, t_filt, query_idx, neighbor_idx, pass_cuts = _execute()
            total_t_retrieval += t_ret
            total_t_features += t_feat
            total_t_filter += t_filt

        avg_t_retrieval = total_t_retrieval / self.num_runs
        avg_t_features = total_t_features / self.num_runs
        avg_t_filter = total_t_filter / self.num_runs

        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_cuts)
        
        return {
            "Time_Retrieval_s": avg_t_retrieval,
            "Time_Features_s": avg_t_features,
            "Time_Filter_s": avg_t_filter,
            "Time_Total_s": avg_t_retrieval + avg_t_features + avg_t_filter,
            **metrics
        }

    def run_ml_pipeline(self, retriever, classifier, X_retrieval=None) -> dict:
        X_search = X_retrieval if X_retrieval is not None else self.dataset.X
        
        def _execute():
            t0 = time.perf_counter()
            retriever.fit(X_search)
            _, indices = retriever.kneighbors(X_search, self.k_neighbors)
            
            n_queries = len(X_search)
            query_idx = np.repeat(np.arange(n_queries), self.k_neighbors)
            neighbor_idx = indices.flatten()
            
            valid_pairs = query_idx != neighbor_idx
            query_idx_filtered = query_idx[valid_pairs]
            neighbor_idx_filtered = neighbor_idx[valid_pairs]
            t_retrieval = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            features = compute_pair_features(self.dataset.X[query_idx_filtered], self.dataset.X[neighbor_idx_filtered])
            t_features = time.perf_counter() - t1
            
            t2 = time.perf_counter()
            probs = classifier.predict_proba(features)[:, 1]
            pass_ml = probs >= self.ml_threshold
            t_filter = time.perf_counter() - t2
            
            return t_retrieval, t_features, t_filter, query_idx_filtered, neighbor_idx_filtered, pass_ml

        for _ in range(self.warmup_runs):
            _execute()

        total_t_retrieval = 0.0
        total_t_features = 0.0
        total_t_filter = 0.0
        
        for _ in range(self.num_runs):
            t_ret, t_feat, t_filt, query_idx, neighbor_idx, pass_ml = _execute()
            total_t_retrieval += t_ret
            total_t_features += t_feat
            total_t_filter += t_filt

        avg_t_retrieval = total_t_retrieval / self.num_runs
        avg_t_features = total_t_features / self.num_runs
        avg_t_filter = total_t_filter / self.num_runs

        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_ml)
        
        return {
            "Time_Retrieval_s": avg_t_retrieval,
            "Time_Features_s": avg_t_features,
            "Time_Filter_s": avg_t_filter,
            "Time_Total_s": avg_t_retrieval + avg_t_features + avg_t_filter,
            **metrics
        }
        
    def run_all_pairs_pipeline(self, classifier) -> dict:
        """Execute and evaluate a brute-force pipeline comparing all possible hit pairs.

        Generates an O(N^2) pairwise grid across the entire dataset before applying 
        the classifier. This is primarily used as a robust baseline or for very 
        small datasets where index creation overhead outweighs the benefit.

        Args:
            classifier: A trained machine learning model implementing `predict_proba`.

        Returns:
            A dictionary containing timing breakdowns and final classification metrics.
        """
        def _execute():
            t0 = time.perf_counter()
            n_points = len(self.dataset.X)
            q_grid, n_grid = np.meshgrid(np.arange(n_points), np.arange(n_points), indexing='ij')
            
            query_idx = q_grid.flatten()
            neighbor_idx = n_grid.flatten()
            
            valid = query_idx != neighbor_idx
            query_idx_filtered = query_idx[valid]
            neighbor_idx_filtered = neighbor_idx[valid]
            t_retrieval = time.perf_counter() - t0 
            
            t1 = time.perf_counter()
            features = compute_pair_features(self.dataset.X[query_idx_filtered], self.dataset.X[neighbor_idx_filtered])
            t_features = time.perf_counter() - t1
            
            t2 = time.perf_counter()
            probs = classifier.predict_proba(features)[:, 1]
            pass_ml = probs >= self.ml_threshold
            t_filter = time.perf_counter() - t2
            
            return t_retrieval, t_features, t_filter, query_idx_filtered, neighbor_idx_filtered, pass_ml

        for _ in range(self.warmup_runs):
            _execute()

        total_t_retrieval = 0.0
        total_t_features = 0.0
        total_t_filter = 0.0
        
        for _ in range(self.num_runs):
            t_ret, t_feat, t_filt, query_idx, neighbor_idx, pass_ml = _execute()
            total_t_retrieval += t_ret
            total_t_features += t_feat
            total_t_filter += t_filt

        avg_t_retrieval = total_t_retrieval / self.num_runs
        avg_t_features = total_t_features / self.num_runs
        avg_t_filter = total_t_filter / self.num_runs
        
        metrics = self._evaluate_predictions(query_idx, neighbor_idx, pass_ml)
        
        return {
            "Time_Retrieval_s": avg_t_retrieval,
            "Time_Features_s": avg_t_features,
            "Time_Filter_s": avg_t_filter,
            "Time_Total_s": avg_t_retrieval + avg_t_features + avg_t_filter,
            **metrics
        }