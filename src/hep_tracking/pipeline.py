"""Pipeline evaluation strategies for high-energy physics track building.[cite: 11]"""

import time
import numpy as np

from hep_tracking.dataset import TrackDataset
from hep_tracking.features import compute_pair_features

class PipelineEvaluator:
    """Evaluates different candidate generation and filtering pipelines.[cite: 11]

    This class provides standardized methods for running and timing various 
    track-building pipelines (e.g., geometric cuts, machine learning, and brute-force). 
    It automatically calculates the theoretical maximum number of valid pairs 
    in the dataset to establish baseline efficiency.[cite: 11]

    Attributes:
        dataset: The dataset containing hit features and ground truth labels.[cite: 11]
        k_neighbors: The number of nearest neighbors to retrieve during the 
            candidate generation phase.[cite: 11]
        geom_cuts: Dictionary containing threshold values for geometric filtering 
            (e.g., 'max_delta_r', 'min_dot_product').[cite: 11]
        ml_threshold: Probability cutoff for the machine learning classifier.[cite: 11]
        true_pairs_count: Precalculated total number of valid signal pairs 
            possible within the dataset.[cite: 11]
    """

    def __init__(self, dataset: TrackDataset, k_neighbors: int, geom_cuts: dict, ml_threshold: float = 0.5):
        """Initialize the pipeline evaluator.[cite: 11]

        Args:
            dataset: The target tracking dataset to evaluate.[cite: 11]
            k_neighbors: Number of nearest neighbors for initial retrieval.[cite: 11]
            geom_cuts: Dictionary defining the hard physical limits for pairs.[cite: 11]
            ml_threshold: Probability threshold for positive pair classification. 
                Defaults to 0.5.[cite: 11]
        """
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.geom_cuts = geom_cuts
        self.ml_threshold = ml_threshold
        
        self.true_pairs_count = self._calculate_total_true_pairs()
        
    def _calculate_total_true_pairs(self) -> int:
        """Calculate the theoretical maximum number of valid hit pairs in the dataset.[cite: 11]

        Returns:
            The total number of valid pairs (ignoring noise hits labeled as -1).[cite: 11]
        """
        valid_labels = self.dataset.y[self.dataset.y != -1]
        _, counts = np.unique(valid_labels, return_counts=True)
        return int(np.sum(counts * (counts - 1)))
        
    def _evaluate_predictions(self, query_idx: np.ndarray, neighbor_idx: np.ndarray, mask: np.ndarray) -> dict:
        """Calculate purity and efficiency metrics for a proposed set of pairs.[cite: 11]

        Args:
            query_idx: Array of indices representing the initial hit in the pair.[cite: 11]
            neighbor_idx: Array of indices representing the proposed neighbor hit.[cite: 11]
            mask: Boolean array indicating which pairs passed the filtering stage.[cite: 11]

        Returns:
            A dictionary containing the Purity, Efficiency, and total count of 
            Proposed_Pairs.[cite: 11]
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
        # Używamy przekazanej przestrzeni do wyszukiwania lub domyślnej 5D
        X_search = X_retrieval if X_retrieval is not None else self.dataset.X
        
        t0 = time.perf_counter()
        retriever.fit(X_search)
        _, indices = retriever.kneighbors(X_search, self.k_neighbors)
        
        n_queries = len(X_search)
        query_idx = np.repeat(np.arange(n_queries), self.k_neighbors)
        neighbor_idx = indices.flatten()
        
        valid_pairs = query_idx != neighbor_idx
        query_idx = query_idx[valid_pairs]
        neighbor_idx = neighbor_idx[valid_pairs]
        t_retrieval = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        # Ekstrakcja cech ZAWSZE używa oryginalnych danych 5D (self.dataset.X)
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

    def run_ml_pipeline(self, retriever, classifier, X_retrieval=None) -> dict:
        X_search = X_retrieval if X_retrieval is not None else self.dataset.X
        
        t0 = time.perf_counter()
        retriever.fit(X_search)
        _, indices = retriever.kneighbors(X_search, self.k_neighbors)
        
        n_queries = len(X_search)
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
        """Execute and evaluate a brute-force pipeline comparing all possible hit pairs.[cite: 11]

        Generates an O(N^2) pairwise grid across the entire dataset before applying 
        the classifier.[cite: 11] This is primarily used as a robust baseline or for very 
        small datasets where index creation overhead outweighs the benefit.[cite: 11]

        Args:
            classifier: A trained machine learning model implementing `predict_proba`.[cite: 11]

        Returns:
            A dictionary containing timing breakdowns and final classification metrics.[cite: 11]
        """
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