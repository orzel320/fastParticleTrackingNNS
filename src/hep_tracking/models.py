"""Wrappers for exact and approximate nearest neighbor (ANN) search algorithms.

This module provides a unified interface for evaluating various nearest neighbor 
implementations, including brute-force matrix operations (NumPy/CuPy), tree-based 
methods (SciPy/Scikit-learn), and highly optimized ANN libraries (FAISS/HNSWlib).
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import faiss
import hnswlib
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

_GPU_RES = faiss.StandardGpuResources()


class BaseKNN(ABC):
    """Abstract base class defining the standard interface for K-Nearest Neighbors search."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Train or construct the underlying search index.

        Args:
            X: Feature matrix used to populate the search index.
        """
        pass

    @abstractmethod
    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the index for the nearest neighbors of the provided points.

        Args:
            X: Feature matrix of the query points.
            k: The number of nearest neighbors to retrieve for each query point.

        Returns:
            A tuple containing two arrays:
                - distances: Array of shape (n_samples, k) with distances to the neighbors.
                - indices: Array of shape (n_samples, k) with the indices of the neighbors.
        """
        pass


class NumpyBruteForce(BaseKNN):
    """Exact nearest neighbor search using batched NumPy matrix operations.

    This implementation computes pairwise distances manually. To prevent memory 
    overflows on large datasets, queries are processed in chunks.

    Attributes:
        max_mem_bytes: Maximum memory footprint allocated for the distance 
            computation block.
        X_train: The indexed dataset stored in memory.
    """
    def __init__(self, max_mem_bytes: int = 512 * 1024 * 1024):
        """Initialize the NumPy brute-force index.

        Args:
            max_mem_bytes: Limit for internal memory allocations during queries. 
                Defaults to 512 MB.
        """
        self.max_mem_bytes = max_mem_bytes
        self.X_train = None

    def fit(self, X: np.ndarray) -> None:
        """Store the training data in memory.

        Args:
            X: Feature matrix to index.
        """
        self.X_train = X

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbors using batched matrix multiplication.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        n_samples = X.shape[0]
        n_train = self.X_train.shape[0]
        
        squared_norms_train = np.sum(self.X_train * self.X_train, axis=1)
        squared_norms_query = np.sum(X * X, axis=1)

        nearest_distances = np.empty((n_samples, k), dtype=np.float32)
        nearest_indices = np.empty((n_samples, k), dtype=np.int64)

        bytes_per_row = n_train * 12 * 3
        chunk_size = max(1, self.max_mem_bytes // bytes_per_row)

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)

            distances = (
                squared_norms_query[start_idx:end_idx, None]
                + squared_norms_train[None, :]
                - 2.0 * (X[start_idx:end_idx] @ self.X_train.T)
            )
            distances = np.maximum(distances, 0.0)

            rows = np.arange(start_idx, end_idx)
            if n_samples == n_train:
                distances[rows - start_idx, rows] = np.inf

            partitioned_indices = np.argpartition(distances, k, axis=1)[:, :k]

            for i in range(end_idx - start_idx):
                order = np.argsort(distances[i, partitioned_indices[i]])
                nearest_indices[start_idx + i] = partitioned_indices[i, order]
                nearest_distances[start_idx + i] = np.sqrt(
                    distances[i, partitioned_indices[i, order]]
                )

        return nearest_distances, nearest_indices


try:
    import cupy as cp
    
    class CuPyBruteForce(BaseKNN):
        """Exact nearest neighbor search using batched CuPy matrix operations on GPU.

        Behaves identically to NumpyBruteForce but utilizes GPU acceleration 
        and explicitly manages the CuPy memory pool to prevent VRAM exhaustion.

        Attributes:
            max_vram_bytes: Limit for VRAM allocations during chunked queries.
            X_train: The indexed dataset stored in GPU memory.
        """
        def __init__(self, max_vram_bytes: int = 512 * 1024 * 1024):
            """Initialize the CuPy brute-force index.

            Args:
                max_vram_bytes: Limit for internal VRAM allocations. Defaults to 512 MB.
            """
            self.max_vram_bytes = max_vram_bytes
            self.X_train = None

        def fit(self, X: np.ndarray) -> None:
            """Transfer and store the training data in GPU memory.

            Args:
                X: Feature matrix to index.
            """
            self.X_train = cp.asarray(X)

        def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Query nearest neighbors using batched GPU operations.

            Args:
                X: Feature matrix of query points.
                k: Number of neighbors to retrieve.

            Returns:
                A tuple of (distances, indices) to the nearest neighbors.
            """
            X_gpu = cp.asarray(X)
            n_samples = X_gpu.shape[0]
            n_train = self.X_train.shape[0]

            squared_norms_train = cp.sum(self.X_train * self.X_train, axis=1)
            squared_norms_query = cp.sum(X_gpu * X_gpu, axis=1)

            nearest_indices = np.empty((n_samples, k), dtype=np.int64)
            nearest_distances = np.empty((n_samples, k), dtype=np.float32)

            bytes_per_row = n_train * 12 * 3
            chunk_size = max(1, self.max_vram_bytes // bytes_per_row)

            mempool = cp.get_default_memory_pool()

            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)

                distances = (
                    squared_norms_query[start_idx:end_idx, None]
                    + squared_norms_train[None, :]
                    - 2.0 * (X_gpu[start_idx:end_idx] @ self.X_train.T)
                )
                distances = cp.maximum(distances, 0.0)

                rows = cp.arange(start_idx, end_idx)
                if n_samples == n_train:
                    distances[rows - start_idx, rows] = cp.inf

                partitioned_indices = cp.argpartition(distances, k, axis=1)[:, :k]

                dist_cpu = distances.get()
                part_idx_cpu = partitioned_indices.get()

                for i in range(end_idx - start_idx):
                    sorted_order = np.argsort(dist_cpu[i, part_idx_cpu[i]])
                    nearest_indices[start_idx + i] = part_idx_cpu[i, sorted_order]
                    nearest_distances[start_idx + i] = np.sqrt(
                        dist_cpu[i, part_idx_cpu[i, sorted_order]]
                    )

                del distances
                del partitioned_indices
                mempool.free_all_blocks()

            return nearest_distances, nearest_indices

except ImportError:
    pass


class FaissExact(BaseKNN):
    """Exact L2 nearest neighbor search using FAISS (IndexFlatL2).
    
    Attributes:
        use_gpu: Indicates whether the index is transferred to the GPU.
        index: The underlying FAISS index object.
    """
    def __init__(self, use_gpu: bool = False):
        """Initialize the exact FAISS index.

        Args:
            use_gpu: If True, uses the standard GPU resources for evaluation. 
                Defaults to False.
        """
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the FAISS FlatL2 index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        cpu_index = faiss.IndexFlatL2(features_contig.shape[1])
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index
        self.index.add(features_contig)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the FAISS FlatL2 index.

        Retrieves `k + 1` neighbors and discards the first one (the point itself) 
        to ensure proper distance matrices.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)
        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()
        return distances[:, 1:], indices[:, 1:]


class FaissIVFFlat(BaseKNN):
    """Approximate nearest neighbor search using FAISS Inverted File index (IndexIVFFlat).

    Attributes:
        nlist: Number of Voronoi cells (clusters) used to partition the data.
        nprobe: Number of cells visited during the search phase.
        use_gpu: Indicates whether GPU resources are utilized.
        index: The underlying FAISS index object.
    """
    def __init__(self, nlist: int = 100, nprobe: int = 1, use_gpu: bool = False):
        """Initialize the IVFFlat FAISS index.

        Args:
            nlist: Number of clusters. Defaults to 100.
            nprobe: Number of clusters to visit during query. Defaults to 1.
            use_gpu: If True, transfers index to the GPU. Defaults to False.
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Train the quantizer and construct the IVF index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        dimension = features_contig.shape[1]

        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.train(features_contig)
        self.index.add(features_contig)
        self.index.nprobe = self.nprobe

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the FAISS IVFFlat index.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class FaissIVFPQ(BaseKNN):
    """Approximate search using FAISS IVF with Product Quantization (IndexIVFPQ).

    Compresses vectors into codes to optimize search speed and memory usage.
    
    Attributes:
        nlist: Number of Voronoi cells (clusters).
        m: Number of sub-vectors for product quantization.
        nbits: Number of bits per sub-vector index.
        nprobe: Number of clusters to visit during query.
        use_gpu: Indicates whether GPU resources are utilized.
        index: The underlying FAISS index object.
    """
    def __init__(self, nlist: int = 100, m: int = 5, nbits: int = 8, nprobe: int = 1, use_gpu: bool = False):
        """Initialize the IVFPQ FAISS index.

        Args:
            nlist: Number of clusters. Defaults to 100.
            m: Number of sub-quantizers. Must divide the dimension space evenly. Defaults to 5.
            nbits: Bits allocated per sub-quantizer. Defaults to 8.
            nprobe: Number of clusters to probe during search. Defaults to 1.
            use_gpu: If True, uses GPU acceleration. Defaults to False.
        """
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Train the sub-quantizers and construct the IVFPQ index.

        Args:
            X: Feature matrix to index.

        Raises:
            ValueError: If the feature dimension is not perfectly divisible by `m`.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        dimension = features_contig.shape[1]

        if dimension % self.m != 0:
            raise ValueError(f"Wymiar przestrzeni ({dimension}) musi byÄ‡ podzielny przez m ({self.m}).")

        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, self.m, self.nbits)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.train(features_contig)
        self.index.add(features_contig)
        self.index.nprobe = self.nprobe

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the FAISS IVFPQ index.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class HnswGraph(BaseKNN):
    """Approximate nearest neighbor search using HNSW (Hierarchical Navigable Small World) graphs.
    
    Attributes:
        m: Number of bi-directional links created for every new element.
        ef_construction: Size of the dynamic list for the nearest neighbors during index creation.
        ef: Size of the dynamic list for the nearest neighbors during search.
        num_threads: Number of threads used by hnswlib. Defaults to -1 (all available).
        index: The underlying hnswlib index object.
    """
    def __init__(self, m: int = 16, ef_construction: int = 200, ef: int = 50, num_threads: int = -1):
        """Initialize the HNSW index.

        Args:
            m: Max links per node. Defaults to 16.
            ef_construction: Search depth during index build. Defaults to 200.
            ef: Search depth during query. Defaults to 50.
            num_threads: Number of CPU threads to utilize. Defaults to -1.
        """
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the HNSW graph index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        n_samples, dimension = features_contig.shape

        self.index = hnswlib.Index(space="l2", dim=dimension)
        self.index.init_index(max_elements=n_samples, ef_construction=self.ef_construction, M=self.m)
        self.index.set_num_threads(self.num_threads)
        self.index.add_items(features_contig)
        self.index.set_ef(self.ef)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the HNSW index.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        indices, distances = self.index.knn_query(features_contig, k=k + 1)

        return distances[:, 1:].astype(np.float32), indices[:, 1:]


class ScipyCKDTree(BaseKNN):
    """Exact nearest neighbor search using SciPy's cKDTree implementation.
    
    Attributes:
        workers: Number of threads used during querying. Defaults to -1 (all available).
        tree: The underlying SciPy cKDTree object.
    """
    def __init__(self, workers: int = -1):
        """Initialize the cKDTree wrapper.

        Args:
            workers: Number of worker threads for parallel queries. Defaults to -1.
        """
        self.workers = workers
        self.tree = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the KD-Tree index.

        Args:
            X: Feature matrix to index.
        """
        self.tree = cKDTree(X)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the KD-Tree for exact neighbors.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        distances, indices = self.tree.query(X, k=k + 1, workers=self.workers)
        return distances[:, 1:].astype(np.float32), indices[:, 1:]


class SklearnKNN(BaseKNN):
    """Exact nearest neighbor search using scikit-learn's NearestNeighbors.

    Attributes:
        algorithm: Algorithm utilized by scikit-learn (e.g., "kd_tree" or "ball_tree").
        leaf_size: Leaf size parameter regulating tree node density.
        n_jobs: Number of parallel jobs used for querying.
        nn: The underlying scikit-learn NearestNeighbors estimator.
    """
    def __init__(self, algorithm: Literal["kd_tree", "ball_tree"] = "kd_tree", leaf_size: int = 100, n_jobs: int = -1):
        """Initialize the Scikit-learn KNN wrapper.

        Args:
            algorithm: Tree algorithm to use ("kd_tree" or "ball_tree"). Defaults to "kd_tree".
            leaf_size: Number of points at which to switch to brute-force. Defaults to 100.
            n_jobs: Number of parallel jobs for queries. Defaults to -1.
        """
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.nn = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the scikit-learn nearest neighbor tree.

        Args:
            X: Feature matrix to index.
        """
        self.nn = NearestNeighbors(
            n_neighbors=1, 
            algorithm=self.algorithm, 
            leaf_size=self.leaf_size, 
            n_jobs=self.n_jobs
        )
        self.nn.fit(X)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query the scikit-learn estimator for exact neighbors.

        Args:
            X: Feature matrix of query points.
            k: Number of neighbors to retrieve.

        Returns:
            A tuple of (distances, indices) to the nearest neighbors.
        """
        distances, indices = self.nn.kneighbors(X, n_neighbors=k + 1)
        return distances[:, 1:].astype(np.float32), indices[:, 1:]