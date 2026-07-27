"""Wrappers for exact and approximate nearest neighbor (ANN) search algorithms.

This module provides a unified interface for evaluating various nearest neighbor
implementations, including brute-force matrix operations (NumPy/CuPy), tree-based
methods (SciPy/Scikit-learn), and highly optimized ANN libraries (FAISS/HNSWlib).
"""

from abc import ABC, abstractmethod
from typing import Literal

import faiss
import hnswlib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

if hasattr(faiss, "StandardGpuResources"):
    _GPU_RES = faiss.StandardGpuResources()
    HAS_FAISS_GPU = True
else:
    _GPU_RES = None
    HAS_FAISS_GPU = False


def _get_gpu_resources():
    """Lazily create (and cache) the shared FAISS GPU resources handle.

    This is created on first use rather than at module import time, so that
    importing `hep_tracking.models` (e.g. for the CPU-only `ScipyCKDTree`,
    used by `generate_candidates.py` and the test suite) does not require a
    GPU / CUDA runtime to be present on the machine.

    Returns:
        The process-wide `faiss.StandardGpuResources` instance.
    """
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def is_gpu_available() -> bool:
    """Check whether a working FAISS GPU backend is available on this machine.

    The result is cached after the first call, since the underlying check
    (allocating GPU resources) is relatively expensive and the answer cannot
    change during the lifetime of the process.

    Returns:
        True if at least one CUDA GPU is visible to FAISS, False otherwise
        (e.g. no GPU present, no CUDA driver, or a CPU-only `faiss-cpu`
        install with no GPU support compiled in).
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            _GPU_AVAILABLE = faiss.get_num_gpus() > 0
        except AttributeError:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def filter_gpu_configs(configs: list, verbose: bool = True) -> list:
    """Drop model configs that explicitly require GPU (`use_gpu=True`) if none is available.

    Intended for notebooks/cells whose whole point is an intentional CPU vs GPU
    comparison (e.g. crossover-N experiments), where `use_gpu=True` configs are
    deliberate and must NOT be silently downgraded to CPU (that would make the
    comparison meaningless). On a machine without a working GPU, this instead
    removes those specific configs up-front (with a clear message) so the rest
    of the notebook still runs to completion on whatever CPU-only configs remain,
    rather than crashing deep inside the benchmark loop.

    Args:
        configs: A list of `KNNModelConfig` (or similar) objects, each with a
            `.model_kwargs` dict that may contain a `use_gpu` key.
        verbose: If True, prints which configs (if any) were skipped.

    Returns:
        A new list with any `use_gpu=True` configs removed if no GPU is
        available; otherwise the original list unchanged.
    """
    if is_gpu_available():
        return configs

    kept, skipped = [], []
    for cfg in configs:
        if cfg.model_kwargs.get("use_gpu") is True:
            skipped.append(cfg.name)
        else:
            kept.append(cfg)

    if verbose and skipped:
        print(
            f"[GPU niedostępne] Pomijam {len(skipped)} konfiguracji wymagających GPU: "
            f"{', '.join(skipped)}. Reszta eksperymentu uruchomi się normalnie na CPU."
        )

    return kept


def _resolve_use_gpu(requested: bool | None) -> bool:
    """Resolve the effective use_gpu flag for a retriever.

    Args:
        requested: The `use_gpu` value passed by the caller.
            - `None` (default): auto-detect - use GPU if one is available,
              otherwise silently fall back to CPU. This is the recommended
              setting for "just works on whatever machine this runs on".
            - `True`: explicitly require GPU. Raises immediately with a clear
              message if no GPU is available, instead of failing later with
              a confusing FAISS/CUDA error.
            - `False`: explicitly force CPU, even if a GPU is available.

    Returns:
        The concrete boolean to actually use.

    Raises:
        RuntimeError: If `requested=True` but no usable GPU was detected.
    """
    if requested is None:
        return is_gpu_available()
    if requested and not is_gpu_available():
        raise RuntimeError(
            "use_gpu=True zostało zażądane, ale nie wykryto działającego GPU "
            "(FAISS zgłasza 0 dostępnych urządzeń CUDA). Zainstaluj GPU-owe "
            'zależności przez `pip install -e ".[gpu]"` i upewnij się, że '
            "sterownik CUDA jest poprawnie zainstalowany, albo pozostaw "
            "domyślne use_gpu=None (auto-detekcja) / use_gpu=False (wymuś CPU)."
        )
    return requested


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
    print("Skipping due to lack of CuPy library")
    pass


class FaissExact(BaseKNN):
    """Exact L2 nearest neighbor search using FAISS (IndexFlatL2).

    Attributes:
        use_gpu: Indicates whether the index is transferred to the GPU.
        index: The underlying FAISS index object.
    """

    def __init__(self, use_gpu: bool | None = None, metric: str = "L2"):
        """Initialize the exact FAISS index.

        Args:
            use_gpu: If True, uses the standard GPU resources for evaluation.
                If False, always uses CPU. If None (default), auto-detects:
                uses GPU when one is available, otherwise falls back to CPU
                transparently.
            metric: The distance metric ("L2" for Euclidean, "IP" for Inner Product)
        """
        self.use_gpu = use_gpu if HAS_FAISS_GPU else False
        self.metric = metric.upper()
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the FAISS FlatL2 index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        dimension = features_contig.shape[1]

        if self.metric == "IP":
            cpu_index = faiss.IndexFlatIP(dimension)
        else:
            cpu_index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_get_gpu_resources(), 0, cpu_index)
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
            _get_gpu_resources().syncDefaultStreamCurrentDevice()
        return distances[:, 1:], indices[:, 1:]


class FaissIVFFlat(BaseKNN):
    """Approximate nearest neighbor search using FAISS Inverted File index (IndexIVFFlat).

    Attributes:
        nlist: Number of Voronoi cells (clusters) used to partition the data.
        nprobe: Number of cells visited during the search phase.
        use_gpu: Indicates whether GPU resources are utilized.
        index: The underlying FAISS index object.
    """

    def __init__(
        self,
        nlist: int = 100,
        nprobe: int = 1,
        use_gpu: bool | None = None,
        metric: str = "L2",
    ):
        """Initialize the IVFFlat FAISS index.

        Args:
            nlist: Number of clusters. Defaults to 100.
            nprobe: Number of clusters to visit during query. Defaults to 1.
            use_gpu: If True, transfers index to the GPU. If False, always
                uses CPU. If None (default), auto-detects: uses GPU when one
                is available, otherwise falls back to CPU transparently.
            metric: The distance metric ("L2" or "IP")
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu if HAS_FAISS_GPU else False
        self.metric = metric.upper()
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Train the quantizer and construct the IVF index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        dimension = features_contig.shape[1]

        if self.metric == "IP":
            quantizer = faiss.IndexFlatIP(dimension)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            faiss_metric = faiss.METRIC_L2

        cpu_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss_metric)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_get_gpu_resources(), 0, cpu_index)
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
            _get_gpu_resources().syncDefaultStreamCurrentDevice()

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
        metric: Distance metric to use ("L2" or "IP").
        index: The underlying FAISS index object.
    """

    def __init__(
        self,
        nlist: int = 100,
        m: int = 5,
        nbits: int = 8,
        nprobe: int = 1,
        use_gpu: bool | None = None,
        metric: str = "L2",
    ):
        """Initialize the IVFPQ FAISS index.

        Args:
            nlist: Number of clusters. Defaults to 100.
            m: Number of sub-quantizers. Must divide the dimension space evenly. Defaults to 5.
            nbits: Bits allocated per sub-quantizer. Defaults to 8.
            nprobe: Number of clusters to probe during search. Defaults to 1.
            use_gpu: If True, uses GPU acceleration. If False, always uses
                CPU. If None (default), auto-detects: uses GPU when one is
                available, otherwise falls back to CPU transparently.
            metric: Distance metric to use ("L2" or "IP").
        """
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu if HAS_FAISS_GPU else False
        self.metric = metric.upper()
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
            raise ValueError(
                f"Wymiar przestrzeni ({dimension}) musi byÄ‡ podzielny przez m ({self.m})."
            )

        if self.metric == "IP":
            quantizer = faiss.IndexFlatIP(dimension)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            faiss_metric = faiss.METRIC_L2

        cpu_index = faiss.IndexIVFPQ(
            quantizer, dimension, self.nlist, self.m, self.nbits, faiss_metric
        )

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_get_gpu_resources(), 0, cpu_index)
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
            _get_gpu_resources().syncDefaultStreamCurrentDevice()

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

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        num_threads: int = -1,
        space: str = "l2",
    ):
        """Initialize the HNSW index.

        Args:
            m: Max links per node. Defaults to 16.
            ef_construction: Search depth during index build. Defaults to 200.
            ef: Search depth during query. Defaults to 50.
            num_threads: Number of CPU threads to utilize. Defaults to -1.
            space: The metric space to use ("l2", "ip", or "cosine"). Defaults to "l2".
        """
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.space = space.lower()
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Construct the HNSW graph index.

        Args:
            X: Feature matrix to index.
        """
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        n_samples, dimension = features_contig.shape

        self.index = hnswlib.Index(space=self.space, dim=dimension)
        self.index.init_index(
            max_elements=n_samples, ef_construction=self.ef_construction, M=self.m
        )
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

    def __init__(self, workers: int = -1, eps: float = 0.0):
        """Initialize the cKDTree wrapper.

        Args:
            workers: Number of worker threads for parallel queries. Defaults to -1.
            eps: Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1 + eps) times the
                distance to the real k-th nearest neighbor. Defaults to 0.0.
        """
        self.workers = workers
        self.eps = eps
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
        distances, indices = self.tree.query(
            X, k=k + 1, workers=self.workers, eps=self.eps
        )
        return distances[:, 1:].astype(np.float32), indices[:, 1:]


class SklearnKNN(BaseKNN):
    """Exact nearest neighbor search using scikit-learn's NearestNeighbors.

    Attributes:
        algorithm: Algorithm utilized by scikit-learn (e.g., "kd_tree" or "ball_tree").
        leaf_size: Leaf size parameter regulating tree node density.
        n_jobs: Number of parallel jobs used for querying.
        nn: The underlying scikit-learn NearestNeighbors estimator.
    """

    def __init__(
        self,
        algorithm: Literal["kd_tree", "ball_tree"] = "kd_tree",
        leaf_size: int = 100,
        n_jobs: int = -1,
    ):
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
            n_jobs=self.n_jobs,
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
