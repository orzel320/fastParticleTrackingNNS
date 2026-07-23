from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import faiss
import hnswlib
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

_GPU_RES = faiss.StandardGpuResources()


class BaseKNN(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        pass


class NumpyBruteForce(BaseKNN):
    def __init__(self, max_mem_bytes: int = 512 * 1024 * 1024):
        self.max_mem_bytes = max_mem_bytes
        self.X_train = None

    def fit(self, X: np.ndarray) -> None:
        self.X_train = X

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
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
        def __init__(self, max_vram_bytes: int = 512 * 1024 * 1024):
            self.max_vram_bytes = max_vram_bytes
            self.X_train = None

        def fit(self, X: np.ndarray) -> None:
            self.X_train = cp.asarray(X)

        def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        cpu_index = faiss.IndexFlatL2(features_contig.shape[1])
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index
        self.index.add(features_contig)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)
        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()
        return distances[:, 1:], indices[:, 1:]


class FaissIVFFlat(BaseKNN):
    def __init__(self, nlist: int = 100, nprobe: int = 1, use_gpu: bool = False):
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
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
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class FaissIVFPQ(BaseKNN):
    def __init__(self, nlist: int = 100, m: int = 5, nbits: int = 8, nprobe: int = 1, use_gpu: bool = False):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        dimension = features_contig.shape[1]

        if dimension % self.m != 0:
            raise ValueError(f"Wymiar przestrzeni ({dimension}) musi być podzielny przez m ({self.m}).")

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
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class HnswGraph(BaseKNN):
    def __init__(self, m: int = 16, ef_construction: int = 200, ef: int = 50, num_threads: int = -1):
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        n_samples, dimension = features_contig.shape

        self.index = hnswlib.Index(space="l2", dim=dimension)
        self.index.init_index(max_elements=n_samples, ef_construction=self.ef_construction, M=self.m)
        self.index.set_num_threads(self.num_threads)
        self.index.add_items(features_contig)
        self.index.set_ef(self.ef)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        features_contig = np.ascontiguousarray(X, dtype=np.float32)
        indices, distances = self.index.knn_query(features_contig, k=k + 1)

        return distances[:, 1:].astype(np.float32), indices[:, 1:]


class ScipyCKDTree(BaseKNN):
    def __init__(self, workers: int = -1):
        self.workers = workers
        self.tree = None

    def fit(self, X: np.ndarray) -> None:
        self.tree = cKDTree(X)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.tree.query(X, k=k + 1, workers=self.workers)
        return distances[:, 1:].astype(np.float32), indices[:, 1:]


class SklearnKNN(BaseKNN):
    def __init__(self, algorithm: Literal["kd_tree", "ball_tree"] = "kd_tree", leaf_size: int = 100, n_jobs: int = -1):
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.nn = None

    def fit(self, X: np.ndarray) -> None:
        self.nn = NearestNeighbors(
            n_neighbors=1, 
            algorithm=self.algorithm, 
            leaf_size=self.leaf_size, 
            n_jobs=self.n_jobs
        )
        self.nn.fit(X)

    def kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.nn.kneighbors(X, n_neighbors=k + 1)
        return distances[:, 1:].astype(np.float32), indices[:, 1:]