import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree, BallTree
import cupy as cp
from sklearn.neighbors import NearestNeighbors


def knn_numpy_brute_force(
    features, k, chunk_size=2048, max_mem_bytes=512 * 1024 * 1024
):
    """Computes exact k-nearest neighbors using a brute-force numpy approach.

    This function calculates pairwise Euclidean distances using vectorization and matrix
    multiplication.

    :param features: Feature matrix of shape (N, D).
    :type features: numpy.ndarray
    :param k: Number of nearest neighbors to find (excluding the point itself).
    :type k: int
    :param max_mem_bytes: Maximum memory allowance in bytes for the chunk computation.
                          Defaults to 512 MB.
    :type max_mem_bytes: int, optional
    :return: A tuple of (distances, indices) arrays, each of shape (N, k).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    n_samples = features.shape[0]
    squared_norms = np.sum(features * features, axis=1)

    nearest_indices = np.empty((n_samples, k), dtype=np.int64)
    nearest_distances = np.empty((n_samples, k), dtype=np.float32)

    bytes_per_row = n_samples * 12 * 3
    chunk_size = max(1, max_mem_bytes // bytes_per_row)

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)

        distances = (
            squared_norms[start_idx:end_idx, None]
            + squared_norms[None, :]
            - 2.0 * (features[start_idx:end_idx] @ features.T)
        )
        distances = np.maximum(distances, 0.0)

        rows = np.arange(start_idx, end_idx)
        distances[rows - start_idx, rows] = np.inf

        partitioned_indices = np.argpartition(distances, k, axis=1)[:, :k]

        for i in range(end_idx - start_idx):
            sorted_order = np.argsort(distances[i, partitioned_indices[i]])
            nearest_indices[start_idx + i] = partitioned_indices[i, sorted_order]
            nearest_distances[start_idx + i] = np.sqrt(
                distances[i, partitioned_indices[i, sorted_order]]
            )

    return nearest_distances, nearest_indices


def knn_cupy_brute_force(
    features, k, chunk_size=2048, max_vram_bytes=512 * 1024 * 1024
):
    """Computes exact k-nearest neighbors using CuPy.

    This function accelerates the brute-force pairwise distance calculation using GPU
    matrix operations.

    :param features: Feature matrix of shape (N, D).
    :type features: numpy.ndarray
    :param k: Number of nearest neighbors to find (excluding the point itself).
    :type k: int
    :param max_vram_bytes: Maximum VRAM allowance in bytes for the chunk computation.
                           Defaults to 512 MB.
    :type max_vram_bytes: int, optional
    :return: A tuple of (distances, indices) arrays, each of shape (N, k).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    features_gpu = cp.asarray(features)
    n_samples = features_gpu.shape[0]

    squared_norms = cp.sum(features_gpu * features_gpu, axis=1)

    nearest_indices = np.empty((n_samples, k), dtype=np.int64)
    nearest_distances = np.empty((n_samples, k), dtype=np.float32)

    bytes_per_row = n_samples * 12 * 3
    chunk_size = max(1, max_vram_bytes // bytes_per_row)

    mempool = cp.get_default_memory_pool()

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)

        distances = (
            squared_norms[start_idx:end_idx, None]
            + squared_norms[None, :]
            - 2.0 * (features_gpu[start_idx:end_idx] @ features_gpu.T)
        )
        distances = cp.maximum(distances, 0.0)

        rows = cp.arange(start_idx, end_idx)
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


def knn_scipy_ckdtree(features, k):
    """Computes exact k-nearest neighbors using scipy's cKDTree.

    Builds an axis-aligned KD-tree and queries it in parallel across all CPU cores.
    The query searches for k+1 neighbors and discards the first one (self).

    :param features: Feature matrix of shape (N, D).
    :type features: numpy.ndarray
    :param k: Number of nearest neighbors to find.
    :type k: int
    :return: A tuple of (distances, indices) arrays, each of shape (N, k).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    tree = cKDTree(features)
    distances, indices = tree.query(features, k=k + 1, workers=-1)

    return distances[:, 1:].astype(np.float32), indices[:, 1:]


def knn_sklearn_kdtree(features, k, leaf_size=100):
    """Computes exact k-nearest neighbors using scikit-learn's KDTree via NearestNeighbors.

    Wrapped in the NearestNeighbors estimator to enable joblib-based multiprocessing.
    Space is partitioned using axis-aligned hyperplanes.

    :param features: Feature matrix of shape (N, D).
    :type features: numpy.ndarray
    :param k: Number of nearest neighbors to find.
    :type k: int
    :param leaf_size: Number of points at which to switch to brute-force.
    :type leaf_size: int, optional
    :return: A tuple of (distances, indices) arrays, each of shape (N, k).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    nn = NearestNeighbors(
        n_neighbors=k + 1, algorithm="kd_tree", leaf_size=leaf_size, n_jobs=-1
    )
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    return distances[:, 1:].astype(np.float32), indices[:, 1:]


def knn_sklearn_balltree(features, k, leaf_size=100):
    """Computes exact k-nearest neighbors using scikit-learn's BallTree via NearestNeighbors.

    Wrapped in the NearestNeighbors estimator to enable joblib-based multiprocessing.
    Space is partitioned using nested hyperspheres.

    :param features: Feature matrix of shape (N, D).
    :type features: numpy.ndarray
    :param k: Number of nearest neighbors to find.
    :type k: int
    :param leaf_size: Number of points at which to switch to brute-force.
    :type leaf_size: int, optional
    :return: A tuple of (distances, indices) arrays, each of shape (N, k).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    nn = NearestNeighbors(
        n_neighbors=k + 1, algorithm="ball_tree", leaf_size=leaf_size, n_jobs=-1
    )
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    return distances[:, 1:].astype(np.float32), indices[:, 1:]
