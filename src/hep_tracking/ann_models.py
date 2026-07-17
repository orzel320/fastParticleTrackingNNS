import numpy as np
import faiss
import hnswlib

_GPU_RES = faiss.StandardGpuResources()


class FaissIVFFlat:
    """Approximate Nearest Neighbors using FAISS Inverted File index.

    Partitions the vector space using k-means clustering. Searches are limited
    to the partitions closest to the query vector.

    Attributes:
        nlist (int): Number of space partitions (Voronoi cells).
        nprobe (int): Number of neighboring partitions to search during a query.
        use_gpu (bool): Whether to perform operations on the GPU.
        index (faiss.Index): The underlying FAISS index object.
    """

    def __init__(self, nlist: int = 100, nprobe: int = 1, use_gpu: bool = False):
        """Initializes the FAISS IVFFlat wrapper.

        Args:
            nlist: Number of space partitions.
            nprobe: Number of partitions to search during a query.
            use_gpu: Whether to perform operations on the GPU.
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def build(self, features: np.ndarray) -> None:
        """Trains the index and adds the dataset vectors.

        Args:
            features: Feature matrix of shape (N, D). It will be cast to a
                      C-contiguous float32 array as required by FAISS.
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)
        dimension = features_contig.shape[1]

        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, dimension, self.nlist, faiss.METRIC_L2
        )

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.train(features_contig)
        self.index.add(features_contig)
        self.index.nprobe = self.nprobe

    def query(self, features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries the k nearest neighbors for the given features.

        Args:
            features: Feature matrix of shape (N, D) to query.
            k: Number of nearest neighbors to find (excluding the point itself).

        Returns:
            A tuple of (distances, indices) arrays, each of shape (N, k).
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class FaissIVFPQ:
    """Approximate Nearest Neighbors using FAISS with Product Quantization.

    Combines inverted file partitioning with product quantization to heavily
    compress the memory footprint of the vectors.

    Attributes:
        nlist (int): Number of space partitions.
        m (int): Number of subquantizers. Must strictly divide the feature dimension.
        nbits (int): Number of bits per subquantizer index (usually 8).
        nprobe (int): Number of partitions to search during a query.
        use_gpu (bool): Whether to perform operations on the GPU.
        index (faiss.Index): The underlying FAISS index object.
    """

    def __init__(
        self,
        nlist: int = 100,
        m: int = 5,
        nbits: int = 8,
        nprobe: int = 1,
        use_gpu: bool = False,
    ):
        """Initializes the FAISS IVFPQ wrapper.

        Args:
            nlist: Number of space partitions.
            m: Number of subquantizers. Must strictly divide the feature dimension.
            nbits: Number of bits per subquantizer index (usually 8).
            nprobe: Number of partitions to search during a query.
            use_gpu: Whether to perform operations on the GPU.
        """
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None

    def build(self, features: np.ndarray) -> None:
        """Trains the index and adds the dataset vectors.

        Args:
            features: Feature matrix of shape (N, D). It will be cast to a
                      C-contiguous float32 array as required by FAISS.

        Raises:
            ValueError: If the feature dimension is not divisible by the number
                        of subquantizers (m).
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)
        dimension = features_contig.shape[1]

        if dimension % self.m != 0:
            raise ValueError(
                f"Wymiar przestrzeni ({dimension}) musi być podzielny przez m ({self.m})."
            )

        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFPQ(
            quantizer, dimension, self.nlist, self.m, self.nbits
        )

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(_GPU_RES, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.train(features_contig)
        self.index.add(features_contig)
        self.index.nprobe = self.nprobe

    def query(self, features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries the k nearest neighbors for the given features.

        Args:
            features: Feature matrix of shape (N, D) to query.
            k: Number of nearest neighbors to find (excluding the point itself).

        Returns:
            A tuple of (distances, indices) arrays, each of shape (N, k).
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)
        distances, indices = self.index.search(features_contig, k + 1)

        if self.use_gpu:
            _GPU_RES.syncDefaultStreamCurrentDevice()

        return distances[:, 1:], indices[:, 1:]


class HnswGraph:
    """Approximate Nearest Neighbors using Hierarchical Navigable Small World graphs.

    Constructs a multi-layer graph where greedy routing is used to find
    the closest neighbors.

    Attributes:
        m (int): Maximum number of outgoing connections in the graph per element.
        ef_construction (int): Size of the dynamic list for nearest neighbors during index construction.
        ef (int): Size of the dynamic list for nearest neighbors during the search phase.
        num_threads (int): Number of threads to use for index construction and queries.
        index (hnswlib.Index): The underlying HNSW index object.
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        num_threads: int = -1,
    ):
        """Initializes the HNSW graph wrapper.

        Args:
            m: Maximum number of outgoing connections in the graph per element.
            ef_construction: Size of the dynamic list for nearest neighbors during index construction.
            ef: Size of the dynamic list for nearest neighbors during the search phase.
            num_threads: Number of threads to use. Defaults to -1 (all available cores).
        """
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.index = None

    def build(self, features: np.ndarray) -> None:
        """Trains the index and adds the dataset vectors.

        Args:
            features: Feature matrix of shape (N, D). It will be cast to a
                      C-contiguous float32 array.
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)
        n_samples, dimension = features_contig.shape

        self.index = hnswlib.Index(space="l2", dim=dimension)
        self.index.init_index(
            max_elements=n_samples, ef_construction=self.ef_construction, M=self.m
        )
        self.index.set_num_threads(self.num_threads)
        self.index.add_items(features_contig)
        self.index.set_ef(self.ef)

    def query(self, features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries the k nearest neighbors for the given features.

        Args:
            features: Feature matrix of shape (N, D) to query.
            k: Number of nearest neighbors to find (excluding the point itself).

        Returns:
            A tuple of (distances, indices) arrays, each of shape (N, k).
        """
        features_contig = np.ascontiguousarray(features, dtype=np.float32)

        indices, distances = self.index.knn_query(features_contig, k=k + 1)

        return distances[:, 1:].astype(np.float32), indices[:, 1:]