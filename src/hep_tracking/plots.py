import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def plot_3d_hits(features, labels, output_path=None):
    """Visualizes synthetic detector hits in a 3D scatter plot.

    :param features: Feature matrix of shape (N, 5), where the first three columns are spatial coordinates.
    :type features: numpy.ndarray
    :param labels: Array of shape (N,) containing track IDs or -1 for noise.
    :type labels: numpy.ndarray
    :param output_path: Path to save the generated plot. If None, the plot is displayed interactively.
    :type output_path: str or pathlib.Path, optional
    """
    figure = plt.figure(figsize=(7, 6))
    axes = figure.add_subplot(111, projection="3d")

    signal_mask = labels >= 0
    noise_mask = ~signal_mask

    axes.scatter(
        features[noise_mask, 0],
        features[noise_mask, 1],
        features[noise_mask, 2],
        s=3,
        c="lightgray",
        alpha=0.4,
        label="noise",
    )

    axes.scatter(
        features[signal_mask, 0],
        features[signal_mask, 1],
        features[signal_mask, 2],
        s=6,
        c=labels[signal_mask],
        cmap="tab20",
        alpha=0.9,
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("Synthetic detector hits: color = track id")
    axes.legend(loc="upper left")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(figure)


def plot_distance_distributions(features, labels, output_path=None, sample_size=2000):
    """Plots histograms of pairwise distances for same-track vs. cross-track hits.

    Sub-samples the data if it exceeds sample_size to prevent memory explosion
    during pairwise distance calculation.

    :param features: Feature matrix of shape (N, 5).
    :type features: numpy.ndarray
    :param labels: Array of shape (N,) containing track IDs or -1 for noise.
    :type labels: numpy.ndarray
    :param output_path: Path to save the generated plot. If None, the plot is displayed interactively.
    :type output_path: str or pathlib.Path, optional
    :param sample_size: Maximum number of points to sample for distance computation.
    :type sample_size: int
    """
    signal_mask = labels >= 0
    filtered_features = features[signal_mask]
    filtered_labels = labels[signal_mask]

    if len(filtered_features) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(filtered_features), sample_size, replace=False)
        filtered_features = filtered_features[indices]
        filtered_labels = filtered_labels[indices]

    distances = pdist(filtered_features, metric="euclidean")

    label_matrix_a, label_matrix_b = np.meshgrid(filtered_labels, filtered_labels)
    same_track_mask_matrix = label_matrix_a == label_matrix_b

    same_track_mask_flat = squareform(same_track_mask_matrix, checks=False)

    same_track_distances = distances[same_track_mask_flat]
    cross_track_distances = distances[~same_track_mask_flat]

    figure, axes = plt.subplots(figsize=(8, 5))

    axes.hist(
        same_track_distances, bins=50, alpha=0.6, density=True, label="Same Track"
    )
    axes.hist(
        cross_track_distances, bins=50, alpha=0.6, density=True, label="Cross Track"
    )

    axes.set_xlabel("Euclidean Distance in Feature Space")
    axes.set_ylabel("Density")
    axes.set_title("Pairwise Distance Distribution")
    axes.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(figure)

def plot_pareto_frontier(results: dict, title: str = "Wydajność algorytmów ANN: Recall vs. QPS", output_path: str = None):
    """Plots the Recall vs. QPS Pareto frontier for different ANN models.

    Args:
        results (dict): Dictionary containing plotting data. Expected format:
                        {'ModelName': {'recall': [...], 'qps': [...], 'labels': [...]}}
        title (str): Title of the plot.
        output_path (str, optional): If provided, saves the plot to this path instead of showing it.
    """
    plt.figure(figsize=(10, 7))

    colors = {"IVFFlat": "blue", "IVFPQ": "green", "HNSW": "red"}
    markers = {"IVFFlat": "o", "IVFPQ": "s", "HNSW": "^"}

    for name, data in results.items():
        c = colors.get(name, "black")
        m = markers.get(name, "x")
        
        plt.plot(
            data["recall"], data["qps"], 
            marker=m, color=c, 
            linestyle='-', linewidth=2, markersize=8, label=name
        )
        
        if "labels" in data:
            for i, label in enumerate(data["labels"]):
                if i % 2 == 0 or i == len(data["labels"]) - 1:
                    plt.annotate(
                        label, (data["recall"][i], data["qps"][i]), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9, alpha=0.7
                    )

    plt.yscale("log")
    plt.xlabel("Recall (Czułość względem dokładnego k-NN)", fontsize=12)
    plt.ylabel("QPS - Queries Per Second (Wyżej = szybciej)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.xlim(0.4, 1.05)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Zapisano wykres do: {output_path}")
    else:
        plt.show()
        
    plt.close()


def plot_crossover(sizes: list, cpu_times: list, gpu_times: list, title: str = "Crossover N: CPU vs GPU", output_path: str = None):
    """Plots query time vs dataset size to identify the CPU/GPU performance crossover point.

    Converts input times from seconds to milliseconds for better readability.

    Args:
        sizes (list): List of dataset sizes (N).
        cpu_times (list): List of CPU query times in seconds.
        gpu_times (list): List of GPU query times in seconds.
        title (str): Title of the plot.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    
    cpu_ms = [t * 1000 for t in cpu_times]
    gpu_ms = [t * 1000 for t in gpu_times]

    plt.plot(sizes, cpu_ms, marker='o', linewidth=2, color='blue', label='CPU')
    plt.plot(sizes, gpu_ms, marker='s', linewidth=2, color='red', label='GPU')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rozmiar datasetu N (Liczba hitów)", fontsize=12)
    plt.ylabel("Czas zapytania (ms)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Zapisano wykres do: {output_path}")
    else:
        plt.show()
        
    plt.close()

def plot_ann_scaling(sizes: list, results_time: dict, use_gpu: bool, title: str = "Przestrzeń 8D: Exact vs ANN", output_path: str = None):
    """Plots the final execution time comparison between exact kNN and ANN models.

    Dynamically updates the legend labels for FAISS ANN algorithms based on the 
    `use_gpu` flag to accurately reflect the hardware environment.

    Args:
        sizes (list): List of dataset sizes (N).
        results_time (dict): Dictionary mapping model names to lists of execution times.
        use_gpu (bool): Indicates if FAISS ANN models (IVFFlat/IVFPQ) utilized the GPU.
        title (str, optional): Title of the plot.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(sizes, results_time.get("Exact_CPU", []), marker='o', color='black', linestyle='-', linewidth=2, label='Exact kNN (CPU)')
    plt.plot(sizes, results_time.get("Exact_GPU", []), marker='v', color='purple', linestyle='-', linewidth=2, label='Exact kNN (GPU)')

    ivf_env = "GPU" if use_gpu else "CPU"
    plt.plot(sizes, results_time.get("IVFFlat", []), marker='s', color='blue', linestyle='--', linewidth=2, label=f'IVFFlat ({ivf_env})')
    plt.plot(sizes, results_time.get("IVFPQ", []), marker='D', color='green', linestyle='--', linewidth=2, label=f'IVFPQ ({ivf_env})')
    plt.plot(sizes, results_time.get("HNSW", []), marker='^', color='red', linestyle='--', linewidth=2, label='HNSW (CPU)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rozmiar zbioru danych N (Liczba hitów)", fontsize=12)
    plt.ylabel("Czas zapytania (sekundy)", fontsize=12)

    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
        
    plt.close()


def plot_exact_vs_ann(sizes: list, results_time: dict, title: str = "Wymiar 5D: Exact kNN vs ANN", output_path: str = None):
    """Plots the comparison between Exact kNN and ANN approaches.

    Args:
        sizes (list): List of dataset sizes (N).
        results_time (dict): Dictionary mapping model names to execution times.
        title (str): Title of the plot.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, results_time["Exact_CPU"], marker='o', color='black', linestyle='-', linewidth=2, label='FAISS Brute (CPU) - exact')
    plt.plot(sizes, results_time["Exact_GPU"], marker='v', color='purple', linestyle='-', linewidth=2, label='FAISS Brute (GPU) - exact')
    plt.plot(sizes, results_time["cKDTree_CPU"], marker='d', color='teal', linestyle='-', linewidth=2, label='Scipy cKDTree (CPU) - exact')
    plt.plot(sizes, results_time["IVFFlat_GPU"], marker='s', color='red', linestyle='--', linewidth=2, label='IVFFlat (GPU) - approx')
    plt.plot(sizes, results_time["HNSW_CPU"], marker='^', color='orange', linestyle='--', linewidth=2, label='HNSW (CPU) - approx')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rozmiar zbioru danych N (Liczba hitów)", fontsize=12)
    plt.ylabel("Czas zapytania (sekundy)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()