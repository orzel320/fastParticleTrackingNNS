import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import matplotlib.pyplot as plt

from hep_tracking.models import (
    knn_numpy_brute_force,
    knn_cupy_brute_force,
    knn_scipy_ckdtree,
    knn_sklearn_kdtree,
    knn_sklearn_balltree,
)
from hep_tracking.plots import plot_3d_hits, plot_distance_distributions


def measure_execution_time(target_function, num_runs=3, warmup_runs=1):
    for _ in range(warmup_runs):
        target_function()

    execution_times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        target_function()
        execution_times.append(time.perf_counter() - start_time)

    return min(execution_times)


def generate_sanity_plots(dataset_path: Path):
    dataset_file = dataset_path / "dataset_hard_10k.npz"
    
    if not dataset_file.exists():
        print(f"Sanity dataset {dataset_file.name} not found. Skipping plots.")
        return

    loaded_data = np.load(dataset_file)
    features = loaded_data["X"]
    labels = loaded_data["y"]

    scatter_output = "scatter_3d.pdf"
    plot_3d_hits(features, labels, output_path=scatter_output)
    print(f" Saved 3D scatter plot to {scatter_output}")

    hist_output = "distances_hist.pdf"
    plot_distance_distributions(features, labels, output_path=hist_output)
    print(f" Saved distance histograms to {hist_output}")


def run_scaling_benchmark(data_dir="data", output_plot="scaling.pdf"):
    dataset_path = Path(data_dir)
    
    generate_sanity_plots(dataset_path)

    target_sizes = {"1k": 1000, "10k": 10000, "100k": 100000, "1M": 1000000}
    dataset_modes = ["easy", "hard"]
    k_neighbors = 5

    methods = {
        "Numpy Brute Force": knn_numpy_brute_force,
        "CuPy Brute Force": knn_cupy_brute_force,
        "Scipy cKDTree": knn_scipy_ckdtree,
        "Sklearn KDTree (Opt)": knn_sklearn_kdtree,
        "Sklearn BallTree (Opt)": knn_sklearn_balltree,
    }

    results = {
        mode: {method: [] for method in methods} for mode in dataset_modes
    }

    for mode in dataset_modes:
        print(f"\n--- Benchmarking mode: {mode.upper()} ---")
        for size_label, size_val in target_sizes.items():
            filename = dataset_path / f"dataset_{mode}_{size_label}.npz"
            
            if not filename.exists():
                print(f"Dataset {filename} not found. Skipping.")
                for method_name in methods:
                    results[mode][method_name].append(None)
                continue

            loaded_data = np.load(filename)
            features = loaded_data["X"]

            print(f" Loaded {size_label} ({features.shape[0]} hits)")

            for method_name, method_callable in methods.items():
                if (method_name == "Numpy Brute Force" or method_name == "CuPy Brute Force") and size_val > 100000:
                    print(f"Skipping {method_name} for {size_label} (too slow)")
                    results[mode][method_name].append(None)
                    continue
                def benchmark_wrapper():
                    method_callable(features, k_neighbors)

                min_time = measure_execution_time(benchmark_wrapper)
                results[mode][method_name].append(min_time)
                print(f"  {method_name}: {min_time * 1000:.2f} ms")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    numeric_sizes = list(target_sizes.values())

    for idx, mode in enumerate(dataset_modes):
        ax = axes[idx]
        for method_name, method_times in results[mode].items():
            valid_sizes = [
                s for s, t in zip(numeric_sizes, method_times) if t is not None
            ]
            valid_times = [t for t in method_times if t is not None]
            
            ax.plot(
                valid_sizes,
                valid_times,
                marker="o",
                label=method_name,
                linewidth=2,
                markersize=6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Scaling for mode: {mode.title()}")
        ax.set_xlabel("Number of hits (N)")
        if idx == 0:
            ax.set_ylabel("Minimum Wall-clock Time (s)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close(figure)
    print(f"\nScaling plot saved as '{output_plot}'")


if __name__ == "__main__":
    run_scaling_benchmark()