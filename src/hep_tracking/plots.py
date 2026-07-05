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