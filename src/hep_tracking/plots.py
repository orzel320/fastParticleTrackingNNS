"""Visualization utilities for tracking datasets, model performance, and metrics."""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hep_tracking.dataset import TrackDataset


def plot_3d_hits(dataset: TrackDataset, output_path: str = None):
    """Plot a 3D scatter visualization of synthetic detector hits.

    Distinguishes visually between valid signal tracks (colored by track ID) 
    and background noise (rendered as semi-transparent light gray points).

    Args:
        dataset: The dataset containing spatial hit features and track labels.
        output_path: Optional file path to save the generated figure. If None, 
            the plot is displayed interactively via `plt.show()`.
    """
    figure = plt.figure(figsize=(7, 6))
    axes = figure.add_subplot(111, projection="3d")

    features = dataset.X
    labels = dataset.y

    signal_mask = labels >= 0
    noise_mask = ~signal_mask

    axes.scatter(
        features[noise_mask, 0], features[noise_mask, 1], features[noise_mask, 2],
        s=3, c="lightgray", alpha=0.4, label="noise"
    )

    axes.scatter(
        features[signal_mask, 0], features[signal_mask, 1], features[signal_mask, 2],
        s=6, c=labels[signal_mask], cmap="tab20", alpha=0.9
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("Synthetic detector hits")
    axes.legend(loc="upper left")

    plt.tight_layout()

    if output_path:
                plt.savefig(output_path, bbox_inches="tight")
            
    plt.show()
    plt.close()


def plot_distance_distributions(dataset: TrackDataset, output_path: str = None, sample_size: int = 2000):
    """Plot the pairwise Euclidean distance distributions between hits.

    Generates overlapping histograms comparing the distances between hits belonging 
    to the same track versus hits belonging to different tracks. To ensure memory 
    efficiency, the distance matrix is computed on a random subset of the data.

    Args:
        dataset: The dataset containing hit features and track labels.
        output_path: Optional file path to save the generated figure.
        sample_size: Maximum number of signal hits to sample for the pairwise 
            distance calculation. Defaults to 2000.
    """
    features = dataset.X
    labels = dataset.y

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

    axes.hist(same_track_distances, bins=50, alpha=0.6, density=True, label="Same Track")
    axes.hist(cross_track_distances, bins=50, alpha=0.6, density=True, label="Cross Track")

    axes.set_xlabel("Euclidean Distance in Feature Space")
    axes.set_ylabel("Density")
    axes.set_title("Pairwise Distance Distribution")
    axes.legend()

    plt.tight_layout()

    if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        
    plt.show()
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, title: str = "Pareto Frontier", output_path: str = None):
    """Plot a Pareto frontier comparing model Recall against Queries Per Second (QPS).

    Visualizes the trade-off between search quality (Recall) and search speed (QPS) 
    across different model configurations. QPS is displayed on a logarithmic scale.

    Args:
        df: A pandas DataFrame containing at least 'Model', 'Recall', 'QPS', 
            and 'Dataset' columns.
        title: The title of the plot. Defaults to "Pareto Frontier".
        output_path: Optional file path to save the generated figure.
    """
    plt.figure(figsize=(10, 7))

    markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*'])
    colors = itertools.cycle(plt.cm.tab10.colors)

    for model, group in df.groupby("Model"):
        group = group.sort_values("Recall")
        plt.plot(
            group["Recall"], group["QPS"], 
            marker=next(markers), color=next(colors), 
            linestyle='-', linewidth=2, markersize=8, label=model
        )
        
        for i, (_, row) in enumerate(group.iterrows()):
            if i % 2 == 0 or i == len(group) - 1:
                plt.annotate(
                    row.get("Dataset", ""), (row["Recall"], row["QPS"]), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=9, alpha=0.7
                )

    plt.yscale("log")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("QPS", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.xlim(max(0.0, df["Recall"].min() - 0.1), min(1.05, df["Recall"].max() + 0.05))
    plt.tight_layout()

    if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        
    plt.show()
    plt.close()


def plot_scaling(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str = "Scaling Performance", 
    log_x: bool = True, 
    log_y: bool = True, 
    output_path: str = None
):
    """Plot scaling performance metrics across models.

    Args:
        df: DataFrame containing the metrics to plot, grouped by the 'Model' column.
        x_col: The column name to use for the X-axis.
        y_col: The column name to use for the Y-axis.
        title: The title of the plot. Defaults to "Scaling Performance".
        log_x: If True, applies a logarithmic scale to the X-axis. Defaults to True.
        log_y: If True, applies a logarithmic scale to the Y-axis. Defaults to True.
        output_path: Optional file path to save the generated figure.
    """
    plt.figure(figsize=(10, 6))

    markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*'])
    colors = itertools.cycle(plt.cm.tab10.colors)

    for model, group in df.groupby("Model"):
        group = group.sort_values(x_col)
        plt.plot(
            group[x_col], group[y_col], 
            marker=next(markers), color=next(colors), 
            linestyle='-', linewidth=2, label=model
        )

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
        
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_roc_curves(models_dict: dict, test_dataset: TrackDataset, output_path: str = None):
    """Plot Receiver Operating Characteristic (ROC) curves for multiple models.

    Args:
        models_dict: Dictionary mapping model names to trained classifier objects 
            that implement a `predict_proba` method.
        test_dataset: The dataset used to evaluate the ROC curves.
        output_path: Optional file path to save the generated figure.
    """
    figure, axes = plt.subplots(figsize=(8, 6))

    X_test = test_dataset.X
    y_test = test_dataset.y

    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

    axes.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("ROC Curves Comparison")
    axes.legend(loc="lower right")
    axes.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_pr_curves(models_dict: dict, test_dataset: TrackDataset, output_path: str = None):
    """Plot Precision-Recall (PR) curves for multiple classification models.

    Also plots a baseline indicating the proportion of positive samples in the dataset.

    Args:
        models_dict: Dictionary mapping model names to trained classifier objects 
            that implement a `predict_proba` method.
        test_dataset: The dataset used to evaluate the PR curves.
        output_path: Optional file path to save the generated figure.
    """
    figure, axes = plt.subplots(figsize=(8, 6))

    X_test = test_dataset.X
    y_test = test_dataset.y

    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        axes.plot(recall, precision, lw=2, label=f"{model_name} (PR AUC = {pr_auc:.4f})")

    baseline = np.sum(y_test) / len(y_test)
    axes.axhline(baseline, color="black", lw=1, linestyle="--", label=f"Baseline ({baseline:.2f})")
    
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title("Precision-Recall Curves Comparison")
    axes.legend(loc="lower left")
    axes.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_metric_dimension_heatmap(
    df: pd.DataFrame, 
    metric: str, 
    title: str, 
    cmap: str = "viridis", 
    log_scale: bool = False, 
    output_path: str = None
):
    """Plot a heatmap of a specific metric across different dataset dimensions and sizes.

    Creates subplots for each model found in the DataFrame. The rows correspond 
    to dataset dimensions ('Dimension' column) and the columns correspond to 
    dataset sizes ('Size' column).

    Args:
        df: DataFrame containing at least 'Model', 'Dimension', 'Size', and the 
            target metric column.
        metric: The column name of the metric to plot (e.g., 'QPS' or 'Recall').
        title: The overarching title for the entire figure.
        cmap: The colormap to use for the heatmap. Defaults to "viridis".
        log_scale: If True, applies logarithmic color scaling via LogNorm. Defaults to False.
        output_path: Optional file path to save the generated figure.
    """
    models = df["Model"].unique()
    n_algos = len(models)

    fig, axes = plt.subplots(1, n_algos, figsize=(5.5 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    vmin, vmax = df[metric].min(), df[metric].max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    im = None
    for ax, model in zip(axes, models):
        model_df = df[df["Model"] == model]
        pivot = model_df.pivot_table(index="Dimension", columns="Size", values=metric, aggfunc='mean')
        
        im = ax.imshow(
            pivot.values, aspect="auto", cmap=cmap, 
            norm=norm, 
            vmin=vmin if not log_scale else None, 
            vmax=vmax if not log_scale else None
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{s:,}" for s in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{d}D" for d in pivot.index])
        ax.set_xlabel("Size")
        ax.set_title(model, fontsize=12)

        threshold = (vmin * vmax) ** 0.5 if log_scale else (vmin + vmax) / 2
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if (val < threshold and log_scale) or (val > threshold and not log_scale and cmap != "RdYlGn") or (val < threshold and cmap == "RdYlGn") else "black"
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center", color=color, fontsize=8)

    axes[0].set_ylabel("Dimension")
    fig.colorbar(im, ax=axes, label=metric, fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=14)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_metric_lines_by_dimension(
    df: pd.DataFrame, 
    metric: str, 
    title: str, 
    log_y: bool = False, 
    output_path: str = None
):
    """Plot a line chart of a metric across dataset sizes, grouped into subplots by dimension.

    Args:
        df: DataFrame containing at least 'Model', 'Dimension', 'Size', and the 
            target metric column.
        metric: The column name of the metric to plot on the Y-axis.
        title: The overarching title for the entire figure.
        log_y: If True, applies a logarithmic scale to the Y-axis. Defaults to False.
        output_path: Optional file path to save the generated figure.
    """
    dimensions = sorted(df["Dimension"].unique())
    models = df["Model"].unique()
    
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.5 * len(dimensions), 5), sharey=True)
    if len(dimensions) == 1:
        axes = [axes]

    markers = itertools.cycle(['o', 's', '^', 'D', 'v'])
    colors = itertools.cycle(plt.cm.tab10.colors)
    model_style = {m: (next(colors), next(markers)) for m in models}

    for ax, d in zip(axes, dimensions):
        dim_df = df[df["Dimension"] == d]
        
        for model in models:
            model_df = dim_df[dim_df["Model"] == model].sort_values("Size")
            if not model_df.empty:
                c, m = model_style[model]
                ax.plot(
                    model_df["Size"], model_df[metric], 
                    marker=m, color=c, linestyle='--', linewidth=2, label=model
                )

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
            
        ax.set_xlabel("Size")
        ax.set_title(f"{d}D", fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    axes[0].set_ylabel(metric)
    axes[-1].legend(fontsize=10, loc="best")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight") # DPI nie jest wymagane dla PDF, bo to wektory
    
    plt.show()
    plt.close()

def plot_silver_bullet(df: pd.DataFrame, output_path: str = None):
    """Plot the Silver-Bullet: Physics Quality vs Total Time per Event.
    
    Generates scatter plots separated by Dataset, where faint points represent 
    individual events and large crosses represent the mean performance of each pipeline.

    Args:
        df: DataFrame containing per-event pipeline evaluation results.
            Must include 'Time_Total_s', 'Physics_Quality', 'Pipeline', and 'Dataset'.
        output_path: Optional file path to save the generated figure.
    """
    summary_df = df.groupby(["Dataset", "Pipeline"]).agg(
        Physics_Quality_Mean=("Physics_Quality", "mean"),
        Time_Total_Mean=("Time_Total_s", "mean")
    ).reset_index()

    g = sns.relplot(
        data=summary_df,
        x="Time_Total_Mean",
        y="Physics_Quality_Mean",
        hue="Pipeline",
        col="Dataset",
        s=250,
        marker="X",
        edgecolor="black",
        linewidth=1.5,
        kind="scatter",
        height=6,
        aspect=1.2,
        facet_kws={'sharey': False, 'sharex': False}
    )

    for dataset_name, ax in g.axes_dict.items():
        dataset_df = df[df["Dataset"] == dataset_name]
        
        sns.scatterplot(
            data=dataset_df,
            x="Time_Total_s",
            y="Physics_Quality",
            hue="Pipeline",
            ax=ax,
            alpha=0.15,
            s=40,
            legend=False
        )
        
        ax.set_xscale("log")
        
        y_min = dataset_df["Physics_Quality"].min()
        y_max = dataset_df["Physics_Quality"].max()
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
        
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
        ax.grid(True, which="both", ls="--", alpha=0.5)

    g.set_axis_labels("Total Time per Event (seconds)", "Physics Quality (Purity x Efficiency)")
    g.set_titles(col_template="Dataset: {col_name}")
    g.figure.suptitle("The Silver-Bullet Plot: Physics Quality vs Total Time", y=1.05, fontsize=16, fontweight="bold")
    
    sns.move_legend(g, "center left", bbox_to_anchor=(1.02, 0.5), title="Pipeline (X = Mean)")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight") # DPI nie jest wymagane dla PDF, bo to wektory
    
    plt.show()
    plt.close()

def plot_time_vs_size(df: pd.DataFrame, output_path: str = None):
    """Plot the execution time scalability against the number of hits per event.

    Args:
        df: DataFrame containing evaluation results with 'Hits', 'Time_Total_s', and 'Pipeline'.
        output_path: Optional file path to save the generated figure.
    """
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, 
        x="Hits", 
        y="Time_Total_s", 
        hue="Pipeline", 
        marker="o",
        errorbar=None, 
        linewidth=2
    )

    plt.title("Scalability: Execution Time vs Event Size", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Hits in Event", fontsize=12)
    plt.ylabel("Total Processing Time (seconds)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Pipeline", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight") # DPI nie jest wymagane dla PDF, bo to wektory
    
    plt.show()
    plt.close()


def plot_time_breakdown(df: pd.DataFrame, output_path: str = None):
    """Plot a stacked bar chart showing the breakdown of time spent in each pipeline stage.

    Args:
        df: DataFrame containing execution times for Retrieval, Features, and Filter stages.
        output_path: Optional file path to save the generated figure.
    """
    summary = df.groupby("Pipeline")[
        ["Time_Retrieval_s", "Time_Features_s", "Time_Filter_s"]
    ].mean().sort_values(by="Time_Retrieval_s")

    ax = summary.plot(
        kind="bar", 
        stacked=True, 
        figsize=(12, 7), 
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black"
    )

    plt.title("Bottleneck Analysis: Average Time Breakdown per Pipeline", fontsize=14, fontweight="bold")
    plt.xlabel("Pipeline Configuration", fontsize=12)
    plt.ylabel("Average Time per Event (seconds)", fontsize=12)
    
    plt.legend(
        ["Retrieval (kNN/ANN)", "Feature Engineering", "Filtering (ML/Cuts)"], 
        title="Pipeline Stage",
        bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_purity_vs_efficiency(df: pd.DataFrame, output_path: str = None):
    """Plot the trade-off between Purity (Precision) and Efficiency (Recall).

    Args:
        df: DataFrame containing 'Efficiency', 'Purity', and 'Pipeline'.
        output_path: Optional file path to save the generated figure.
    """
    summary_df = df.groupby("Pipeline").agg(
        Purity_Mean=("Purity", "mean"),
        Efficiency_Mean=("Efficiency", "mean")
    ).reset_index()

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=summary_df, 
        x="Efficiency_Mean", 
        y="Purity_Mean", 
        hue="Pipeline", 
        s=150,
        alpha=0.6,
        edgecolor="black", 
        linewidth=1.5,
        marker="D"
    )

    plt.title("Physics Quality Trade-off: Purity vs Efficiency", fontsize=14, fontweight="bold")
    plt.xlabel("Efficiency (Recall)", fontsize=12)
    plt.ylabel("Purity (Precision)", fontsize=12)
    
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Pipeline (Mean values)", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    plt.close()