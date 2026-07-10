import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hep_tracking.models import knn_scipy_ckdtree


def generate_and_save_candidates(data_dir="data", k_neighbors=5):
    """Runs fast exact kNN on generated datasets and saves candidate indices for classification."""
    dataset_path = Path(data_dir)

    datasets_to_process = ["dataset_hard_100k.npz", "dataset_hard_1M.npz"]

    for dataset_name in datasets_to_process:
        file_path = dataset_path / dataset_name

        if not file_path.exists():
            print(f"File {dataset_name} not found. Skipping candidate generation.")
            continue

        print(f"Processing {dataset_name} to generate candidates...")
        data = np.load(file_path)
        features = data["X"]

        distances, indices = knn_scipy_ckdtree(features, k=k_neighbors)

        output_name = dataset_path / f"candidates_{dataset_name.split('_', 1)[1]}"
        np.savez_compressed(output_name, indices=indices, distances=distances)

        print(f"  Saved candidates to: {output_name.name}")


if __name__ == "__main__":
    generate_and_save_candidates()
