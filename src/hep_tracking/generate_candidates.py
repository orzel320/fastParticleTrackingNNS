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
        event_ids = data["event_id"]
        
        all_indices = np.zeros((features.shape[0], k_neighbors), dtype=np.int64)
        all_distances = np.zeros((features.shape[0], k_neighbors), dtype=np.float32)
        
        unique_events = np.unique(event_ids)
        for ev_id in unique_events:
            mask = (event_ids == ev_id)
            ev_features = features[mask]
            
            distances, indices = knn_scipy_ckdtree(ev_features, k=k_neighbors)
            
            global_idx = np.where(mask)[0]
            global_neighbors = global_idx[indices]
            
            all_indices[mask] = global_neighbors
            all_distances[mask] = distances

        output_name = dataset_path / f"candidates_{dataset_name.split('_', 1)[1]}"
        np.savez_compressed(output_name, indices=all_indices, distances=all_distances)

        print(f"  Saved candidates to: {output_name.name}")


if __name__ == "__main__":
    generate_and_save_candidates()
