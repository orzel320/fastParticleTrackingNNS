from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hep_tracking.models import ScipyCKDTree 

SPATIAL_COLUMNS = slice(0, 3)

def generate_and_save_candidates(data_dir: str = "data", k_neighbors: int = 5, spatial_only: bool = False):
    dataset_path = Path(data_dir)

    for file_path in dataset_path.glob("dataset_*.npz"):
        print(f"Przetwarzanie {file_path.name} w celu wygenerowania kandydatów...")
        data = np.load(file_path)
        features = data["X"]
        event_ids = data["event_id"]

        knn_features = features[:, SPATIAL_COLUMNS] if spatial_only else features

        all_indices = np.zeros((features.shape[0], k_neighbors), dtype=np.int64)
        all_distances = np.zeros((features.shape[0], k_neighbors), dtype=np.float32)

        unique_events = np.unique(event_ids)
        exact_knn = ScipyCKDTree(workers=-1)
        
        for ev_id in unique_events:
            mask = (event_ids == ev_id)
            ev_features = knn_features[mask]

            exact_knn.fit(ev_features)
            distances, indices = exact_knn.kneighbors(ev_features, k=k_neighbors)

            global_idx = np.where(mask)[0]
            global_neighbors = global_idx[indices]

            all_indices[mask] = global_neighbors
            all_distances[mask] = distances

        suffix = "_spatial" if spatial_only else ""
        dataset_base_name = file_path.stem.replace("dataset_", "")
        output_name = dataset_path / f"candidates_{dataset_base_name}{suffix}.npz"
        
        np.savez_compressed(output_name, indices=all_indices, distances=all_distances)
        print(f"  Zapisano kandydatów do: {output_name.name}")