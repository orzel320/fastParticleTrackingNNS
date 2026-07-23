from pathlib import Path
import numpy as np

class TrackDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, event_ids: np.ndarray):
        self.X = X
        self.y = y
        self.event_ids = event_ids

    @classmethod
    def from_npz(cls, file_path: str | Path) -> "TrackDataset":
        data = np.load(file_path)
        return cls(
            X=data["X"],
            y=data["y"],
            event_ids=data["event_id"]
        )

    def get_spatial_features(self) -> np.ndarray:
        return self.X[:, :3]

    def get_padded_features(self, target_dim: int = 8) -> np.ndarray:
        current_dim = self.X.shape[1]
        
        if current_dim >= target_dim:
            return np.ascontiguousarray(self.X, dtype=np.float32)
            
        pad_width = target_dim - current_dim
        padded_features = np.pad(self.X, ((0, 0), (0, pad_width)), mode='constant')
        
        return np.ascontiguousarray(padded_features, dtype=np.float32)
        
    def filter_by_event(self, target_event_ids: np.ndarray) -> "TrackDataset":
        mask = np.isin(self.event_ids, target_event_ids)
        return TrackDataset(
            X=self.X[mask],
            y=self.y[mask],
            event_ids=self.event_ids[mask]
        )

    def __len__(self) -> int:
        return len(self.X)