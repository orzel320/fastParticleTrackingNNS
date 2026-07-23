"""Dataset abstraction layer for high-energy physics tracking data."""

from pathlib import Path
import numpy as np


class TrackDataset:
    """Encapsulates hit features, labels, and event mappings for particle tracking.

    Attributes:
        X: Array of shape (N, D) containing hit features (e.g., position, direction).
        y: Array of shape (N,) containing track labels (-1 for noise).
        event_ids: Array of shape (N,) assigning each hit to an event index.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, event_ids: np.ndarray):
        """Initialize a TrackDataset instance.

        Args:
            X: Feature matrix representing hit attributes.
            y: Target track identification labels.
            event_ids: Event identifiers for each hit.
        """
        self.X = X
        self.y = y
        self.event_ids = event_ids

    @classmethod
    def from_npz(cls, file_path: str | Path) -> "TrackDataset":
        """Load a dataset from a compressed `.npz` file.

        Expects the `.npz` archive to contain the keys "X", "y", and "event_id".

        Args:
            file_path: Path to the input `.npz` file.

        Returns:
            An instantiated `TrackDataset` initialized with the loaded arrays.
        """
        data = np.load(file_path)
        return cls(
            X=data["X"],
            y=data["y"],
            event_ids=data["event_id"]
        )

    def get_spatial_features(self) -> np.ndarray:
        """Extract spatial coordinates (x, y, z) from the feature matrix.

        Returns:
            A slice of shape (N, 3) containing only spatial positions.
        """
        return self.X[:, :3]

    def get_padded_features(self, target_dim: int = 8) -> np.ndarray:
        """Pad the feature matrix with zeros to match a target dimension.

        If the current feature dimension is already greater than or equal to 
        `target_dim`, the feature matrix is returned as a contiguous array.

        Args:
            target_dim: Desired number of feature columns. Defaults to 8.

        Returns:
            A contiguous float32 array padded with zeros up to `target_dim` columns.
        """
        current_dim = self.X.shape[1]
        
        if current_dim >= target_dim:
            return np.ascontiguousarray(self.X, dtype=np.float32)
            
        pad_width = target_dim - current_dim
        padded_features = np.pad(self.X, ((0, 0), (0, pad_width)), mode='constant')
        
        return np.ascontiguousarray(padded_features, dtype=np.float32)
        
    def filter_by_event(self, target_event_ids: np.ndarray) -> "TrackDataset":
        """Create a subset dataset containing hits matching target event IDs.

        Args:
            target_event_ids: Array or collection of event IDs to retain.

        Returns:
            A new `TrackDataset` instance containing only the selected events.
        """
        mask = np.isin(self.event_ids, target_event_ids)
        return TrackDataset(
            X=self.X[mask],
            y=self.y[mask],
            event_ids=self.event_ids[mask]
        )

    def __len__(self) -> int:
        """Return the total number of hits in the dataset.

        Returns:
            The number of rows in the feature array `X`.
        """
        return len(self.X)