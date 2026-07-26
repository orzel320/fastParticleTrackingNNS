"""Feature engineering and dataset manipulation for hit pairs."""

import numpy as np
from hep_tracking.dataset import TrackDataset


def compute_pair_features(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Compute relative geometric features between pairs of track hits.

    Calculates a 7-dimensional feature vector for each pair, encompassing 
    spatial differences, cylindrical radius differences, angular relationships, 
    and vector dot products.

    Args:
        features_a: Feature matrix of the query hits.
        features_b: Feature matrix of the candidate neighbor hits.

    Returns:
        A (N, 7) float32 array containing the computed pairwise features.
    """
    n_pairs = features_a.shape[0]
    
    out = np.empty((n_pairs, 7), dtype=np.float32)

    out[:, :3] = features_a[:, :3] - features_b[:, :3]

    r_a = np.hypot(features_a[:, 0], features_a[:, 1])
    r_b = np.hypot(features_b[:, 0], features_b[:, 1])
    out[:, 3] = r_a - r_b

    cross_prod = features_a[:, 1] * features_b[:, 0] - features_a[:, 0] * features_b[:, 1]
    dot_prod_xy = features_a[:, 0] * features_b[:, 0] + features_a[:, 1] * features_b[:, 1]
    out[:, 4] = np.arctan2(cross_prod, dot_prod_xy)

    out[:, 5] = np.sqrt(np.einsum('ij,ij->i', out[:, :3], out[:, :3]))

    out[:, 6] = features_a[:, 3] * features_b[:, 3] + features_a[:, 4] * features_b[:, 4]

    return out


def create_pair_dataset(
    dataset: TrackDataset, 
    candidate_indices: np.ndarray, 
    max_neg_ratio: float = 5.0, 
    seed: int = 42
) -> TrackDataset:
    """Construct a dataset of hit pairs for binary classification.

    Transforms nearest-neighbor candidate indices into a binary classification dataset. 
    Pairs belonging to the same track are labeled as positive (1), while mismatched 
    hits or noise hits are labeled as negative (0). To address class imbalance, 
    the negative class is randomly downsampled.

    Args:
        dataset: The source track dataset containing original features and labels.
        candidate_indices: An array of shape (N, K) representing the nearest 
            neighbor indices for each hit.
        max_neg_ratio: Maximum allowable ratio of negative to positive pairs. 
            Defaults to 5.0.
        seed: Random seed used for shuffling and downsampling. Defaults to 42.

    Returns:
        A new `TrackDataset` object where `X` contains the pairwise features and 
        `y` contains binary classification labels.
    """
    n_queries, k_neighbors = candidate_indices.shape

    queries = np.repeat(np.arange(n_queries), k_neighbors)
    candidates = candidate_indices.flatten()

    valid_mask = queries != candidates
    queries = queries[valid_mask]
    candidates = candidates[valid_mask]

    labels_q = dataset.y[queries]
    labels_c = dataset.y[candidates]

    positive_mask = (labels_q == labels_c) & (labels_q != -1)
    negative_mask = ~positive_mask

    pos_indices = np.where(positive_mask)[0]
    neg_indices = np.where(negative_mask)[0]

    n_pos = len(pos_indices)
    max_neg = int(n_pos * max_neg_ratio)

    rng = np.random.default_rng(seed)
    if len(neg_indices) > max_neg:
        neg_indices = rng.choice(neg_indices, max_neg, replace=False)

    selected_indices = np.concatenate([pos_indices, neg_indices])
    rng.shuffle(selected_indices)

    final_queries = queries[selected_indices]
    final_candidates = candidates[selected_indices]

    pair_features = compute_pair_features(dataset.X[final_queries], dataset.X[final_candidates])
    pair_labels = positive_mask[selected_indices].astype(np.int32)
    pair_event_ids = dataset.event_ids[final_queries]

    return TrackDataset(X=pair_features, y=pair_labels, event_ids=pair_event_ids)


def split_by_event(
    dataset: TrackDataset, 
    train_size: float = 0.75, 
    val_size: float = 0.15, 
    seed: int = 42
) -> tuple[TrackDataset, TrackDataset, TrackDataset]:
    """Split a dataset into training, validation, and test sets by event.

    Groups data strictly by `event_ids` to ensure there is no data leakage 
    across splits. Hits belonging to the same event will always remain 
    together in the same dataset partition.

    Args:
        dataset: The source dataset to be split.
        train_size: The proportion of unique events allocated to the training set. 
            Defaults to 0.75.
        val_size: The proportion of unique events allocated to the validation set. 
            Defaults to 0.15.
        seed: Random seed used to shuffle the unique events prior to splitting. 
            Defaults to 42.

    Returns:
        A tuple containing three `TrackDataset` instances: the training set, 
        the validation set, and the test set.

    Raises:
        ValueError: If the dataset contains fewer than 3 unique events, making a 
            3-way split impossible.
    """
    unique_events = np.unique(dataset.event_ids)

    if len(unique_events) < 3:
        raise ValueError(f"Zbyt mało zdarzeń ({len(unique_events)}) do poprawnego podziału na Train/Val/Test.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_events)

    n_events = len(unique_events)
    train_end = int(n_events * train_size)
    val_end = train_end + int(n_events * val_size)

    train_events = unique_events[:train_end]
    val_events = unique_events[train_end:val_end]
    test_events = unique_events[val_end:]

    train_mask = np.isin(dataset.event_ids, train_events)
    val_mask = np.isin(dataset.event_ids, val_events)
    test_mask = np.isin(dataset.event_ids, test_events)

    train_dataset = TrackDataset(X=dataset.X[train_mask], y=dataset.y[train_mask], event_ids=dataset.event_ids[train_mask])
    val_dataset = TrackDataset(X=dataset.X[val_mask], y=dataset.y[val_mask], event_ids=dataset.event_ids[val_mask])
    test_dataset = TrackDataset(X=dataset.X[test_mask], y=dataset.y[test_mask], event_ids=dataset.event_ids[test_mask])

    return train_dataset, val_dataset, test_dataset