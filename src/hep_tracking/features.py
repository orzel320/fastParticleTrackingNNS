import numpy as np
from hep_tracking.dataset import TrackDataset


def compute_pair_features(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
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