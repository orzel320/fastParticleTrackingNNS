import numpy as np


def compute_pair_features(features_a, features_b):
    """Computes physical and geometric features for pairs of hits.

    :param features_a: Feature matrix of the first hits in the pairs, shape (M, 5).
    :type features_a: numpy.ndarray
    :param features_b: Feature matrix of the second hits in the pairs, shape (M, 5).
    :type features_b: numpy.ndarray
    :return: Matrix of engineered pair features, shape (M, 8).
    :rtype: numpy.ndarray
    """
    delta_xyz = features_a[:, :3] - features_b[:, :3]

    r_a = np.hypot(features_a[:, 0], features_a[:, 1])
    r_b = np.hypot(features_b[:, 0], features_b[:, 1])
    delta_r = (r_a - r_b)[:, np.newaxis]

    phi_a = np.arctan2(features_a[:, 1], features_a[:, 0])
    phi_b = np.arctan2(features_b[:, 1], features_b[:, 0])
    delta_phi_raw = phi_a - phi_b
    delta_phi = np.arctan2(np.sin(delta_phi_raw), np.cos(delta_phi_raw))[:, np.newaxis]

    dist_3d = np.linalg.norm(delta_xyz, axis=1)[:, np.newaxis]

    dot_product = (features_a[:, 3] * features_b[:, 3] + features_a[:, 4] * features_b[:, 4])[:, np.newaxis]

    return np.hstack([delta_xyz, delta_r, delta_phi, dist_3d, dot_product]).astype(np.float32)


def create_pair_dataset(features, labels, event_ids, candidate_indices, max_neg_ratio=5.0, seed=42):
    """Generates a binary classification dataset from candidate pairs.

    :param features: Matrix of hit features, shape (N, 5).
    :type features: numpy.ndarray
    :param labels: Array of true track IDs for each hit, shape (N,).
    :type labels: numpy.ndarray
    :param event_ids: Array of event IDs for each hit, shape (N,).
    :type event_ids: numpy.ndarray
    :param candidate_indices: Matrix of candidate neighbor indices, shape (N, k).
    :type candidate_indices: numpy.ndarray
    :param max_neg_ratio: Maximum ratio of negative to positive pairs.
    :type max_neg_ratio: float
    :param seed: Random seed for negative downsampling.
    :type seed: int
    :return: Tuple containing pair features, pair labels, and pair event IDs.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    n_queries, k_neighbors = candidate_indices.shape

    queries = np.repeat(np.arange(n_queries), k_neighbors)
    candidates = candidate_indices.flatten()

    valid_mask = queries != candidates
    queries = queries[valid_mask]
    candidates = candidates[valid_mask]

    labels_q = labels[queries]
    labels_c = labels[candidates]

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

    pair_features = compute_pair_features(features[final_queries], features[final_candidates])
    pair_labels = positive_mask[selected_indices].astype(np.int32)
    pair_event_ids = event_ids[final_queries]

    return pair_features, pair_labels, pair_event_ids


def split_by_event(X, y, event_ids, train_size=0.75, val_size=0.15, seed=42):
    """Splits the dataset into training, validation, and test sets based on event boundaries.

    :param X: Feature matrix of the data.
    :type X: numpy.ndarray
    :param y: Labels array.
    :type y: numpy.ndarray
    :param event_ids: Array of event IDs corresponding to each row in X.
    :type event_ids: numpy.ndarray
    :param train_size: Proportion of the dataset to include in the train split.
    :type train_size: float
    :param val_size: Proportion of the dataset to include in the validation split.
    :type val_size: float
    :param seed: Random seed for shuffling events.
    :type seed: int
    :return: A tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    :rtype: tuple
    """
    unique_events = np.unique(event_ids)

    unique_events = np.unique(event_ids)
    
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

    train_mask = np.isin(event_ids, train_events)
    val_mask = np.isin(event_ids, val_events)
    test_mask = np.isin(event_ids, test_events)

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask], X[test_mask], y[test_mask]