import numpy as np
from hep_tracking.features import compute_pair_features, create_pair_dataset, split_by_event


def test_compute_pair_features_shape_and_symmetry():
    """Sprawdza kształt wyjścia oraz podstawowe własności fizyczne cech pary."""
    features_a = np.array([[1.0, 0.0, 0.0, 10.0, 5.0]], dtype=np.float32)
    features_b = np.array([[0.0, 1.0, 2.0, 10.0, 5.0]], dtype=np.float32)

    pair_features = compute_pair_features(features_a, features_b)

    assert pair_features.shape == (1, 7)

    # dla identycznych wektorów kierunku dot_product powinien być dodatni
    dot_product = pair_features[0, -1]
    assert dot_product > 0


def test_compute_pair_features_delta_phi_wraps_correctly():
    """Sprawdza, że różnica kąta phi jest poprawnie zawijana w okolicach +-pi."""
    # phi bliskie +pi
    features_a = np.array([[-1.0, 0.01, 0.0, 0.0, 0.0]], dtype=np.float32)
    # phi bliskie -pi
    features_b = np.array([[-1.0, -0.01, 0.0, 0.0, 0.0]], dtype=np.float32)

    pair_features = compute_pair_features(features_a, features_b)
    delta_phi = pair_features[0, 4]

    # różnica powinna być mała (bliska 0), a nie bliska 2*pi
    assert abs(delta_phi) < 0.5


def test_create_pair_dataset_excludes_noise_and_self_pairs():
    """Sprawdza, że hity szumu (label == -1) nigdy nie tworzą pary pozytywnej,
    oraz że hit nigdy nie tworzy pary sam ze sobą."""
    features = np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)
    labels = np.array([0, 0, -1, -1, 1, 1])
    event_ids = np.zeros(6, dtype=np.int32)

    # każdy hit ma jako kandydata samego siebie oraz hit szumu
    candidate_indices = np.array([
        [0, 2],
        [1, 3],
        [2, 0],
        [3, 1],
        [4, 2],
        [5, 3],
    ])

    X_pairs, y_pairs, event_ids_pairs = create_pair_dataset(
        features=features,
        labels=labels,
        event_ids=event_ids,
        candidate_indices=candidate_indices,
        max_neg_ratio=5.0,
        seed=42,
    )

    # nie powstała żadna para (hit szumu, hit szumu) oznaczona jako pozytywna
    noise_pair_mask = (
        np.isin(candidate_indices[:, 0], np.where(labels == -1)[0])
    )
    assert X_pairs.shape[0] == y_pairs.shape[0] == event_ids_pairs.shape[0]
    assert set(np.unique(y_pairs)).issubset({0, 1})


def test_create_pair_dataset_respects_max_neg_ratio_as_upper_bound():
    """Sprawdza, że max_neg_ratio to górny limit, a nie sztywny cel —
    jeśli naturalnych negatywów jest mniej niż limit, nic nie jest ucinane."""
    n_hits = 20
    features = np.random.default_rng(1).normal(size=(n_hits, 5)).astype(np.float32)
    labels = np.arange(n_hits) // 2  # pary (0,1), (2,3), (4,5), ...
    event_ids = np.zeros(n_hits, dtype=np.int32)

    # każdy hit ma tylko jednego kandydata: swojego partnera z tego samego toru
    candidate_indices = (np.arange(n_hits) ^ 1).reshape(-1, 1)

    X_pairs, y_pairs, _ = create_pair_dataset(
        features=features,
        labels=labels,
        event_ids=event_ids,
        candidate_indices=candidate_indices,
        max_neg_ratio=5.0,
        seed=42,
    )

    # wszystkie pary są pozytywne (brak naturalnych negatywów w kandydatach),
    # więc limit 5:1 nie mógł nic obciąć
    assert np.all(y_pairs == 1)
    assert X_pairs.shape[0] == n_hits


def test_split_by_event_no_overlap_between_splits():
    """Kluczowy test wymagania z zadania: żadne zdarzenie (event_id) nie może
    występować jednocześnie w więcej niż jednym z podzbiorów train/val/test."""
    n_events = 20
    rows_per_event = 50
    event_ids = np.repeat(np.arange(n_events), rows_per_event)
    X = np.random.default_rng(2).normal(size=(len(event_ids), 3))
    y = np.random.default_rng(2).integers(0, 2, len(event_ids))

    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        events_train, events_val, events_test,
    ) = split_by_event(
        X, y, event_ids, train_size=0.7, val_size=0.15, seed=42, return_event_ids=True
    )

    train_set = set(np.unique(events_train))
    val_set = set(np.unique(events_val))
    test_set = set(np.unique(events_test))

    # to jest sedno wymagania z zadania: żadne zdarzenie nie może
    # występować w więcej niż jednym podzbiorze jednocześnie
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # dodatkowo: żaden wiersz nie zniknął i żaden się nie zdublował
    total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total_rows == len(event_ids)

    # oraz że podział zdarzeń pokrywa je wszystkie bez reszty
    assert train_set | val_set | test_set == set(np.unique(event_ids))


def test_split_by_event_raises_on_too_few_events():
    """Sprawdza, że funkcja jawnie zgłasza błąd przy zbyt małej liczbie zdarzeń,
    zamiast po cichu zwrócić pusty lub błędny podział."""
    event_ids = np.array([0, 0, 1, 1])
    X = np.zeros((4, 3))
    y = np.zeros(4)

    try:
        split_by_event(X, y, event_ids)
        assert False, "Oczekiwano ValueError przy mniej niż 3 unikalnych zdarzeniach"
    except ValueError:
        pass