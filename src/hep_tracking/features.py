import numpy as np


def compute_pair_features(features_a, features_b):
    """Oblicza fizyczne i geometryczne cechy dla par hitów.

    UWAGA METODOLOGICZNA: cecha ``dot_product`` wykorzystuje kolumny 3 i 4
    wejściowego wektora cech (kierunek toru, ``dx_pred``/``dy_pred``) —
    dokładnie te same kolumny, na których liczony jest kNN w
    ``generate_candidates.py``. Ponieważ kandydaci do klasyfikacji są już
    wybierani na podstawie odległości w przestrzeni zawierającej ten sam
    sygnał kierunkowy, ``dot_product`` może być silniej skorelowany z
    prawdziwą etykietą niż wynikałoby to z samej fizyki problemu. To
    prawdopodobne wytłumaczenie, dlaczego RandomForest osiąga wynik 1.0000
    na zbiorze testowym — nie jest to klasyczny wyciek między train/test,
    tylko wyciek przez sposób doboru kandydatów w poprzednim sprincie.

    :param features_a: Macierz cech pierwszych hitów w parach, kształt (M, 5).
    :type features_a: numpy.ndarray
    :param features_b: Macierz cech drugich hitów w parach, kształt (M, 5).
    :type features_b: numpy.ndarray
    :return: Macierz wygenerowanych cech pary, kształt (M, 7).
    :rtype: numpy.ndarray
    """
    delta_xyz = features_a[:, :3] - features_b[:, :3]

    r_a = np.hypot(features_a[:, 0], features_a[:, 1])
    r_b = np.hypot(features_b[:, 0], features_b[:, 1])
    delta_r = (r_a - r_b)[:, np.newaxis]

    phi_a = np.arctan2(features_a[:, 1], features_a[:, 0])
    phi_b = np.arctan2(features_b[:, 1], features_b[:, 0])
    delta_phi_raw = phi_a - phi_b
    # normalizacja różnicy kąta do zakresu (-pi, pi], żeby uniknąć
    # nieciągłości przy przejściu przez -pi/pi
    delta_phi = np.arctan2(np.sin(delta_phi_raw), np.cos(delta_phi_raw))[:, np.newaxis]

    dist_3d = np.linalg.norm(delta_xyz, axis=1)[:, np.newaxis]

    # UWAGA: to jest surowy iloczyn skalarny, nie znormalizowane
    # podobieństwo kosinusowe — miesza ze sobą "zgodność kierunku"
    # i "skalę wektora kierunku". Jeśli w przyszłości potrzebna będzie
    # czysta miara kąta między kierunkami, warto znormalizować wektory
    # (podzielić przez ich normy) przed obliczeniem iloczynu.
    dot_product = (features_a[:, 3] * features_b[:, 3] + features_a[:, 4] * features_b[:, 4])[:, np.newaxis]

    return np.hstack([delta_xyz, delta_r, delta_phi, dist_3d, dot_product]).astype(np.float32)


def create_pair_dataset(features, labels, event_ids, candidate_indices, max_neg_ratio=5.0, seed=42):
    """Generuje zbiór danych do klasyfikacji binarnej na podstawie par kandydatów.

    :param features: Macierz cech hitów, kształt (N, 5).
    :type features: numpy.ndarray
    :param labels: Tablica prawdziwych identyfikatorów torów (truth track ID) dla każdego hitu.
    :type labels: numpy.ndarray
    :param event_ids: Tablica identyfikatorów zdarzeń (event ID) dla każdego hitu.
    :type event_ids: numpy.ndarray
    :param candidate_indices: Macierz indeksów sąsiadów-kandydatów, kształt (N, k).
    :type candidate_indices: numpy.ndarray
    :param max_neg_ratio: Maksymalny stosunek liczby par negatywnych do pozytywnych
        (górny limit — jeśli naturalny stosunek jest niższy, nic nie jest obcinane).
    :type max_neg_ratio: float
    :param seed: Ziarno losowości używane przy downsamplingu negatywów.
    :type seed: int
    :return: Krotka zawierająca cechy par, etykiety par i identyfikatory zdarzeń par.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    n_queries, k_neighbors = candidate_indices.shape

    queries = np.repeat(np.arange(n_queries), k_neighbors)
    candidates = candidate_indices.flatten()

    # usuwamy pary hitu z samym sobą (może się zdarzyć, jeśli punkt
    # znajdzie sam siebie jako "sąsiada")
    valid_mask = queries != candidates
    queries = queries[valid_mask]
    candidates = candidates[valid_mask]

    labels_q = labels[queries]
    labels_c = labels[candidates]

    # hity szumu (label == -1) nigdy nie są traktowane jako para pozytywna,
    # nawet jeśli oba mają label == -1
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


def split_by_event(X, y, event_ids, train_size=0.75, val_size=0.15, seed=42, return_event_ids=False):
    """Dzieli zbiór danych na train/walidacja/test z zachowaniem granic zdarzeń
    (żaden event_id nie występuje jednocześnie w więcej niż jednym podzbiorze).

    :param X: Macierz cech.
    :type X: numpy.ndarray
    :param y: Tablica etykiet.
    :type y: numpy.ndarray
    :param event_ids: Tablica identyfikatorów zdarzeń odpowiadających wierszom X.
    :type event_ids: numpy.ndarray
    :param train_size: Udział zdarzeń trafiających do zbioru treningowego.
    :type train_size: float
    :param val_size: Udział zdarzeń trafiających do zbioru walidacyjnego.
    :type val_size: float
    :param seed: Ziarno losowości używane przy tasowaniu zdarzeń.
    :type seed: int
    :param return_event_ids: Jeśli True, funkcja dodatkowo zwraca event_id
        odpowiadające każdemu z trzech podzbiorów — używane głównie do testów
        jednostkowych, które muszą jawnie zweryfikować brak przecieku zdarzeń
        między train/val/test bez zgadywania na podstawie wartości X.
    :type return_event_ids: bool
    :return: Krotka (X_train, y_train, X_val, y_val, X_test, y_test), a jeśli
        ``return_event_ids=True`` — dodatkowo (event_ids_train, event_ids_val, event_ids_test).
    :rtype: tuple
    """
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

    result = (X[train_mask], y[train_mask], X[val_mask], y[val_mask], X[test_mask], y[test_mask])

    if return_event_ids:
        result = result + (event_ids[train_mask], event_ids[val_mask], event_ids[test_mask])

    return result