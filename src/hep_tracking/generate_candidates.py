import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hep_tracking.models import knn_scipy_ckdtree

# Kolumny 0,1,2 to współrzędne przestrzenne (x, y, z).
# Kolumny 3,4 to składowe przewidywanego kierunku (dx_pred, dy_pred) —
# to one są podejrzewane o wyciek sygnału truth do doboru kandydatów kNN.
SPATIAL_COLUMNS = slice(0, 3)


def generate_and_save_candidates(data_dir="data", k_neighbors=5, spatial_only=False):
    """Uruchamia dokładny (exact) kNN na wygenerowanych zbiorach danych
    i zapisuje indeksy kandydatów do dalszej klasyfikacji.

    UWAGA METODOLOGICZNA: domyślnie (``spatial_only=False``, zachowane dla
    zgodności wstecznej z już wygenerowanymi plikami candidates_*.npz) kNN
    liczony jest na WSZYSTKICH 5 kolumnach cech, czyli także na składowych
    kierunku (dx_pred, dy_pred). Ponieważ kierunek hitów z tego samego toru
    jest niemal identyczny (bardzo mały szum), a hitów z różnych torów —
    losowy, dobór kandydatów w tej przestrzeni już częściowo "podpowiada"
    prawdziwą etykietę. To prawdopodobne źródło podejrzanie idealnych
    wyników klasyfikatora (RandomForest = 1.0000 na zbiorze testowym).

    Ustawienie ``spatial_only=True`` liczy kNN wyłącznie na współrzędnych
    przestrzennych (x, y, z) — to uczciwsza wersja eksperymentu, bliższa
    duchowi zadania ("sample negatives from the kNN output... confusable
    negatives" powinno oznaczać confusable geometrycznie, nie kierunkowo).
    Warto wygenerować kandydatów w obu wariantach i porównać wyniki
    klasyfikacji — jeśli RandomForest przestanie być idealny przy
    ``spatial_only=True``, to potwierdzi powyższą hipotezę.

    :param data_dir: Katalog z danymi wejściowymi/wyjściowymi.
    :type data_dir: str
    :param k_neighbors: Liczba najbliższych sąsiadów do znalezienia.
    :type k_neighbors: int
    :param spatial_only: Jeśli True, kNN liczony jest tylko na kolumnach x, y, z.
    :type spatial_only: bool
    """
    dataset_path = Path(data_dir)

    datasets_to_process = ["dataset_hard_100k.npz", "dataset_hard_1M.npz"]

    for dataset_name in datasets_to_process:
        file_path = dataset_path / dataset_name

        if not file_path.exists():
            print(f"Plik {dataset_name} nie znaleziony. Pomijam generowanie kandydatów.")
            continue

        print(f"Przetwarzanie {dataset_name} w celu wygenerowania kandydatów...")
        data = np.load(file_path)
        features = data["X"]
        event_ids = data["event_id"]

        knn_features = features[:, SPATIAL_COLUMNS] if spatial_only else features

        all_indices = np.zeros((features.shape[0], k_neighbors), dtype=np.int64)
        all_distances = np.zeros((features.shape[0], k_neighbors), dtype=np.float32)

        unique_events = np.unique(event_ids)
        for ev_id in unique_events:
            mask = (event_ids == ev_id)
            ev_features = knn_features[mask]

            distances, indices = knn_scipy_ckdtree(ev_features, k=k_neighbors)

            global_idx = np.where(mask)[0]
            global_neighbors = global_idx[indices]

            all_indices[mask] = global_neighbors
            all_distances[mask] = distances

        suffix = "_spatial" if spatial_only else ""
        output_name = dataset_path / f"candidates_{dataset_name.split('_', 1)[1].replace('.npz', '')}{suffix}.npz"
        np.savez_compressed(output_name, indices=all_indices, distances=all_distances)

        print(f"  Zapisano kandydatów do: {output_name.name}")


if __name__ == "__main__":
    generate_and_save_candidates()
    # generuje też uczciwszą wersję do porównania:
    generate_and_save_candidates(spatial_only=True)