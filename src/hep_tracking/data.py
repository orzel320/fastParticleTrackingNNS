import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np


def generate_tracks(
    n_tracks=8000,
    hits_per_track=15,
    n_noise=400,
    r_max=100.0,
    vertex_spread=40.0,
    sigma_pos=0.3,
    sigma_dir=0.01,
    dir_scale=60.0,
    seed=42,
):
    """Generates a synthetic dataset of particle tracks and noise hits.

    The spatial coordinates are assumed to be in arbitrary consistent units
    (e.g., millimeters). The output features consist of 3D spatial coordinates
    and 2D scaled direction cosines.

    :param n_tracks: Number of true particle tracks to simulate.
    :type n_tracks: int
    :param hits_per_track: Number of detector hits generated per track.
    :type hits_per_track: int
    :param n_noise: Number of random noise hits to scatter in the volume.
    :type n_noise: int
    :param r_max: Maximum radial distance for the outermost hit of a track.
    :type r_max: float
    :param vertex_spread: Half-width of the uniform distribution for origin vertices.
    :type vertex_spread: float
    :param sigma_pos: Standard deviation of the Gaussian noise applied to hit positions.
    :type sigma_pos: float
    :param sigma_dir: Standard deviation of the Gaussian noise applied to direction features.
    :type sigma_dir: float
    :param dir_scale: Scaling factor applied to the direction cosines.
    :type dir_scale: float
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: A tuple containing the feature matrix of shape (N, 5) and the labels array of shape (N,).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)

    vertices = rng.uniform(-vertex_spread, vertex_spread, (n_tracks, 3))
    thetas = rng.uniform(0.3, np.pi - 0.3, n_tracks)
    phis = rng.uniform(-np.pi, np.pi, n_tracks)

    directions = np.column_stack(
        [np.sin(thetas) * np.cos(phis), np.sin(thetas) * np.sin(phis), np.cos(thetas)]
    )

    step_sizes = np.arange(1, hits_per_track + 1) * r_max / hits_per_track
    step_sizes = step_sizes.reshape(1, hits_per_track, 1)

    positions = vertices[:, None, :] + step_sizes * directions[:, None, :]
    positions += rng.normal(0, sigma_pos, positions.shape)

    dx_pred = (
        directions[:, 0:1] + rng.normal(0, sigma_dir, (n_tracks, hits_per_track))
    ) * dir_scale
    dy_pred = (
        directions[:, 1:2] + rng.normal(0, sigma_dir, (n_tracks, hits_per_track))
    ) * dir_scale

    track_features = np.concatenate(
        [positions, dx_pred[..., None], dy_pred[..., None]], axis=2
    )
    track_features = track_features.reshape(-1, 5)

    track_ids = np.repeat(np.arange(n_tracks), hits_per_track)

    if n_noise > 0:
        noise_pos = rng.uniform(-150, 150, (n_noise, 3))
        noise_ct = rng.uniform(-1, 1, n_noise)
        noise_ph = rng.uniform(-np.pi, np.pi, n_noise)
        noise_st = np.sqrt(1 - noise_ct**2)

        noise_dx = noise_st * np.cos(noise_ph) * dir_scale
        noise_dy = noise_st * np.sin(noise_ph) * dir_scale

        noise_features = np.column_stack([noise_pos, noise_dx, noise_dy])
        noise_ids = np.full(n_noise, -1)

        features_array = np.vstack([track_features, noise_features]).astype(np.float32)
        labels_array = np.concatenate([track_ids, noise_ids])
    else:
        features_array = track_features.astype(np.float32)
        labels_array = track_ids

    permutation = rng.permutation(len(features_array))

    return features_array[permutation], labels_array[permutation]


def generate_datasets(output_dir="data"):
    """Generates and saves datasets for benchmark scaling experiments.

    Creates `.npz` files for 'easy' and 'hard' configurations across
    target sizes of approximately 1k, 10k, 100k, and 1M hits, split into events (max 100k hits per event)

    :param output_dir: Destination directory for the generated datasets.
    :type output_dir: str
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_sizes = {"1k": 1_000, "10k": 10_000, "100k": 100_000, "1M": 1_000_000}
    max_hits_per_event = 100_000

    generation_modes = {
        "easy": {
            "hits_per_track": 10,
            "noise_ratio": 0.01,
            "sigma_pos": 0.05,
            "sigma_dir": 0.005,
            "vertex_spread": 80.0,
        },
        "hard": {
            "hits_per_track": 15,
            "noise_ratio": 0.20,
            "sigma_pos": 0.5,
            "sigma_dir": 0.02,
            "vertex_spread": 20.0,
        },
    }

    for mode_name, parameters in generation_modes.items():
        print(f"--- Generating mode: {mode_name.upper()} ---")
        for size_label, total_hits in target_sizes.items():

            # Ustalanie liczby zdarzeń (np. 1M / 100k = 10 zdarzeń)
            n_events = max(1, total_hits // max_hits_per_event)
            hits_per_event = total_hits // n_events

            n_noise_hits = int(hits_per_event * parameters["noise_ratio"])
            n_tracks_per_event = (hits_per_event - n_noise_hits) // parameters[
                "hits_per_track"
            ]

            all_features = []
            all_labels = []
            all_event_ids = []
            global_track_offset = 0

            for event_idx in range(n_events):
                features, labels = generate_tracks(
                    n_tracks=n_tracks_per_event,
                    hits_per_track=parameters["hits_per_track"],
                    n_noise=n_noise_hits,
                    sigma_pos=parameters["sigma_pos"],
                    sigma_dir=parameters["sigma_dir"],
                    vertex_spread=parameters["vertex_spread"],
                )

                signal_mask = labels != -1
                labels[signal_mask] += global_track_offset
                global_track_offset += n_tracks_per_event

                event_ids = np.full(len(labels), event_idx, dtype=np.int32)

                all_features.append(features)
                all_labels.append(labels)
                all_event_ids.append(event_ids)

            final_features = np.vstack(all_features)
            final_labels = np.concatenate(all_labels)
            final_event_ids = np.concatenate(all_event_ids)

            filename = output_path / f"dataset_{mode_name}_{size_label}.npz"
            np.savez_compressed(
                filename, X=final_features, y=final_labels, event_id=final_event_ids
            )
            print(
                f"Saved: {filename.name} (Hits: {final_features.shape[0]}, Tracks: {global_track_offset}, Events: {n_events})"
            )


if __name__ == "__main__":
    generate_datasets()
