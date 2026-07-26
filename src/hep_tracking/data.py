"""Synthetic data generation for high-energy physics tracking datasets.

This module provides functions to simulate particle trajectories, generate 
background noise, and package these into complete datasets saved as 
compressed numpy arrays.
"""

from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hep_tracking.config import TrackSimulationConfig, DatasetConfig

def generate_tracks(
    n_tracks: int,
    n_noise: int,
    config: TrackSimulationConfig,
    seed_offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic particle tracks and noise hits for a single event.

    Simulates linear particle trajectories originating near the origin (vertex) 
    and propagating outwards. Adds Gaussian noise to the positions and directional 
    vectors, and uniformly distributes random noise hits throughout the volume.
    The returned arrays are randomly permuted so the sequence of hits does not 
    trivially reveal the underlying tracks.

    Args:
        n_tracks: The number of distinct particle tracks to simulate.
        n_noise: The total number of uncorrelated noise hits to inject.
        config: The simulation parameters controlling geometry and variance.
        seed_offset: An integer added to the base configuration seed to ensure 
            uncorrelated randomness across different events. Defaults to 0.

    Returns:
        A tuple containing two arrays:
            - features: A (N, 5) array of hit features representing position and direction.
            - labels: A (N,) array of track IDs, where -1 indicates a noise hit.
    """
    rng = np.random.default_rng(config.seed + seed_offset)

    vertices = rng.uniform(-config.vertex_spread, config.vertex_spread, (n_tracks, 3))
    thetas = rng.uniform(0.3, np.pi - 0.3, n_tracks)
    phis = rng.uniform(-np.pi, np.pi, n_tracks)

    directions = np.column_stack([
        np.sin(thetas) * np.cos(phis), 
        np.sin(thetas) * np.sin(phis), 
        np.cos(thetas)
    ])

    step_sizes = np.arange(1, config.hits_per_track + 1) * config.r_max / config.hits_per_track
    step_sizes = step_sizes.reshape(1, config.hits_per_track, 1)

    positions = vertices[:, None, :] + step_sizes * directions[:, None, :]
    positions += rng.normal(0, config.sigma_pos, positions.shape)

    dx_pred = (directions[:, 0:1] + rng.normal(0, config.sigma_dir, (n_tracks, config.hits_per_track))) * config.dir_scale
    dy_pred = (directions[:, 1:2] + rng.normal(0, config.sigma_dir, (n_tracks, config.hits_per_track))) * config.dir_scale

    track_features = np.concatenate([positions, dx_pred[..., None], dy_pred[..., None]], axis=2)
    track_features = track_features.reshape(-1, 5)

    track_ids = np.repeat(np.arange(n_tracks), config.hits_per_track)

    if n_noise > 0:
        noise_pos = rng.uniform(-150, 150, (n_noise, 3))
        noise_ct = rng.uniform(-1, 1, n_noise)
        noise_ph = rng.uniform(-np.pi, np.pi, n_noise)
        noise_st = np.sqrt(1 - noise_ct**2)

        noise_dx = noise_st * np.cos(noise_ph) * config.dir_scale
        noise_dy = noise_st * np.sin(noise_ph) * config.dir_scale

        noise_features = np.column_stack([noise_pos, noise_dx, noise_dy])
        noise_ids = np.full(n_noise, -1)

        features_array = np.vstack([track_features, noise_features]).astype(np.float32)
        labels_array = np.concatenate([track_ids, noise_ids])
    else:
        features_array = track_features.astype(np.float32)
        labels_array = track_ids

    permutation = rng.permutation(len(features_array))

    return features_array[permutation], labels_array[permutation]

def generate_datasets(configs: list[DatasetConfig], output_dir: str = "data") -> None:
    """Generate and serialize multiple synthetic datasets based on configurations.

    Iterates through the provided dataset configurations, calculating the required 
    number of events and hits per event. It aggregates the generated features, 
    labels, and event IDs, then saves them as compressed `.npz` archives to the 
    specified output directory. Track IDs are offset globally across events to 
    ensure uniqueness within a single dataset.

    Args:
        configs: A list of configurations detailing the size and simulation 
            parameters for each target dataset.
        output_dir: The target directory path where the `.npz` files will be saved. 
            The directory is created if it does not exist. Defaults to "data".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for config in configs:
        n_events = max(1, config.target_hits // config.max_hits_per_event)
        hits_per_event = config.target_hits // n_events

        n_noise_hits = int(hits_per_event * config.simulation_params.noise_ratio)
        n_tracks_per_event = (hits_per_event - n_noise_hits) // config.simulation_params.hits_per_track

        all_features = []
        all_labels = []
        all_event_ids = []
        global_track_offset = 0

        for event_idx in range(n_events):
            features, labels = generate_tracks(
                n_tracks=n_tracks_per_event,
                n_noise=n_noise_hits,
                config=config.simulation_params,
                seed_offset=event_idx
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

        filename = output_path / f"dataset_{config.name}.npz"
        np.savez_compressed(filename, X=final_features, y=final_labels, event_id=final_event_ids)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_output_dir = str(project_root / "data")

    print(f"Katalog wyjściowy: {data_output_dir}")
    print("Przygotowywanie konfiguracji...")

    easy_sim = TrackSimulationConfig(
        hits_per_track=10, 
        noise_ratio=0.01, 
        sigma_pos=0.05, 
        sigma_dir=0.005, 
        vertex_spread=80.0
    )
    
    hard_sim = TrackSimulationConfig(
        hits_per_track=15, 
        noise_ratio=0.20, 
        sigma_pos=0.5, 
        sigma_dir=0.02, 
        vertex_spread=20.0
    )

    target_sizes = {"1k": 1_000, "10k": 10_000, "100k": 100_000, "1M": 1_000_000}
    configs = []

    for size_label, size_val in target_sizes.items():
        configs.append(DatasetConfig(f"easy_{size_label}", size_val, easy_sim))
        configs.append(DatasetConfig(f"hard_{size_label}", size_val, hard_sim))

    print("Rozpoczynam generowanie zbiorów danych...")
    generate_datasets(configs, output_dir=data_output_dir)
    print("Generowanie zbiorów zakończone sukcesem!")