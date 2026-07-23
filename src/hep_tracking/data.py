from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hep_tracking.config import TrackSimulationConfig, DatasetConfig

def generate_tracks(
    n_tracks: int,
    n_noise: int,
    config: TrackSimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)

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
                config=config.simulation_params
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