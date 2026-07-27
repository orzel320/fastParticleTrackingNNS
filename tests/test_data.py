import numpy as np

from hep_tracking.config import TrackSimulationConfig
from hep_tracking.data import generate_tracks


def _default_config(seed: int = 42) -> TrackSimulationConfig:
    return TrackSimulationConfig(
        hits_per_track=5,
        noise_ratio=0.1,
        sigma_pos=0.5,
        sigma_dir=0.05,
        vertex_spread=10.0,
        seed=seed,
    )


def test_generate_tracks_dimensions_and_noise():
    """Verifies the dimensions of the generated features and the presence of noise labels."""
    n_tracks_target = 100
    hits_per_track_target = 5
    n_noise_target = 20

    config = _default_config()
    features, labels = generate_tracks(
        n_tracks=n_tracks_target,
        n_noise=n_noise_target,
        config=config,
    )

    expected_total_hits = (n_tracks_target * hits_per_track_target) + n_noise_target

    assert features.shape == (expected_total_hits, 5)
    assert labels.shape == (expected_total_hits,)
    assert -1 in labels


def test_generate_tracks_determinism():
    """Validates that the random seed guarantees exact reproducibility of the dataset."""
    config = _default_config(seed=123)

    features_run_one, labels_run_one = generate_tracks(
        n_tracks=50, n_noise=10, config=config
    )
    features_run_two, labels_run_two = generate_tracks(
        n_tracks=50, n_noise=10, config=config
    )

    assert np.allclose(features_run_one, features_run_two)
    assert np.array_equal(labels_run_one, labels_run_two)
