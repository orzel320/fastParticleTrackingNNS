import numpy as np
from hep_tracking.data import generate_tracks


def test_generate_tracks_dimensions_and_noise():
    """Verifies the dimensions of the generated features and the presence of noise labels.

    Ensures that the feature matrix has 5 columns and that the noise label (-1) 
    is correctly injected into the target array.
    """
    n_tracks_target = 100
    hits_per_track_target = 5
    n_noise_target = 20

    features, labels = generate_tracks(
        n_tracks=n_tracks_target,
        hits_per_track=hits_per_track_target,
        n_noise=n_noise_target,
    )

    expected_total_hits = (n_tracks_target * hits_per_track_target) + n_noise_target

    assert features.shape == (expected_total_hits, 5)
    assert labels.shape == (expected_total_hits,)
    assert -1 in labels


def test_generate_tracks_determinism():
    """Validates that the random seed guarantees exact reproducibility of the dataset."""
    features_run_one, labels_run_one = generate_tracks(
        n_tracks=50, hits_per_track=5, n_noise=10, seed=123
    )
    
    features_run_two, labels_run_two = generate_tracks(
        n_tracks=50, hits_per_track=5, n_noise=10, seed=123
    )

    assert np.allclose(features_run_one, features_run_two)
    assert np.array_equal(labels_run_one, labels_run_two)