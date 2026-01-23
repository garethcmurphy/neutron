"""Unit tests for the synthetic data generation module."""

import numpy as np
import pandas as pd
import pytest

from neutron_explorer.synth import generate_synthetic_tof_runs


class TestGenerateSyntheticTofRuns:
    """Test suite for generate_synthetic_tof_runs function."""

    def test_default_parameters(self):
        """Test data generation with default parameters."""
        runs, X, tof = generate_synthetic_tof_runs()
        
        # Check return types
        assert isinstance(runs, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(tof, np.ndarray)
        
        # Check default dimensions
        assert len(runs) == 400
        assert X.shape == (400, 24 * 300)
        assert len(tof) == 300

    def test_custom_parameters(self):
        """Test data generation with custom parameters."""
        n_runs, n_detectors, n_bins = 100, 12, 200
        runs, X, tof = generate_synthetic_tof_runs(
            n_runs=n_runs,
            n_detectors=n_detectors,
            n_bins=n_bins
        )
        
        # Check custom dimensions
        assert len(runs) == n_runs
        assert X.shape == (n_runs, n_detectors * n_bins)
        assert len(tof) == n_bins

    def test_runs_dataframe_structure(self):
        """Test that the runs DataFrame has the expected structure."""
        runs, _, _ = generate_synthetic_tof_runs(n_runs=50)
        
        # Check required columns
        assert "run_id" in runs.columns
        assert "true_label" in runs.columns
        assert "n_detectors" in runs.columns
        assert "n_bins" in runs.columns
        
        # Check run_id format
        assert runs["run_id"].iloc[0] == "run_0000"
        assert runs["run_id"].iloc[49] == "run_0049"
        
        # Check labels are in valid range (0, 1, 2)
        assert runs["true_label"].min() >= 0
        assert runs["true_label"].max() <= 2

    def test_data_non_negative(self):
        """Test that generated data contains no negative values."""
        _, X, _ = generate_synthetic_tof_runs(n_runs=50)
        assert np.all(X >= 0)

    def test_tof_bins_range(self):
        """Test that TOF bins span the expected range."""
        _, _, tof = generate_synthetic_tof_runs()
        assert tof[0] == 0.0
        assert tof[-1] == 1.0
        assert np.all(np.diff(tof) > 0)  # Monotonically increasing

    def test_reproducibility(self):
        """Test that the same random_state produces identical results."""
        runs1, X1, tof1 = generate_synthetic_tof_runs(random_state=42)
        runs2, X2, tof2 = generate_synthetic_tof_runs(random_state=42)
        
        # Check reproducibility
        pd.testing.assert_frame_equal(runs1, runs2)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(tof1, tof2)

    def test_different_random_states(self):
        """Test that different random_states produce different results."""
        runs1, X1, _ = generate_synthetic_tof_runs(random_state=42)
        runs2, X2, _ = generate_synthetic_tof_runs(random_state=123)
        
        # Results should be different
        assert not np.array_equal(X1, X2)
        assert not runs1["true_label"].equals(runs2["true_label"])

    def test_cluster_distribution(self):
        """Test that all three clusters are represented."""
        runs, _, _ = generate_synthetic_tof_runs(n_runs=400)
        unique_labels = runs["true_label"].unique()
        
        # Should have 3 distinct clusters
        assert len(unique_labels) == 3
        assert set(unique_labels) == {0, 1, 2}

    def test_data_variability(self):
        """Test that generated data has reasonable variability."""
        _, X, _ = generate_synthetic_tof_runs(n_runs=100)
        
        # Check that data is not constant
        assert X.std() > 0
        
        # Check that different runs have different values
        assert not np.array_equal(X[0], X[1])

    def test_small_dataset(self):
        """Test with minimal parameters."""
        runs, X, tof = generate_synthetic_tof_runs(
            n_runs=10,
            n_detectors=4,
            n_bins=50
        )
        
        assert len(runs) == 10
        assert X.shape == (10, 4 * 50)
        assert len(tof) == 50

    def test_metadata_consistency(self):
        """Test that metadata in DataFrame matches actual data dimensions."""
        n_det, n_bin = 15, 250
        runs, X, _ = generate_synthetic_tof_runs(
            n_runs=50,
            n_detectors=n_det,
            n_bins=n_bin
        )
        
        # Check that metadata matches
        assert runs["n_detectors"].iloc[0] == n_det
        assert runs["n_bins"].iloc[0] == n_bin
        assert X.shape[1] == n_det * n_bin
