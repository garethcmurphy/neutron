"""Unit tests for the plotting module."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from neutron_explorer.plotting import ensure_dir, save_pca


class TestEnsureDir:
    """Test suite for ensure_dir function."""

    def test_create_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_subdir")
            assert not os.path.exists(test_dir)

            ensure_dir(test_dir)
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

    def test_existing_directory(self):
        """Test that existing directory does not raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise an error
            ensure_dir(tmpdir)
            assert os.path.exists(tmpdir)

    def test_nested_directories(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "a", "b", "c")
            ensure_dir(nested_dir)
            assert os.path.exists(nested_dir)


class TestSavePca:
    """Test suite for save_pca function."""

    def test_basic_plot_creation(self):
        """Test basic PCA plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample data
            Z = np.random.randn(100, 2)
            labels = np.random.randint(0, 3, size=100)
            output_path = os.path.join(tmpdir, "test_plot.png")

            save_pca(Z, labels, output_path, "Test Plot")

            # Check file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_plot_with_subdirectory(self):
        """Test that subdirectories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(50, 2)
            labels = np.random.randint(0, 2, size=50)
            output_path = os.path.join(tmpdir, "subdir", "test_plot.png")

            save_pca(Z, labels, output_path, "Test")

            assert os.path.exists(output_path)

    def test_custom_labels(self):
        """Test plot with custom axis labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(30, 2)
            labels = np.array([0, 1, 2] * 10)
            output_path = os.path.join(tmpdir, "custom_labels.png")

            save_pca(
                Z, labels, output_path, "Custom Test", xlabel="Component 1", ylabel="Component 2"
            )

            assert os.path.exists(output_path)

    def test_custom_dpi(self):
        """Test plot with custom DPI setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(50, 2)
            labels = np.random.randint(0, 3, size=50)
            path_low = os.path.join(tmpdir, "low_dpi.png")
            path_high = os.path.join(tmpdir, "high_dpi.png")

            save_pca(Z, labels, path_low, "Low DPI", dpi=50)
            save_pca(Z, labels, path_high, "High DPI", dpi=300)

            # Higher DPI should produce larger file
            assert os.path.getsize(path_high) > os.path.getsize(path_low)

    def test_custom_figsize(self):
        """Test plot with custom figure size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(50, 2)
            labels = np.random.randint(0, 3, size=50)
            output_path = os.path.join(tmpdir, "custom_size.png")

            save_pca(Z, labels, output_path, "Custom Size", figsize=(10, 8))

            assert os.path.exists(output_path)

    def test_different_cluster_counts(self):
        """Test plots with different numbers of clusters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(100, 2)

            for n_clusters in [2, 3, 5, 10]:
                labels = np.random.randint(0, n_clusters, size=100)
                output_path = os.path.join(tmpdir, f"clusters_{n_clusters}.png")

                save_pca(Z, labels, output_path, f"{n_clusters} Clusters")
                assert os.path.exists(output_path)

    def test_no_matplotlib_warning(self):
        """Test that plotting doesn't leave figures open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(50, 2)
            labels = np.random.randint(0, 3, size=50)

            # Get initial figure count
            initial_figs = len(plt.get_fignums())

            for i in range(5):
                output_path = os.path.join(tmpdir, f"plot_{i}.png")
                save_pca(Z, labels, output_path, f"Plot {i}")

            # Should not accumulate figures
            final_figs = len(plt.get_fignums())
            assert final_figs == initial_figs

    def test_large_dataset(self):
        """Test with a larger dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(1000, 2)
            labels = np.random.randint(0, 5, size=1000)
            output_path = os.path.join(tmpdir, "large_data.png")

            save_pca(Z, labels, output_path, "Large Dataset")
            assert os.path.exists(output_path)

    def test_data_with_outliers(self):
        """Test plotting data with extreme values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Z = np.random.randn(100, 2)
            # Add some outliers
            Z[0] = [100, 100]
            Z[1] = [-100, -100]
            labels = np.random.randint(0, 3, size=100)
            output_path = os.path.join(tmpdir, "outliers.png")

            save_pca(Z, labels, output_path, "With Outliers")
            assert os.path.exists(output_path)
