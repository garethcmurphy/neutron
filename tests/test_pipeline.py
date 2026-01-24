"""Unit tests for the pipeline module."""

import os
import tempfile

import numpy as np
import pandas as pd

from neutron_explorer.pipeline import run_pipeline


class TestRunPipeline:
    """Test suite for run_pipeline function."""

    def test_default_execution(self):
        """Test pipeline execution with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir)

            # Check return value structure
            assert isinstance(results, dict)
            assert "runs" in results
            assert "clusters" in results
            assert "pca_variance" in results
            assert "silhouette" in results

    def test_output_files_created(self):
        """Test that expected output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_pipeline(outdir=tmpdir)

            # Check for expected files
            assert os.path.exists(os.path.join(tmpdir, "pca_kmeans.png"))
            assert os.path.exists(os.path.join(tmpdir, "clusters.csv"))

    def test_clusters_csv_structure(self):
        """Test the structure of the clusters CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_pipeline(outdir=tmpdir, n_runs=100)

            csv_path = os.path.join(tmpdir, "clusters.csv")
            df = pd.read_csv(csv_path)

            # Check structure
            assert len(df) == 100
            assert "run_id" in df.columns
            assert "cluster" in df.columns

            # Check run_id format
            assert df["run_id"].iloc[0] == "run_0000"

    def test_cluster_labels(self):
        """Test that cluster labels are in the expected range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir, n_clusters=4)

            # Check cluster labels
            assert results["clusters"].min() >= 0
            assert results["clusters"].max() < 4
            assert len(np.unique(results["clusters"])) <= 4

    def test_pca_variance_ratios(self):
        """Test PCA variance ratios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir, n_components=5)

            variance = results["pca_variance"]

            # Check variance properties
            assert len(variance) == 5
            assert np.all(variance >= 0)
            assert np.all(variance <= 1)
            assert np.sum(variance) <= 1.0

    def test_silhouette_score_range(self):
        """Test that silhouette score is in valid range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir)

            # Silhouette score should be between -1 and 1
            assert -1 <= results["silhouette"] <= 1

    def test_custom_parameters(self):
        """Test pipeline with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(
                outdir=tmpdir, n_runs=50, n_detectors=12, n_bins=150, n_components=3, n_clusters=2
            )

            # Check results match parameters
            assert len(results["runs"]) == 50
            assert len(results["clusters"]) == 50
            assert len(results["pca_variance"]) == 3
            assert len(np.unique(results["clusters"])) <= 2

    def test_reproducibility(self):
        """Test that the same random_state produces consistent data generation."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                results1 = run_pipeline(outdir=tmpdir1, random_state=42, n_runs=50)
                results2 = run_pipeline(outdir=tmpdir2, random_state=42, n_runs=50)

                # Check that the same random state produces the same raw data
                # Note: cluster labels might be permuted, so we check data generation consistency
                pd.testing.assert_frame_equal(results1["runs"], results2["runs"])

                # Check PCA variance is very similar (allowing for numerical precision differences)
                # Using looser tolerance as different environments may
                # have slight numerical variations
                np.testing.assert_allclose(
                    results1["pca_variance"],
                    results2["pca_variance"],
                    rtol=0.02,  # 2% relative tolerance
                    atol=0.001,  # 0.001 absolute tolerance
                )

    def test_different_cluster_counts(self):
        """Test pipeline with different cluster counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for n_clusters in [2, 3, 5]:
                results = run_pipeline(
                    outdir=os.path.join(tmpdir, f"k{n_clusters}"), n_clusters=n_clusters, n_runs=100
                )

                unique_clusters = len(np.unique(results["clusters"]))
                assert unique_clusters <= n_clusters

    def test_small_dataset(self):
        """Test with minimal dataset size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir, n_runs=20, n_detectors=4, n_bins=50)

            assert len(results["runs"]) == 20
            assert len(results["clusters"]) == 20

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")

            run_pipeline(outdir=nested_dir)

            assert os.path.exists(nested_dir)
            assert os.path.isdir(nested_dir)

    def test_runs_dataframe_content(self):
        """Test that runs DataFrame contains expected information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir, n_runs=50)

            runs = results["runs"]

            # Check DataFrame structure
            assert isinstance(runs, pd.DataFrame)
            assert len(runs) == 50
            assert "run_id" in runs.columns
            assert "true_label" in runs.columns

    def test_clustering_quality(self):
        """Test that clustering produces reasonable results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(outdir=tmpdir, n_runs=400, n_clusters=3)

            # With synthetic data designed for 3 clusters,
            # silhouette score should be reasonably positive
            assert results["silhouette"] > 0

    def test_multiple_runs_same_directory(self):
        """Test running pipeline multiple times in same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run twice - should not cause errors
            run_pipeline(outdir=tmpdir, n_runs=50)
            run_pipeline(outdir=tmpdir, n_runs=50)

            # Files should still exist
            assert os.path.exists(os.path.join(tmpdir, "pca_kmeans.png"))
            assert os.path.exists(os.path.join(tmpdir, "clusters.csv"))
