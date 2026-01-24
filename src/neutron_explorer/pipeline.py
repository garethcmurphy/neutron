"""
Main analysis pipeline for neutron time-of-flight data.

This module orchestrates the complete workflow of generating synthetic data,
performing dimensionality reduction with PCA, clustering with k-means, and
saving visualizations and results.
"""

import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .plotting import save_pca
from .synth import generate_synthetic_tof_runs


def run_pipeline(
    outdir: str = "outputs",
    n_runs: int = 400,
    n_detectors: int = 24,
    n_bins: int = 300,
    n_components: int = 5,
    n_clusters: int = 3,
    random_state: int = 7,
) -> dict:
    """
    Run the complete neutron data analysis pipeline.

    This function generates synthetic neutron TOF data, applies PCA for dimensionality
    reduction, performs k-means clustering, and saves results including visualizations
    and cluster assignments.

    Args:
        outdir: Output directory for results (default: "outputs")
        n_runs: Number of synthetic runs to generate (default: 400)
        n_detectors: Number of detectors per run (default: 24)
        n_bins: Number of time bins (default: 300)
        n_components: Number of PCA components to compute (default: 5)
        n_clusters: Number of clusters for k-means (default: 3)
        random_state: Random seed for reproducibility (default: 7)

    Returns:
        dict: Dictionary containing:
            - 'runs': DataFrame with run metadata
            - 'clusters': Cluster labels for each run
            - 'pca_variance': Explained variance ratio for each component
            - 'silhouette': Silhouette score for clustering quality

    Example:
        >>> results = run_pipeline(outdir="my_results", n_clusters=4)
        >>> print(f"Silhouette score: {results['silhouette']:.3f}")
    """
    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Generate synthetic data
    runs, x, _ = generate_synthetic_tof_runs(
        n_runs=n_runs, n_detectors=n_detectors, n_bins=n_bins, random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)

    # Apply PCA
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(x_s)

    # Perform k-means clustering
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(z)

    # Calculate clustering quality metric
    silhouette = silhouette_score(z, km.labels_)

    # Save PCA visualization
    save_pca(
        z[:, :2], km.labels_, os.path.join(outdir, "pca_kmeans.png"), "PCA + k-means Clustering"
    )

    # Save cluster assignments
    cluster_df = pd.DataFrame({"run_id": runs.run_id, "cluster": km.labels_})
    cluster_df.to_csv(os.path.join(outdir, "clusters.csv"), index=False)

    # Return results for programmatic access
    return {
        "runs": runs,
        "clusters": km.labels_,
        "pca_variance": pca.explained_variance_ratio_,
        "silhouette": silhouette,
    }
