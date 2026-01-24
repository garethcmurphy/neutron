"""
Visualization utilities for neutron data analysis.

This module provides functions for creating and saving plots of PCA results
and clustering analysis.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to create
    """
    os.makedirs(path, exist_ok=True)


def save_pca(
    z: np.ndarray,
    labels: np.ndarray,
    path: str,
    title: str,
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    dpi: int = 150,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    """
    Create and save a scatter plot of PCA results with cluster labels.

    Args:
        Z: PCA-transformed data array of shape (n_samples, n_components).
            Only the first two components are plotted.
        labels: Cluster labels for each sample
        path: Output file path for the saved figure
        title: Title for the plot
        xlabel: Label for x-axis (default: "PC1")
        ylabel: Label for y-axis (default: "PC2")
        dpi: Resolution of the saved figure (default: 150)
        figsize: Figure size as (width, height) in inches (default: None, uses matplotlib default)

    Example:
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.cluster import KMeans
        >>> # Assuming X is your data
        >>> Z = PCA(n_components=2).fit_transform(X)
        >>> labels = KMeans(n_clusters=3).fit_predict(Z)
        >>> save_pca(Z, labels, "output.png", "PCA Results")
    """
    # Ensure parent directory exists if path contains directory components
    dir_path = os.path.dirname(path)
    if dir_path:
        ensure_dir(dir_path)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()

    plt.scatter(z[:, 0], z[:, 1], c=labels, s=18, cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
