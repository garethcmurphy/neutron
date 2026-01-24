"""
Synthetic data generator for neutron time-of-flight experiments.

This module generates synthetic neutron detector data that mimics real
time-of-flight neutron scattering experiments with configurable parameters.
"""

import numpy as np
import pandas as pd


def generate_synthetic_tof_runs(
    n_runs: int = 400, n_detectors: int = 24, n_bins: int = 300, random_state: int = 7
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate synthetic neutron time-of-flight run data.

    Creates synthetic data for neutron scattering experiments with multiple runs,
    detectors, and time bins. Each run belongs to one of three clusters with
    distinct peak patterns.

    Args:
        n_runs: Number of experimental runs to generate (default: 400)
        n_detectors: Number of detectors per run (default: 24)
        n_bins: Number of time bins for TOF measurement (default: 300)
        random_state: Random seed for reproducibility (default: 7)

    Returns:
        tuple containing:
            - runs_df: DataFrame with run metadata (run_id, true_label, etc.)
            - X_flat: Flattened detector data array of shape (n_runs, n_detectors * n_bins)
            - tof: Time-of-flight bin centers array

    Example:
        >>> runs, X, tof = generate_synthetic_tof_runs(n_runs=100, n_detectors=12)
        >>> print(runs.shape, X.shape, tof.shape)
        (100, 4) (100, 3600) (300,)
    """
    rng = np.random.default_rng(random_state)
    tof = np.linspace(0, 1.0, n_bins)
    true_labels = rng.integers(0, 3, size=n_runs)
    det_scale = rng.lognormal(0, 0.2, size=n_detectors)
    x = np.zeros((n_runs, n_detectors, n_bins))

    def gaussian_peak(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Generate a Gaussian peak centered at mu with width sigma."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    for i in range(n_runs):
        # Define peak patterns for each cluster
        if true_labels[i] == 0:
            peaks = [(0.25, 0.025, 1.0), (0.62, 0.035, 0.6)]
        elif true_labels[i] == 1:
            peaks = [(0.30, 0.030, 0.9), (0.70, 0.030, 0.8)]
        else:
            peaks = [(0.22, 0.020, 0.7), (0.55, 0.040, 1.1)]

        background = 0.08 + 0.05 * rng.random()
        beam = rng.lognormal(0, 0.25)

        for d in range(n_detectors):
            signal = sum(
                amplitude * gaussian_peak(tof, mu, sigma) for mu, sigma, amplitude in peaks
            )
            noise = rng.normal(0, 0.03, size=n_bins)
            x[i, d] = beam * det_scale[d] * (signal + background) + noise

    # Ensure all values are non-negative (physical constraint)
    x = np.clip(x, 0, None)
    x_flat = x.reshape(n_runs, -1)

    runs_df = pd.DataFrame(
        {
            "run_id": [f"run_{i:04d}" for i in range(n_runs)],
            "true_label": true_labels,
            "n_detectors": n_detectors,
            "n_bins": n_bins,
        }
    )
    return runs_df, x_flat, tof
