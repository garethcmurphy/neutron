# synthetic data generator (see chat for explanation)
import numpy as np
import pandas as pd

def generate_synthetic_tof_runs(n_runs=400, n_detectors=24, n_bins=300, random_state=7):
    rng = np.random.default_rng(random_state)
    tof = np.linspace(0, 1.0, n_bins)
    true_labels = rng.integers(0, 3, size=n_runs)
    det_scale = rng.lognormal(0, 0.2, size=n_detectors)
    X = np.zeros((n_runs, n_detectors, n_bins))

    def g(x, mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    for i in range(n_runs):
        peaks = [(0.25,0.025,1.0),(0.62,0.035,0.6)] if true_labels[i]==0 else                 [(0.30,0.030,0.9),(0.70,0.030,0.8)] if true_labels[i]==1 else                 [(0.22,0.020,0.7),(0.55,0.040,1.1)]
        background = 0.08 + 0.05 * rng.random()
        beam = rng.lognormal(0, 0.25)

        for d in range(n_detectors):
            signal = sum(a * g(tof, m, s) for m, s, a in peaks)
            noise = rng.normal(0, 0.03, size=n_bins)
            X[i, d] = beam * det_scale[d] * (signal + background) + noise

    X = np.clip(X, 0, None)
    X_flat = X.reshape(n_runs, -1)

    runs_df = pd.DataFrame({
        "run_id": [f"run_{i:04d}" for i in range(n_runs)],
        "true_label": true_labels,
        "n_detectors": n_detectors,
        "n_bins": n_bins,
    })
    return runs_df, X_flat, tof
