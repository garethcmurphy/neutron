import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .synth import generate_synthetic_tof_runs
from .plotting import save_pca

def run_pipeline(outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    runs, X, _ = generate_synthetic_tof_runs()
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=5).fit_transform(Xs)
    km = KMeans(n_clusters=3, n_init=10).fit(Z)
    save_pca(Z[:,:2], km.labels_, os.path.join(outdir,"pca_kmeans.png"), "PCA + kmeans")
    pd.DataFrame({"run_id": runs.run_id, "cluster": km.labels_}).to_csv(
        os.path.join(outdir,"clusters.csv"), index=False)
