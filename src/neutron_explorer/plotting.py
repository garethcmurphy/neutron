import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_pca(Z, labels, path, title):
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.scatter(Z[:,0], Z[:,1], c=labels, s=18)
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.savefig(path, dpi=150); plt.close()
