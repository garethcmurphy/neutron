"""
Main entry point for running the neutron data analysis pipeline.

This script runs the complete analysis workflow including data generation,
PCA, clustering, and visualization.
"""

from neutron_explorer.pipeline import run_pipeline

if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline completed successfully!")
    print(f"Silhouette score: {results['silhouette']:.3f}")
    print("Results saved to 'outputs/' directory")
