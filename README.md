# Neutron Run Explorer (Synthetic TOF)



A public, machine learning demonstration using synthetic neutron time-of-flight (TOF) data. This project showcases data analysis techniques including Principal Component Analysis (PCA), k-means clustering, and detector correlations for neutron scattering experiments.

🌐 **Live Demo**: [https://garethcmurphy.github.io/neutron/](https://garethcmurphy.github.io/neutron/)

![pca](pca_kmeans.png)

## Overview

The Neutron Run Explorer generates synthetic neutron detector data that mimics real time-of-flight neutron scattering experiments. It demonstrates how machine learning techniques can be applied to:

- Identify patterns in neutron detector readings
- Cluster similar experimental runs
- Visualize high-dimensional detector data
- Analyze detector correlations

## Features

- **Synthetic Data Generation**: Creates realistic neutron TOF data with configurable parameters
- **PCA Analysis**: Reduces high-dimensional detector data to principal components
- **K-Means Clustering**: Groups similar experimental runs
- **Visualization**: Generates informative plots of PCA results and clustering
- **Modular Design**: Easy-to-extend codebase with clear separation of concerns

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/garethcmurphy/neutron.git
cd neutron

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/garethcmurphy/neutron.git
cd neutron

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python run_pipeline.py
```

This will:

1. Generate 400 synthetic neutron runs with 24 detectors each
2. Apply PCA to reduce dimensionality
3. Perform k-means clustering to identify 3 distinct groups
4. Save results to the `outputs/` directory:
   - `pca_kmeans.png`: Visualization of PCA results with cluster labels
   - `clusters.csv`: Cluster assignments for each run

### Programmatic Usage

```python
from neutron_explorer.pipeline import run_pipeline

# Run with default settings
run_pipeline()

# Run with custom output directory
run_pipeline(outdir="my_results")
```

### Advanced Usage

For more control, use the modules directly:

```python
from neutron_explorer.synth import generate_synthetic_tof_runs
from neutron_explorer.plotting import save_pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Generate synthetic data
runs, X, tof = generate_synthetic_tof_runs(n_runs=400, n_detectors=24, n_bins=300)

# Standardize and apply PCA
Xs = StandardScaler().fit_transform(X)
Z = PCA(n_components=5).fit_transform(Xs)

# Cluster the data
km = KMeans(n_clusters=3, n_init=10).fit(Z)

# Visualize results
save_pca(Z[:, :2], km.labels_, "output.png", "My Analysis")
```

## Project Structure

```bash
neutron/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── run_pipeline.py                    # Main entry point
├── src/
│   └── neutron_explorer/              # Main package
│       ├── __init__.py                # Package initialization
│       ├── pipeline.py                # Main analysis pipeline
│       ├── synth.py                   # Synthetic data generation
│       └── plotting.py                # Visualization utilities
├── tests/                             # Unit tests
│   ├── test_synth.py                  # Tests for data generation
│   ├── test_plotting.py               # Tests for plotting
│   └── test_pipeline.py               # Tests for pipeline
└── demo/                              # React GitHub Pages demo
    ├── src/                           # React source files
    ├── public/                        # Static assets
    ├── index.html                     # HTML entry point
    ├── package.json                   # Node dependencies
    └── vite.config.js                 # Vite build configuration
```

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

## Development

### Running Tests

```bash
# Install project + development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/neutron_explorer --cov-report=html
```

### Code Style

This project follows PEP 8 guidelines. To check code style:

```bash
# Install project + development dependencies
pip install -e ".[dev]"

# Run style checks
ruff check .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Synthetic data generation inspired by neutron time-of-flight experiments
- Uses scikit-learn for machine learning algorithms
- Matplotlib for data visualization
