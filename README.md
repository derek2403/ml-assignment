# Clustering Analysis Workflow

This repository contains a complete clustering analysis workflow for data exploration and pattern discovery.

## Features

- **Data Preprocessing**: Handles outliers and missing values
- **Data Scaling and Encoding**: Normalizes data and performs one-hot encoding for categorical features
- **Feature Selection**: Identifies the most important features using statistical methods
- **Dimensionality Reduction**: Applies PCA, t-SNE, Kernel PCA, and UMAP
- **Clustering Analysis**: Uses multiple algorithms (K-Means, Agglomerative, GMM, DBSCAN) with automatic hyperparameter tuning

## Requirements

- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - umap-learn

## Getting Started

1. Clone this repository
2. Place your dataset as `dataset.csv` in the root directory
3. Install required packages: `pip install -r requirements.txt`
4. Run the main script: `python src/main.py`

## Data Format

The input file `dataset.csv` should be a CSV file with:
- The first row containing column headers
- A mix of numerical and categorical features
- No missing column names

## Output

The workflow generates multiple output files:
- Intermediate CSV files from each processing step
- Visualizations of dimensionality reduction results (PNG format)
- Clustering result visualizations (PNG format)
- Evaluation metrics for different clustering approaches
- A summary JSON file with key findings

## Individual Scripts

You can also run individual steps of the workflow:

1. `src/1_data_preprocessing.py` - Handles outliers and missing values
2. `src/2_scaling_encoding.py` - Normalizes and encodes data
3. `src/3_feature_selection.py` - Selects most relevant features
4. `src/4_dimensionality_reduction.py` - Reduces data dimensions for visualization
5. `src/5_clustering.py` - Performs clustering analysis

## Customization

To customize the analysis:
- Edit parameter values in individual scripts
- Modify the clustering algorithms or metrics in `src/5_clustering.py`
- Adjust visualizations in any script