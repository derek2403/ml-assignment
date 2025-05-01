import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset.csv')

# First, let's explore the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Check for non-numeric values in the dataset
def check_for_string_values(df):
    non_numeric_columns = []
    for column in df.columns:
        mask = pd.to_numeric(df[column], errors='coerce').isna() & df[column].notna()
        if mask.any():
            non_numeric_columns.append(column)
            print(f"Column '{column}' has {mask.sum()} non-numeric values")
            print(f"Example non-numeric values: {df.loc[mask, column].unique()[:5]}")
    return non_numeric_columns

print("\nChecking for non-numeric values:")
non_numeric_columns = check_for_string_values(data)

# Drop columns that are categorical or unnecessary for dimensionality reduction
data_cleaned = data.drop(columns=['STN Code', 'Name of Monitoring Location', 'State Name', 'Type Water Body'])

# Handle non-numeric values by replacing them with NaN
for column in data_cleaned.columns:
    data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors='coerce')

# Now check for missing values after conversion
print("\nMissing values after conversion:")
print(data_cleaned.isna().sum())

# Handle missing data by dropping rows with missing values
data_cleaned = data_cleaned.dropna()
print("\nShape after dropping missing values:", data_cleaned.shape)

# Standardize the data (important for dimensionality reduction)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# Apply PCA to reduce dimensionality to 50 dimensions first (for t-SNE/UMAP to work better)
pca = PCA(n_components=min(50, data_cleaned.shape[1]))
data_pca = pca.fit_transform(data_scaled)

# Now apply t-SNE for 2D visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(data_pca)

# Apply UMAP for 2D visualization
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
data_umap = umap_model.fit_transform(data_pca)

# Plotting t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='blue', edgecolor='k', s=100, alpha=0.5)
plt.title("t-SNE Dimensionality Reduction")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# Plotting UMAP results
plt.figure(figsize=(10, 8))
plt.scatter(data_umap[:, 0], data_umap[:, 1], c='red', edgecolor='k', s=100, alpha=0.5)
plt.title("UMAP Dimensionality Reduction")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()
