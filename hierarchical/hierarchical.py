#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler # Good practice
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
import joblib

# Suppress warnings if needed (e.g., memory leak warnings on specific OS/library combos)
# warnings.filterwarnings("ignore", category=UserWarning, message="...")

def load_data(file_path='../data.csv'):
    """Load the scaled dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    # Separate features and target (if Potability exists)
    if 'Potability' in df.columns:
        print("'Potability' column found.")
        y = df['Potability'].astype(int)
        X = df.drop('Potability', axis=1)
    else:
        print("'Potability' column not found.")
        X = df
        y = None
    return X, y, df # Return original df too

def plot_dendrogram(X, method='ward', plot_dir='hierarchical/plots'):
    """Generate and save a dendrogram."""
    print(f"\n--- Generating Dendrogram (method: {method}) ---")
    
    # Calculate linkage matrix
    # Using 'ward' minimizes variance within clusters, suitable for balanced clusters
    # Other options: 'average', 'complete', 'single'
    linked = linkage(X, method=method)
    
    plt.figure(figsize=(15, 8))
    dendrogram(linked,
               orientation='top',
            #    labels=X.index, # Can be too cluttered for large datasets
               distance_sort='descending',
               show_leaf_counts=True,
               truncate_mode='lastp', # Show only the last p merged clusters
               p=30, # Number of clusters to show at the bottom
               show_contracted=True # To visualize density
              )
    plt.title(f'Hierarchical Clustering Dendrogram (Method: {method}, Truncated)')
    plt.xlabel("Cluster size (or sample index if not contracted)")
    plt.ylabel('Distance (Ward)')
    plt.grid(axis='y')
    dendrogram_path = os.path.join(plot_dir, f'dendrogram_{method}.png')
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram saved to {dendrogram_path}")
    print("Inspect the dendrogram to choose an appropriate number of clusters (k). Look for the largest vertical distance without crossing horizontal lines.")

def perform_hierarchical_clustering(X, n_clusters, affinity='euclidean', linkage='ward'):
    """Perform Agglomerative Hierarchical Clustering."""
    print(f"\n--- Performing Hierarchical Clustering (k={n_clusters}, linkage={linkage}) ---")
    # affinity='euclidean' is the default for linkage='ward' and cannot be explicitly set for it.
    # For other linkage methods, affinity might be needed (or replaced by 'metric' in newer sklearn).
    if linkage == 'ward':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        # For non-ward linkage, you might use affinity or metric depending on sklearn version
        # For simplicity, assuming default 'euclidean' if not ward for now. Revisit if using other linkages.
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage) # Add affinity/metric here if needed

    labels = model.fit_predict(X)
    
    silhouette_avg = silhouette_score(X, labels)
    print(f"Hierarchical Clustering completed.")
    print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
    
    return model, labels

def visualize_clusters_pca(X, labels, y, plot_dir='hierarchical/plots', algorithm_name='Hierarchical', k=None, random_state=42):
    """Visualize the clusters using PCA."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    # Determine unique labels and create a color map
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels) # Infer k if not provided
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    # Plot data points with cluster colors
    for cluster_label, color in zip(unique_labels, colors):
        cluster_points = X_pca[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[color],
                    label=f'Cluster {cluster_label}', alpha=0.6)

    plt.title(f'{algorithm_name} Clustering Results (k={k}, PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'hierarchical_clusters_pca_k{k}.png')
    plt.savefig(pca_plot_path)
    plt.close()
    print(f"Cluster PCA plot saved to {pca_plot_path}")

    # --- Optional: Visualize with original Potability labels if available ---
    if y is not None:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, s=50)
        plt.title('PCA Reduced Data Colored by Original Potability')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
        legend_labels = [f'Potability {label}' for label in np.unique(y)]
        plt.legend(handles, legend_labels, title="Original Labels")
        plt.grid(True)
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png') # Same name as kmeans ok
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hierarchical_clustered_data_k{k}.csv')
    LINKAGE_METHOD = 'ward' # Common choice, minimizes variance within clusters
    RANDOM_STATE = 42
    # Set k based on dendrogram inspection or prior knowledge (e.g., k=2 from K-Means)
    CHOSEN_K = 2

    MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hierarchical_model_k{k}.joblib')

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    X, y, df_original_with_target = load_data(DATA_FILE)

    # --- Ensure data is scaled ---
    # Even if loaded from scaled.csv, it's safer to re-apply or verify
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling confirmed/re-applied.")

    # --- Plot Dendrogram ---
    # Plotting dendrogram can be computationally expensive for large datasets.
    # Consider sampling if it takes too long: X_sample = X_scaled[np.random.choice(X_scaled.shape[0], 5000, replace=False)]
    # plot_dendrogram(X_sample, method=LINKAGE_METHOD, plot_dir=PLOT_DIR)
    try:
        plot_dendrogram(X_scaled, method=LINKAGE_METHOD, plot_dir=PLOT_DIR)
    except MemoryError:
        print("MemoryError: Dataset might be too large for full dendrogram generation.")
        print("Consider sampling the data for dendrogram plotting or skipping this step.")

    # --- Perform Hierarchical Clustering with chosen K ---
    print(f"\nProceeding with chosen k = {CHOSEN_K}")
    model, labels = perform_hierarchical_clustering(X_scaled, n_clusters=CHOSEN_K, linkage=LINKAGE_METHOD)

    # --- Save Model ---
    model_filename = MODEL_FILE_TEMPLATE.format(k=CHOSEN_K)
    try:
        # AgglomerativeClustering model state is limited after fit_predict, but save anyway
        joblib.dump(model, model_filename)
        print(f"\nHierarchical model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving Hierarchical model: {e}")

    # --- Add Cluster Labels to Original Data ---
    print(f"\nCluster labels generated for k={CHOSEN_K} (not saving clustered data file).")

    # --- Visualize Results ---
    visualize_clusters_pca(X_scaled, labels, y, plot_dir=PLOT_DIR, k=CHOSEN_K, random_state=RANDOM_STATE)

    print("\nHierarchical clustering process completed!")


if __name__ == "__main__":
    main() 