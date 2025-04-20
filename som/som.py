#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # For legend
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Good practice
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans # To cluster SOM nodes
from minisom import MiniSom
import joblib
import os
import math
import warnings

# Suppress warnings if needed
# warnings.filterwarnings("ignore", category=FutureWarning)

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

def train_som(X_scaled, grid_size=None, sigma=1.5, learning_rate=0.5, num_iterations=5000, random_seed=42):
    """Initialize and train the Self-Organizing Map."""
    print("\n--- Training Self-Organizing Map --- ")
    n_features = X_scaled.shape[1]
    n_samples = X_scaled.shape[0]
    
    # Determine grid size (heuristic: 5 * sqrt(N)) if not provided
    if grid_size is None:
        map_size = math.ceil(5 * math.sqrt(n_samples))
        # Make it roughly square-ish
        map_rows = map_cols = math.ceil(math.sqrt(map_size))
        grid_size = (map_rows, map_cols)
        print(f"Automatically determined grid size: {grid_size}")
    else:
        map_rows, map_cols = grid_size
        print(f"Using provided grid size: {grid_size}")

    som = MiniSom(map_rows, map_cols, n_features,
                sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian', random_seed=random_seed)

    print(f"Initializing SOM weights...")
    som.pca_weights_init(X_scaled) # Initialize weights using PCA
    print(f"Training SOM for {num_iterations} iterations...")
    som.train_batch(X_scaled, num_iterations, verbose=True) # Use train_batch for efficiency
    # som.train_random(X_scaled, num_iterations, verbose=True)
    print("SOM training completed.")
    return som

def plot_som_distance_map(som, plot_dir='som/plots'):
    """Plot the SOM's distance map (U-Matrix)."""
    print("\n--- Plotting SOM Distance Map (U-Matrix) --- ")
    distance_map = som.distance_map()
    plt.figure(figsize=(10, 10))
    plt.pcolor(distance_map.T, cmap='bone_r') # plotting the distance map as background
    plt.colorbar(label='Inter-neuron distance')
    plt.title('SOM Distance Map (U-Matrix)')
    plt.xticks(np.arange(som.get_weights().shape[0] + 1))
    plt.yticks(np.arange(som.get_weights().shape[1] + 1))
    plt.grid(True)
    plt.tight_layout()
    u_matrix_path = os.path.join(plot_dir, 'som_u_matrix.png')
    plt.savefig(u_matrix_path)
    plt.close()
    print(f"U-Matrix plot saved to {u_matrix_path}")

def find_optimal_k_for_nodes(som_weights, max_k=10, random_state=42):
    """Find optimal k for clustering SOM nodes using silhouette score."""
    print("\n--- Finding optimal k for clustering SOM nodes ---")
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    # Reshape weights for clustering (each node is a sample)
    num_nodes = som_weights.shape[0] * som_weights.shape[1]
    node_vectors = som_weights.reshape(num_nodes, -1)

    for k in k_range:
        print(f"Calculating K-Means on nodes for k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        node_labels = kmeans.fit_predict(node_vectors)
        if len(set(node_labels)) > 1:
            score = silhouette_score(node_vectors, node_labels)
            silhouette_scores.append(score)
            print(f"  Node Silhouette Score: {score:.4f}")
        else:
            print("  Only one cluster found for nodes, score not applicable.")
            silhouette_scores.append(np.nan)
            
    valid_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) if not np.isnan(score)]
    if not valid_scores:
        print("Error: Could not calculate silhouette scores for any k on nodes.")
        return 2 # Default fallback
    
    valid_k_range, valid_silhouette_scores = zip(*valid_scores)
    optimal_k = valid_k_range[np.argmax(valid_silhouette_scores)]
    print(f"Optimal k for SOM node clustering based on Silhouette Score: {optimal_k}")
    return optimal_k

def cluster_som_nodes(som_weights, k, random_state=42):
    """Cluster the SOM nodes using K-Means."""
    print(f"\n--- Clustering SOM Nodes into {k} clusters --- ")
    num_nodes = som_weights.shape[0] * som_weights.shape[1]
    node_vectors = som_weights.reshape(num_nodes, -1)
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    node_labels = kmeans.fit_predict(node_vectors)
    print("Node clustering completed.")
    return node_labels.reshape(som_weights.shape[0], som_weights.shape[1]) # Reshape back to grid

def plot_som_node_clusters(som, node_cluster_labels, plot_dir='som/plots'):
    """Visualize the clustered SOM nodes on the grid."""
    print("\n--- Plotting SOM Node Clusters Map --- ")
    plt.figure(figsize=(10, 10))
    num_clusters = len(np.unique(node_cluster_labels))
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    
    plt.pcolor(node_cluster_labels.T, cmap=plt.cm.get_cmap('viridis', num_clusters))
    plt.colorbar(ticks=range(num_clusters), label='Node Cluster ID')
    
    # Add markers for data points colored by their BMU's cluster (optional, can be slow)
    # Or just show the node cluster background
    
    plt.title(f'SOM Grid Colored by Node Cluster (k={num_clusters})')
    plt.xticks(np.arange(som.get_weights().shape[0] + 1))
    plt.yticks(np.arange(som.get_weights().shape[1] + 1))
    plt.grid(True)
    plt.tight_layout()
    node_clusters_path = os.path.join(plot_dir, f'som_node_clusters_k{num_clusters}.png')
    plt.savefig(node_clusters_path)
    plt.close()
    print(f"SOM Node Clusters plot saved to {node_clusters_path}")

def get_data_point_labels(som, X_scaled, node_cluster_labels):
    """Assign cluster labels to data points based on their BMU's cluster."""
    print("\n--- Assigning cluster labels to data points --- ")
    data_labels = np.zeros(len(X_scaled), dtype=int)
    for i, x in enumerate(X_scaled):
        bmu_row, bmu_col = som.winner(x) # Find Best Matching Unit
        data_labels[i] = node_cluster_labels[bmu_row, bmu_col]
    print("Data point labels assigned.")
    return data_labels

def visualize_clusters_pca(X, labels, y, plot_dir='som/plots', algorithm_name='SOM', k=None, random_state=42):
    """Visualize the final data clusters using PCA."""
    print("\n--- Visualizing Final Data Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
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
    pca_plot_path = os.path.join(plot_dir, f'som_data_clusters_pca_k{k}.png')
    plt.savefig(pca_plot_path)
    plt.close()
    print(f"Data Cluster PCA plot saved to {pca_plot_path}")

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
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png') # Overwrite is fine
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'som_clustered_data_k{k}.csv')
    SOM_MODEL_FILE = os.path.join(OUTPUT_DIR, 'som_model.joblib') # Changed extension
    MAX_K_NODES = 10 # Max clusters for SOM nodes
    RANDOM_STATE = 42
    SOM_ITERATIONS = 10000 # More iterations can improve SOM quality
    # SOM_GRID_SIZE = (10, 10) # Optionally override automatic size
    SOM_GRID_SIZE = None

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    X, y, df_original_with_target = load_data(DATA_FILE)

    # --- Ensure data is scaled ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling confirmed/re-applied.")

    # --- Train SOM ---
    som = train_som(X_scaled, grid_size=SOM_GRID_SIZE, num_iterations=SOM_ITERATIONS, random_seed=RANDOM_STATE)

    # --- Visualize SOM U-Matrix ---
    plot_som_distance_map(som, plot_dir=PLOT_DIR)

    # --- Cluster SOM Nodes ---
    som_weights = som.get_weights()
    optimal_k_nodes = find_optimal_k_for_nodes(som_weights, max_k=MAX_K_NODES, random_state=RANDOM_STATE)
    node_cluster_labels = cluster_som_nodes(som_weights, k=optimal_k_nodes, random_state=RANDOM_STATE)
    plot_som_node_clusters(som, node_cluster_labels, plot_dir=PLOT_DIR)

    # --- Assign Labels to Data Points ---
    data_labels = get_data_point_labels(som, X_scaled, node_cluster_labels)

    # --- Evaluate Final Data Clustering ---
    silhouette_avg = np.nan
    if len(set(data_labels)) > 1:
        try:
            silhouette_avg = silhouette_score(X_scaled, data_labels)
            print(f"\nFinal Data Point Clustering Evaluation:")
            print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        except Exception as e:
             print(f"Error calculating final silhouette score: {e}")
    else:
         print("\nOnly one cluster assigned to data points. Silhouette score not applicable.")

    # --- Save SOM Model ---
    joblib.dump(som, SOM_MODEL_FILE) # Use joblib or pickle
    print(f"\nSOM model saved to {SOM_MODEL_FILE}")

    # --- Add Cluster Labels to Original Data ---
    df_clustered = df_original_with_target.loc[X.index].copy()
    df_clustered['Cluster'] = data_labels
    # clustered_data_file = CLUSTERED_DATA_FILE_TEMPLATE.format(k=optimal_k_nodes)
    # df_clustered.to_csv(clustered_data_file, index=False)
    # print(f"Clustered data (k={optimal_k_nodes}) saved to {clustered_data_file}")
    print(f"\nCluster labels generated for k={optimal_k_nodes} (not saving clustered data file).")

    # --- Visualize Final Data Clusters ---
    visualize_clusters_pca(X_scaled, data_labels, y, plot_dir=PLOT_DIR, k=optimal_k_nodes, random_state=RANDOM_STATE)

    print("\nSOM clustering process completed!")


if __name__ == "__main__":
    main() 