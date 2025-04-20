#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler # Good practice
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import warnings
import joblib

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

def find_optimal_eps(X, min_samples, plot_dir='dbscan/plots'):
    """Generate k-distance graph to help find optimal eps."""
    print(f"\n--- Finding Optimal Eps using k-distance graph (min_samples={min_samples}) ---")
    
    # Calculate distances to k-th nearest neighbor (k = min_samples)
    # n_neighbors needs to be min_samples itself if we want the distance to the k-th neighbor
    # Or min_samples + 1 if we need to include the point itself in the neighbors list
    # Sklearn's NearestNeighbors includes the point itself, so use min_samples + 1 to get k=min_samples neighbors excluding self
    # However, common practice uses k=min_samples directly in the nn search
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Get the distance to the k-th neighbor (k = min_samples)
    # distances[:, min_samples-1] is the distance to the k-th neighbor if n_neighbors=min_samples
    kth_distances = np.sort(distances[:, min_samples-1], axis=0)
    
    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(kth_distances)
    plt.title(f'k-Distance Graph (k = {min_samples})')
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.grid(True)
    k_dist_path = os.path.join(plot_dir, f'k_distance_graph_k{min_samples}.png')
    plt.savefig(k_dist_path)
    plt.close()
    print(f"k-Distance graph saved to {k_dist_path}")
    print("Inspect the graph: Look for the 'elbow' or point of maximum curvature.")
    print("This point's distance value on the y-axis is a good candidate for 'eps'.")
    
    # Placeholder for automatic elbow detection (can be complex)
    # For now, we'll rely on visual inspection and set a default or suggested value
    # Heuristic: Choose a point around where the curve starts to rise sharply.
    # Example: find the index where the slope increases most significantly
    # This is a simplified heuristic
    try:
        diffs = np.diff(kth_distances, 2) # Second derivative (approximation)
        elbow_index = np.argmax(diffs) + 1 # +1 to adjust index
        suggested_eps = kth_distances[elbow_index]
        print(f"--> Tentative suggested 'eps' based on max curvature heuristic: {suggested_eps:.4f}")
    except Exception as e:
        print(f"Could not automatically suggest eps: {e}")
        suggested_eps = 2.0 # Default fallback
        print(f"Using default eps fallback: {suggested_eps}")
        
    return suggested_eps # Return the suggested or default value

def perform_dbscan(X, eps, min_samples):
    """Perform DBSCAN clustering."""
    print(f"\n--- Performing DBSCAN (eps={eps:.4f}, min_samples={min_samples}) ---")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X)
    
    # Core samples indices
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    
    # Calculate Silhouette Score (only on non-noise points)
    if n_clusters_ > 0: # Cannot calculate silhouette score with 0 or 1 cluster
        non_noise_mask = (labels != -1)
        if np.sum(non_noise_mask) > 1: # Need at least 2 points that are not noise
            X_filtered = X[non_noise_mask]
            labels_filtered = labels[non_noise_mask]
            if len(set(labels_filtered)) > 1: # Need more than 1 cluster among non-noise points
                silhouette_avg = silhouette_score(X_filtered, labels_filtered)
                print(f'Average Silhouette Score (excluding noise): {silhouette_avg:.4f}')
            else:
                 print("Silhouette Score cannot be calculated: Only one cluster found among non-noise points.")
        else:
             print("Silhouette Score cannot be calculated: Not enough non-noise points found.")
    else:
        print("Silhouette Score cannot be calculated: No clusters found (all noise?).")
        
    return dbscan, labels, core_samples_mask, n_clusters_, n_noise_

def visualize_clusters_pca(X, labels, core_samples_mask, y, plot_dir='dbscan/plots', algorithm_name='DBSCAN', n_clusters=None, n_noise=None, eps=None, min_samples=None, random_state=42):
    """Visualize the clusters using PCA."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    unique_labels = set(labels)
    
    # Create a color map: one color per cluster + black for noise
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0))) # Colors for actual clusters
    color_map = {label: colors[i] for i, label in enumerate(sorted([l for l in unique_labels if l != -1]))}
    color_map[-1] = (0, 0, 0, 1) # Black for noise

    # Plot points
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color_map[k]), markeredgecolor='k', markersize=10, label=f'Cluster {k}' if k != -1 else None)
        
        xy = X_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color_map[k]), markeredgecolor='k', markersize=5, label='Noise' if k == -1 else None)

    # Add legend entries if they weren't created (e.g., only noise found)
    handles, current_labels = plt.gca().get_legend_handles_labels()
    if 'Noise' not in current_labels and -1 in unique_labels:
        # Create a dummy plot for the legend
        noise_handle = plt.Line2D([0], [0], marker='o', color='w', label='Noise', 
                                markerfacecolor=tuple(color_map[-1]), markersize=5, markeredgecolor='k')
        handles.append(noise_handle)
    if len([l for l in unique_labels if l != -1]) > 0 and not any(l.startswith('Cluster') for l in current_labels):
         # Create dummy plot for clusters if needed
         cluster_handle = plt.Line2D([0], [0], marker='o', color='w', label=f'Clusters (Total: {n_clusters})', 
                                markerfacecolor='grey', markersize=10, markeredgecolor='k')
         handles.append(cluster_handle)

    plt.legend(handles=handles)
    title = f'{algorithm_name} Results (PCA Reduced)\neps={eps:.2f}, min_samples={min_samples} | Clusters: {n_clusters}, Noise: {n_noise}'
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'dbscan_clusters_pca_eps{eps:.2f}_ms{min_samples}.png')
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
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png') # Overwrite is fine
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_clustered_data_eps{eps:.2f}_ms{ms}.csv')
    NOISE_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_noise_points_eps{eps:.2f}_ms{ms}.csv')
    MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_model_eps{eps:.2f}_ms{ms}.joblib')
    RANDOM_STATE = 42
    
    # --- DBSCAN Parameters ---
    # Reduced min_samples to be less restrictive
    N_DIMS = 9
    MIN_SAMPLES = max(5, N_DIMS) # Reduced from 2*N_DIMS to be less restrictive
    
    # Grid search for best eps
    print("\n--- Finding optimal parameters through grid search ---")
    eps_range = np.linspace(0.5, 3.0, 10)
    best_eps = None
    best_silhouette = -1
    best_noise_ratio = 1.0  # Initialize with worst case
    
    # Load and scale data
    X, y, df_original_with_target = load_data(DATA_FILE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nPerforming grid search for optimal eps...")
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:  # Skip if less than 2 clusters
            continue
            
        # Calculate noise ratio
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Calculate silhouette score for non-noise points
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            try:
                silhouette = silhouette_score(X_scaled[non_noise_mask], 
                                            labels[non_noise_mask])
                
                # Balance between silhouette score and noise ratio
                # We want high silhouette score but reasonable noise ratio
                if silhouette > best_silhouette and noise_ratio < 0.3:  # Limit noise to 30%
                    best_silhouette = silhouette
                    best_eps = eps
                    print(f"New best eps: {eps:.2f} (Silhouette: {silhouette:.3f}, Noise ratio: {noise_ratio:.2%})")
            except:
                continue
    
    if best_eps is None:
        print("Could not find optimal eps, using default value.")
        best_eps = 1.5
    
    print(f"\nSelected parameters: eps={best_eps:.2f}, min_samples={MIN_SAMPLES}")
    
    # --- Perform DBSCAN with best parameters ---
    model, labels, core_mask, n_clusters, n_noise = perform_dbscan(X_scaled, eps=best_eps, min_samples=MIN_SAMPLES)

    # --- Save Model ---
    model_filename = MODEL_FILE_TEMPLATE.format(eps=best_eps, ms=MIN_SAMPLES)
    try:
        joblib.dump(model, model_filename)
        print(f"\nDBSCAN model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving DBSCAN model: {e}")

    # --- Visualize Results ---
    visualize_clusters_pca(X_scaled, labels, core_mask, y, plot_dir=PLOT_DIR, 
                           n_clusters=n_clusters, n_noise=n_noise, eps=best_eps,
                           min_samples=MIN_SAMPLES, random_state=RANDOM_STATE)

    print("\nDBSCAN clustering process completed!")

if __name__ == "__main__":
    main() 