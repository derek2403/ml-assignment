# Path: dbscan/dbscan.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import warnings
import joblib

# --- Configuration ---
# Input data file (already has PCA components)
INPUT_DATA_FILE = '../final.csv'
# No need for original unprocessed data since final.csv contains both PCA and original columns

OUTPUT_DIR = '.' # Save outputs in the current (dbscan) directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_clustered_data_eps{eps:.2f}_ms{ms}.csv')
NOISE_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_noise_points_eps{eps:.2f}_ms{ms}.csv')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_model_eps{eps:.2f}_ms{ms}.joblib')
RANDOM_STATE = 42
N_DIMS = 2  # Use only first 2 PCA components for clustering (PC1, PC2)
# Rule of thumb for min_samples: >= D+1 or 2*D. Let's use 2*D.
MIN_SAMPLES_PARAM = max(5, 2 * N_DIMS)  # At least 5, but could be higher based on dimensions

warnings.filterwarnings('ignore') # Suppress warnings

def load_pca_data(file_path):
    """Load the PCA data."""
    print(f"Loading PCA data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")
        
        # Extract only the PCA components for clustering
        pca_cols = [col for col in df.columns if col.startswith('PC') and col[2:].isdigit()]
        pca_cols = pca_cols[:N_DIMS]  # Use only the first N_DIMS components
        
        print(f"Using PCA columns: {pca_cols}")
        X = df[pca_cols]
        
        # Also return the full dataframe for later reference
        return X, df
    except FileNotFoundError:
        print(f"Error: Input data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading PCA data: {e}")
        return None, None

def find_eps_for_k_clusters(X, k, min_samples, max_eps=2.0, num_steps=20):
    """Find an eps value that gives approximately k clusters."""
    print(f"\n--- Finding eps value for approximately {k} clusters ---")
    eps_values = np.linspace(0.1, max_eps, num_steps)
    best_eps = None
    best_diff = float('inf')
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Count number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        diff = abs(n_clusters - k)
        
        print(f"  eps={eps:.3f}: found {n_clusters} clusters (difference from target: {diff})")
        
        if diff < best_diff:
            best_diff = diff
            best_eps = eps
    
    print(f"Best eps for {k} clusters: {best_eps:.3f} (gives {k + (best_diff if k > best_diff else -best_diff)} clusters)")
    return best_eps

def find_optimal_eps(X, min_samples, plot_dir):
    """Find Optimal Eps using k-distance graph."""
    print(f"\n--- Finding Optimal Eps using k-distance graph (min_samples={min_samples}) ---")
    if X is None or len(X) == 0:
        print("Error: Input data for find_optimal_eps is invalid.")
        return 1.0 # Return a default value

    nn = NearestNeighbors(n_neighbors=min_samples)
    try:
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
    except Exception as e:
        print(f"Error fitting NearestNeighbors: {e}")
        return 1.0 # Return a default value

    # Get the k-th distances
    if distances.shape[1] < min_samples:
        kth_distances = np.sort(distances[:, -1], axis=0)
    else:
        kth_distances = np.sort(distances[:, min_samples-1], axis=0)

    # Plotting
    os.makedirs(plot_dir, exist_ok=True) # Ensure plot dir exists
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(kth_distances)), kth_distances) # Correct x-axis
    plt.title(f'k-Distance Graph (k = {min_samples})')
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance (eps)')
    plt.grid(True)
    k_dist_path = os.path.join(plot_dir, f'k_distance_graph_k{min_samples}.png')
    try:
        plt.savefig(k_dist_path)
        print(f"k-Distance graph saved to {k_dist_path}")
    except Exception as e:
        print(f"Error saving k-distance plot: {e}")
    plt.close()

    # Heuristic to find elbow
    suggested_eps = 1.0 # Default
    try:
        # Calculate the second derivative (acceleration)
        diffs = np.diff(kth_distances, 2)
        # Find the index of the maximum change in slope (the elbow)
        elbow_index = np.argmax(diffs) + 1 # +1 to adjust index after diff
        if 0 <= elbow_index < len(kth_distances):
             suggested_eps = kth_distances[elbow_index]
             print(f"--> Tentative suggested 'eps' based on max curvature heuristic: {suggested_eps:.4f}")
        else:
             print("Could not reliably find elbow index, using default eps.")
    except Exception as e:
        print(f"Could not automatically suggest eps using curvature: {e}")

    return suggested_eps

def perform_dbscan(X, eps, min_samples):
    """Perform DBSCAN clustering."""
    print(f"\n--- Performing DBSCAN (eps={eps:.4f}, min_samples={min_samples}) ---")
    if X is None or len(X) == 0:
        print("Error: Input data for perform_dbscan is invalid.")
        return None, np.array([]), np.array([]), 0, 0

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    try:
        labels = dbscan.fit_predict(X)
    except Exception as e:
        print(f"Error during DBSCAN fitting: {e}")
        return None, np.array([]), np.array([]), 0, 0

    # Identify core samples
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(dbscan, 'core_sample_indices_') and dbscan.core_sample_indices_ is not None:
         if len(dbscan.core_sample_indices_) > 0 :
              valid_indices = dbscan.core_sample_indices_[dbscan.core_sample_indices_ < len(labels)]
              core_samples_mask[valid_indices] = True
         else:
              print("Warning: DBSCAN found no core samples.")
    else:
        print("Warning: Could not access core_sample_indices_ from DBSCAN model.")

    # Number of clusters in labels, ignoring noise if present.
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')

    # Calculate Silhouette Score (for non-noise points)
    if n_clusters_ > 0:
        non_noise_mask = (labels != -1)
        if np.sum(non_noise_mask) > 1:
            X_filtered = X[non_noise_mask]
            labels_filtered = labels[non_noise_mask]
            if len(set(labels_filtered)) > 1:
                try:
                    silhouette_avg = silhouette_score(X_filtered, labels_filtered)
                    print(f'Average Silhouette Score (excluding noise): {silhouette_avg:.4f}')
                except ValueError as e:
                    print(f"Could not calculate Silhouette Score: {e}")
            else:
                 print("Silhouette Score cannot be calculated: Only one cluster found among non-noise points.")
        else:
             print("Silhouette Score cannot be calculated: Not enough non-noise points found.")
    else:
        print("Silhouette Score cannot be calculated: No clusters found.")

    return dbscan, labels, core_samples_mask, n_clusters_, n_noise_

def visualize_clusters(X, labels, core_samples_mask, plot_dir, filename_suffix, algorithm_name='DBSCAN', 
                       n_clusters=None, n_noise=None, eps=None, min_samples=None):
    """Visualize the clusters to match the reference KMeans plot style."""
    print(f"\n--- Visualizing Clusters ({filename_suffix}) ---")
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    unique_labels = set(labels)

    # Create a color map using viridis (same as reference)
    num_colors_needed = max(1, n_clusters) if n_clusters is not None else len([l for l in unique_labels if l != -1])
    num_colors_needed = max(1, num_colors_needed)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors_needed))
    color_map = {label: colors[i] for i, label in enumerate(sorted([l for l in unique_labels if l != -1]))}
    color_map[-1] = (0, 0, 0, 1)  # Black for noise
    
    # For scatter plot approach (simpler and more like the reference)
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    # Main scatter plot with larger points (s=150 like in your reference)
    scatter = plt.scatter(X_values[:, 0], X_values[:, 1], 
                         c=labels, 
                         cmap='viridis',
                         s=150,  # Matched to your reference
                         alpha=0.7,  # Same alpha as reference
                         edgecolor='w',  # White edge for better visibility
                         linewidth=0.5)
    
    # Add title and labels
    if filename_suffix == "k3_fixed":
        plt.title(f'DBSCAN Clustering (k≈3)\neps={eps:.3f}, min_samples={min_samples}', fontsize=14)
    else:
        plt.title(f'DBSCAN Clustering\neps={eps:.3f}, min_samples={min_samples}', fontsize=14)
    
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add grid with the same style as reference
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits to match the reference plot if needed
    plt.xlim(-4.5, 6.5)
    plt.ylim(-4.5, 4.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Cluster')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    pca_plot_path = os.path.join(plot_dir, f'dbscan_clusters_{filename_suffix}.png')
    try:
        plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to {pca_plot_path}")
    except Exception as e:
        print(f"Error saving visualization plot: {e}")
    plt.close()

def main():
    # Ensure output directories exist
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load PCA Data
    X, full_df = load_pca_data(INPUT_DATA_FILE)
    if X is None:
        return  # Exit if data loading failed
    
    # STEP 1: Run DBSCAN with k=3 clusters (approximately)
    target_k = 3
    eps_for_k3 = find_eps_for_k_clusters(X, k=target_k, min_samples=MIN_SAMPLES_PARAM)
    
    # Perform DBSCAN with the eps that gives ~3 clusters
    model_k3, labels_k3, core_mask_k3, n_clusters_k3, n_noise_k3 = perform_dbscan(
        X, eps=eps_for_k3, min_samples=MIN_SAMPLES_PARAM)
    
    # Save the k=3 model to joblib file
    if model_k3 is not None:
        k3_model_file = MODEL_FILE_TEMPLATE.format(eps=eps_for_k3, ms=MIN_SAMPLES_PARAM)
        try:
            joblib.dump(model_k3, k3_model_file)
            print(f"DBSCAN model (k≈3) saved to {k3_model_file}")
        except Exception as e:
            print(f"Error saving DBSCAN model (k≈3): {e}")
    
    # Visualize the k=3 clustering
    visualize_clusters(X, labels_k3, core_mask_k3, plot_dir=PLOT_DIR, 
                      filename_suffix="k3_fixed", n_clusters=n_clusters_k3, n_noise=n_noise_k3, 
                      eps=eps_for_k3, min_samples=MIN_SAMPLES_PARAM)
    
    # Save k=3 clustering results
    if model_k3 is not None:
        full_df_k3 = full_df.copy()
        full_df_k3['Cluster'] = labels_k3
        full_df_k3.to_csv(CLUSTERED_DATA_FILE_TEMPLATE.format(eps=eps_for_k3, ms=MIN_SAMPLES_PARAM), index=False)
        print(f"Saved k=3 clustered data to {CLUSTERED_DATA_FILE_TEMPLATE.format(eps=eps_for_k3, ms=MIN_SAMPLES_PARAM)}")
    
    # STEP 2: Find optimal eps using k-distance graph
    suggested_eps = find_optimal_eps(X, min_samples=MIN_SAMPLES_PARAM, plot_dir=PLOT_DIR)
    print(f"Optimal eps from k-distance graph: {suggested_eps:.4f}")
    
    # Perform DBSCAN with the optimal eps
    model_opt, labels_opt, core_mask_opt, n_clusters_opt, n_noise_opt = perform_dbscan(
        X, eps=suggested_eps, min_samples=MIN_SAMPLES_PARAM)
    
    # Save the optimal model to joblib file
    if model_opt is not None:
        opt_model_file = MODEL_FILE_TEMPLATE.format(eps=suggested_eps, ms=MIN_SAMPLES_PARAM)
        try:
            joblib.dump(model_opt, opt_model_file)
            print(f"DBSCAN model (optimal) saved to {opt_model_file}")
        except Exception as e:
            print(f"Error saving DBSCAN model (optimal): {e}")
    
    # Visualize the optimal clustering
    visualize_clusters(X, labels_opt, core_mask_opt, plot_dir=PLOT_DIR, 
                      filename_suffix="optimal", n_clusters=n_clusters_opt, n_noise=n_noise_opt, 
                      eps=suggested_eps, min_samples=MIN_SAMPLES_PARAM)
    
    # Save optimal clustering results
    if model_opt is not None:
        full_df_opt = full_df.copy()
        full_df_opt['Cluster'] = labels_opt
        full_df_opt.to_csv(CLUSTERED_DATA_FILE_TEMPLATE.format(eps=suggested_eps, ms=MIN_SAMPLES_PARAM), index=False)
        print(f"Saved optimal clustered data to {CLUSTERED_DATA_FILE_TEMPLATE.format(eps=suggested_eps, ms=MIN_SAMPLES_PARAM)}")
    
    print("\nDBSCAN clustering process completed! Two plots have been generated:")
    print(f"1. Fixed k=3 clusters (eps={eps_for_k3:.4f}) - Model saved to {MODEL_FILE_TEMPLATE.format(eps=eps_for_k3, ms=MIN_SAMPLES_PARAM)}")
    print(f"2. Optimal clustering (eps={suggested_eps:.4f}) - Model saved to {MODEL_FILE_TEMPLATE.format(eps=suggested_eps, ms=MIN_SAMPLES_PARAM)}")

if __name__ == "__main__":
    main()