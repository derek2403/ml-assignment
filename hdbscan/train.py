# Path: hdbscan/train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Keep removed
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.preprocessing import StandardScaler # REMOVED - Input is PCA data
# from sklearn.decomposition import PCA # REMOVED - Input is PCA data
import joblib
import os
import warnings

# === Configuration ===
# Input data file - updated to final.csv
INPUT_DATA_FILE = '../final.csv'
# Original unprocessed data file (needed for merging final results)
ORIGINAL_UNPROCESSED_DATA_FILE = '../../dataset.csv' # Adjust path if needed

OUTPUT_DIR = '.' # Output in the current directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hdbscan_model_{suffix}.joblib')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hdbscan_clustered_data_{suffix}.csv')
RANDOM_STATE = 42

# --- HDBSCAN Parameters (tune these) ---
MIN_CLUSTER_SIZE = 15  # How many points minimum to form a cluster? Try varying (e.g., 5, 10, 20)
MIN_SAMPLES = None     # How many samples in neighborhood to be core point? None often works well (defaults related to min_cluster_size). Try setting (e.g., 5, 10).
CLUSTER_SELECTION_EPSILON = 0.0 # Threshold for merging clusters. 0.0 is often default. Try small values if needed.
METRIC = 'euclidean' # Distance metric

# --- Ensure output directories exist ---
os.makedirs(PLOT_DIR, exist_ok=True)

# Set random seeds for reproducibility (less relevant for core HDBSCAN algo)
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore') # Suppress warnings

# === Helper Functions ===

def load_pca_data(file_path):
    """Load the PCA data."""
    print(f"Loading PCA data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")
        
        # Extract only the PCA components for clustering
        pca_cols = [col for col in df.columns if col.startswith('PC') and col[2:].isdigit()]
        print(f"Using PCA columns: {pca_cols}")
        X = df[pca_cols]
        
        # Return numpy array for HDBSCAN, column names, and full dataframe
        return X.values, pca_cols, df
    except FileNotFoundError:
        print(f"Error: Input data file not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading PCA data: {e}")
        return None, None, None

def load_original_data_for_reference(file_path):
    """Loads original data just to get reference columns like ID, Name etc."""
    print(f"Loading original (unprocessed) data for reference from {file_path}...")
    try:
        # Use latin1 encoding as determined before
        df_orig = pd.read_csv(file_path, encoding='latin1')
        # Clean column names for consistency
        df_orig.columns = df_orig.columns.str.replace('\n', '', regex=False).str.replace('?', '', regex=False).str.strip()
        df_orig = df_orig.rename(columns={'STNCode': 'STN_Code'}) # Rename ID column if needed
        print(f"Original reference data loaded with shape: {df_orig.shape}")
        # Keep only identifier/reference columns
        ref_cols = ['STN_Code', 'Name of Monitoring Location', 'Type Water Body', 'State Name'] # Adjust if needed
        # Filter out columns that don't exist in the original file
        ref_cols_exist = [col for col in ref_cols if col in df_orig.columns]
        return df_orig[ref_cols_exist]
    except FileNotFoundError:
        print(f"Error: Original data file not found at {file_path}")
        print("Cannot add reference info to clustered output.")
        return None
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None

# def scale_data(X): # REMOVED - No scaling needed for PCA input

def find_params_for_k_clusters(X, target_k=3, min_size_range=range(5, 30), min_samples_range=range(1, 15)):
    """Find HDBSCAN parameters that give approximately target_k clusters."""
    print(f"\n--- Finding HDBSCAN parameters for approximately {target_k} clusters ---")
    
    best_params = {
        'min_cluster_size': 15,  # Default
        'min_samples': None,     # Default
        'cluster_selection_epsilon': 0.0  # Default
    }
    best_diff = float('inf')
    
    # Try different combinations of parameters
    for min_cluster_size in min_size_range:
        for min_samples in min_samples_range:
            # Skip invalid combinations
            if min_samples > min_cluster_size:
                continue
                
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=0.0,
                    core_dist_n_jobs=-1
                )
                clusterer.fit(X)
                labels = clusterer.labels_
                
                # Count unique clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                diff = abs(n_clusters - target_k)
                
                print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}: {n_clusters} clusters (diff: {diff})")
                
                if diff < best_diff:
                    best_diff = diff
                    best_params['min_cluster_size'] = min_cluster_size
                    best_params['min_samples'] = min_samples
                    
                # If exact match, we can stop
                if diff == 0:
                    break
                    
            except Exception as e:
                print(f"  Error with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}")
                
        # If exact match, we can stop
        if best_diff == 0:
            break
            
    print(f"Best parameters for {target_k} clusters: min_cluster_size={best_params['min_cluster_size']}, min_samples={best_params['min_samples']}")
    return best_params

def perform_hdbscan(X, min_cluster_size=15, min_samples=None, cluster_selection_epsilon=0.0, metric='euclidean'):
    """Perform HDBSCAN clustering."""
    print("\n--- Performing HDBSCAN Clustering ---")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='{metric}', cluster_selection_epsilon={cluster_selection_epsilon}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1
    )

    # Fit on the data
    clusterer.fit(X)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    outlier_scores = clusterer.outlier_scores_

    # --- Basic Cluster Statistics ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nHDBSCAN completed.")
    print(f"  Number of clusters found: {n_clusters}")
    print(f"  Number of noise points: {n_noise} ({(n_noise / len(labels) * 100):.1f}%)")
    print(f"  Total points: {len(labels)}")

    # --- Evaluation (on core points) ---
    if n_clusters > 1:
        core_samples_mask = (labels != -1)
        X_core = X[core_samples_mask]
        labels_core = labels[core_samples_mask]

        if len(X_core) > 1 and len(set(labels_core)) > 1:
            try:
                silhouette_avg = silhouette_score(X_core, labels_core)
                print(f"  Average Silhouette Score (core points): {silhouette_avg:.4f}")
            except ValueError as e:
                print(f"  Could not calculate Silhouette Score: {e}")

            try:
                davies_bouldin_avg = davies_bouldin_score(X_core, labels_core)
                print(f"  Davies-Bouldin Score (core points): {davies_bouldin_avg:.4f}")
            except ValueError as e:
                print(f"  Could not calculate Davies-Bouldin Score: {e}")
        else:
            print("  Not enough core points or clusters (>1) to calculate evaluation scores.")
    else:
        print("  Not enough clusters found (>1) to calculate evaluation scores.")

    return clusterer, labels, probabilities, outlier_scores

def visualize_clusters(X, labels, plot_dir, filename_suffix, algorithm_name='HDBSCAN'):
    """Visualize the clusters using the first 2 components with matched style."""
    print(f"\n--- Visualizing Clusters ({filename_suffix}) ---")
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    # Extract the first two PCA components for plotting
    X_plot = X[:, :2]
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create color map using viridis
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, n_clusters)))
    color_map = {label: colors[i] for i, label in enumerate(sorted([l for l in unique_labels if l != -1]))}
    color_map[-1] = (0.3, 0.3, 0.3, 1)  # Gray for noise
    
    # Main scatter plot
    scatter = plt.scatter(
        X_plot[:, 0], X_plot[:, 1], 
        c=[color_map.get(label, (0.5, 0.5, 0.5, 1)) for label in labels],
        s=150,  # Matched to reference
        alpha=0.7,  # Same alpha as reference
        edgecolor='w',  # White edge for better visibility
        linewidth=0.5
    )
    
    # Add title and labels
    plt.title(f'{algorithm_name} Clustering ({n_clusters} clusters found)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add grid with the same style as reference
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits to match the reference plot
    plt.xlim(-4.5, 6.5)
    plt.ylim(-4.5, 4.5)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, f'hdbscan_clusters_{filename_suffix}.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving visualization plot: {e}")
    plt.close()

# === Main Execution Logic ===
def main():
    # --- Load Data ---
    X_pca_values, pca_feature_names, full_df = load_pca_data(INPUT_DATA_FILE)
    if X_pca_values is None:
        return # Exit if loading failed

    df_original_ref = load_original_data_for_reference(ORIGINAL_UNPROCESSED_DATA_FILE)

    # --- Scaling Removed ---
    # X_scaled, scaler = scale_data(X_original) # No longer needed

    # --- STEP 1: Find parameters for k≈3 clusters ---
    target_k = 3
    print(f"\n=== Finding HDBSCAN parameters for k≈{target_k} ===")
    k3_params = find_params_for_k_clusters(X_pca_values, target_k=target_k)
    
    # --- STEP 2: Run HDBSCAN with k≈3 parameters ---
    print(f"\n=== Running HDBSCAN with parameters for k≈{target_k} ===")
    hdbscan_k3, labels_k3, probs_k3, outlier_scores_k3 = perform_hdbscan(
        X_pca_values, 
        min_cluster_size=k3_params['min_cluster_size'], 
        min_samples=k3_params['min_samples'],
        cluster_selection_epsilon=k3_params['cluster_selection_epsilon']
    )
    
    # Save k≈3 model and results
    k3_suffix = "k3_approx"
    k3_model_file = MODEL_FILE_TEMPLATE.format(suffix=k3_suffix)
    try:
        joblib.dump(hdbscan_k3, k3_model_file)
        print(f"HDBSCAN model (k≈3) saved to {k3_model_file}")
    except Exception as e:
        print(f"Error saving HDBSCAN model (k≈3): {e}")
    
    # Visualize k≈3 clustering
    visualize_clusters(X_pca_values, labels_k3, plot_dir=PLOT_DIR, filename_suffix=k3_suffix)
    
    # Save k≈3 clustering results
    k3_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=k3_suffix)
    df_k3_output = full_df.copy()
    df_k3_output['HDBSCAN_Cluster'] = labels_k3
    df_k3_output['HDBSCAN_Prob'] = probs_k3
    df_k3_output['HDBSCAN_OutlierScore'] = outlier_scores_k3
    try:
        df_k3_output.to_csv(k3_output_file, index=False)
        print(f"Clustered data for k≈3 saved to {k3_output_file}")
    except Exception as e:
        print(f"Error saving clustered data for k≈3: {e}")

    # --- STEP 3: Run HDBSCAN with optimized parameters ---
    print("\n=== Running HDBSCAN with optimized parameters ===")
    # These are generally good defaults for HDBSCAN
    opt_min_cluster_size = 15  # Default in hdbscan package
    opt_min_samples = 5        # A common choice
    
    hdbscan_opt, labels_opt, probs_opt, outlier_scores_opt = perform_hdbscan(
        X_pca_values, 
        min_cluster_size=opt_min_cluster_size, 
        min_samples=opt_min_samples
    )
    
    # Check if optimal is different from k≈3
    n_clusters_k3 = len(set(labels_k3)) - (1 if -1 in labels_k3 else 0)
    n_clusters_opt = len(set(labels_opt)) - (1 if -1 in labels_opt else 0)
    
    if n_clusters_opt == n_clusters_k3 and np.array_equal(labels_k3, labels_opt):
        print(f"Optimized parameters produced the same clustering as k≈3 parameters. Skipping duplicate processing.")
    else:
        # Save optimal model and results
        opt_suffix = "optimal"
        opt_model_file = MODEL_FILE_TEMPLATE.format(suffix=opt_suffix)
        try:
            joblib.dump(hdbscan_opt, opt_model_file)
            print(f"HDBSCAN model (optimal) saved to {opt_model_file}")
        except Exception as e:
            print(f"Error saving HDBSCAN model (optimal): {e}")
        
        # Visualize optimal clustering
        visualize_clusters(X_pca_values, labels_opt, plot_dir=PLOT_DIR, filename_suffix=opt_suffix)
        
        # Save optimal clustering results
        opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=opt_suffix)
        df_opt_output = full_df.copy()
        df_opt_output['HDBSCAN_Cluster'] = labels_opt
        df_opt_output['HDBSCAN_Prob'] = probs_opt
        df_opt_output['HDBSCAN_OutlierScore'] = outlier_scores_opt
        try:
            df_opt_output.to_csv(opt_output_file, index=False)
            print(f"Clustered data for optimal parameters saved to {opt_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for optimal parameters: {e}")

    print("\nHDBSCAN clustering process completed!")
    print("Summary:")
    print(f"1. k≈3 clustering: {n_clusters_k3} clusters, parameters: min_cluster_size={k3_params['min_cluster_size']}, min_samples={k3_params['min_samples']}")
    print(f"2. Optimal clustering: {n_clusters_opt} clusters, parameters: min_cluster_size={opt_min_cluster_size}, min_samples={opt_min_samples}")


if __name__ == "__main__":
    main()