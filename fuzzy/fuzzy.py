# Path: fuzzy/fuzzy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
import joblib
import os
import warnings
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Input data file (PREPROCESSED + PCA REDUCED) relative to this script
INPUT_DATA_FILE = '../final.csv'  # Updated to use your final.csv
# Original unprocessed data file (needed for merging final results)
ORIGINAL_UNPROCESSED_DATA_FILE = '../dataset.csv'

OUTPUT_DIR = '.'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'fuzzy_model_results_k{k}.joblib')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'fuzzy_clustered_data_k{k}.csv')

MAX_CLUSTERS = 10
FUZZINESS_M = 2.0
RANDOM_STATE = 42

warnings.filterwarnings('ignore')

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
        
        # Extract data in different formats needed for FCM
        X_orig = df[pca_cols]  # samples x features
        X_fcm = X_orig.T  # features x samples for skfuzzy
        
        # Return both formats and the full dataframe
        return X_orig, X_fcm, df
    except FileNotFoundError:
        print(f"Error: Input data file not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading PCA data: {e}")
        return None, None, None

def find_optimal_clusters_fcm(X_fcm, max_clusters=10, m=2, plot_dir='plots'):
    """Find optimal number of clusters (c) for FCM using validity indices."""
    print(f"\n--- Finding Optimal Clusters (c) for FCM (m={m}) ---")
    fpcs = []
    pes = []
    cluster_range = range(2, max_clusters + 1)

    # Data should be features x samples for skfuzzy.cmeans
    data = X_fcm.values if isinstance(X_fcm, pd.DataFrame) else X_fcm

    for ncenters in cluster_range:
        print(f"Calculating for c={ncenters}...")
        try:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data, ncenters, m, error=0.005, maxiter=1000, init=None, seed=RANDOM_STATE)

            fpcs.append(fpc)
            # Partition Entropy calculation
            pe_val = -np.sum(u * np.log2(u + 1e-9)) / data.shape[1]
            pes.append(pe_val)
            print(f"  FPC: {fpc:.4f}, PE: {pe_val:.4f}")

        except Exception as e:
            print(f"  Error during FCM validity calculation for c={ncenters}: {e}. Appending NaN.")
            fpcs.append(np.nan)
            pes.append(np.nan)

    # Filter out NaN values if any occurred
    valid_indices = [(c, fpc, pe) for c, fpc, pe in zip(cluster_range, fpcs, pes) if not (np.isnan(fpc) or np.isnan(pe))]
    if not valid_indices:
        print("Error: Could not calculate validity indices for any cluster count. Defaulting to c=3.")
        return 3

    valid_c_range, valid_fpc_scores, valid_pe_scores = zip(*valid_indices)

    # --- Plotting Validity Indices ---
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (c)')
    ax1.set_ylabel('FPC (Higher is better)', color=color)
    ax1.plot(valid_c_range, valid_fpc_scores, marker='o', linestyle='--', color=color, label='FPC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('PE (Lower is better)', color=color)
    ax2.plot(valid_c_range, valid_pe_scores, marker='s', linestyle=':', color=color, label='PE')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('FCM Cluster Validity Indices')
    plt.xticks(list(cluster_range))

    validity_plot_path = os.path.join(plot_dir, 'fcm_validity_indices.png')
    try:
        plt.savefig(validity_plot_path)
        print(f"Validity indices plot saved to {validity_plot_path}")
    except Exception as e:
        print(f"Error saving validity plot: {e}")
    plt.close()

    # Determine optimal clusters from FPC
    optimal_c_fpc = valid_c_range[np.argmax(valid_fpc_scores)]
    optimal_c_pe = valid_c_range[np.argmin(valid_pe_scores)]
    print(f"Optimal c suggested by max FPC: {optimal_c_fpc}")
    print(f"Optimal c suggested by min PE: {optimal_c_pe}")
    
    # Default to FPC's suggestion
    optimal_c = optimal_c_fpc
    print(f"Using optimal c = {optimal_c} (based on max FPC).")

    return optimal_c

def perform_fcm(X_fcm, n_clusters, m=2, error=0.005, maxiter=1000, random_seed=42):
    """Perform Fuzzy C-Means clustering."""
    print(f"\n--- Performing Fuzzy C-Means (c={n_clusters}, m={m}) ---")
    # Data should be features x samples for skfuzzy.cmeans
    data = X_fcm.values if isinstance(X_fcm, pd.DataFrame) else X_fcm

    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, n_clusters, m, error=error, maxiter=maxiter, init=None, seed=random_seed)
    except Exception as e:
        print(f"Error during FCM execution: {e}")
        return None

    # Calculate Partition Entropy (PE)
    pe = -np.sum(u * np.log2(u + 1e-9)) / data.shape[1]

    # Calculate Xie-Beni (XB) Index
    X_orig_shape = data.T
    xb = calculate_xie_beni(X_orig_shape, cntr, u, m)

    print(f"FCM completed.")
    print(f"  FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
    print(f"  PE (Partition Entropy): {pe:.4f}")
    print(f"  XB (Xie-Beni Index - Lower is better): {xb:.4f}")

    # Hard cluster assignments
    labels = np.argmax(u, axis=0)
    silhouette_avg = np.nan
    if n_clusters > 1:
        try:
            silhouette_avg = silhouette_score(X_orig_shape, labels)
            print(f"  Silhouette Score (hard labels): {silhouette_avg:.4f}")
        except Exception as e:
            print(f"  Could not calculate Silhouette Score: {e}")
    else:
        print("  Only one cluster assigned. Silhouette score not applicable.")

    # Package results
    fcm_results = {
        'centers': cntr,
        'membership': u,
        'fpc': fpc,
        'pe': pe,
        'xb': xb,
        'hard_labels': labels,
        'silhouette': silhouette_avg,
        'n_clusters': n_clusters,
        'fuzziness_m': m
    }

    return fcm_results

def calculate_xie_beni(data_points, centers, membership, m):
    """Calculate Xie-Beni Index."""
    n_samples = data_points.shape[0]
    n_clusters = centers.shape[0]

    term1_sum = 0
    # Ensure data_points is numpy array for calculations
    data_points_np = data_points.values if isinstance(data_points, pd.DataFrame) else data_points
    centers_np = centers.values if isinstance(centers, pd.DataFrame) else centers

    for k in range(n_clusters):
        for i in range(n_samples):
            term1_sum += (membership[k, i] ** m) * (np.linalg.norm(data_points_np[i] - centers_np[k]) ** 2)

    min_center_dist_sq = np.inf
    for k1 in range(n_clusters):
        for k2 in range(k1 + 1, n_clusters):
            dist_sq = np.linalg.norm(centers_np[k1] - centers_np[k2]) ** 2
            if dist_sq < min_center_dist_sq:
                min_center_dist_sq = dist_sq

    if min_center_dist_sq == 0 or np.isinf(min_center_dist_sq):
        print("Warning: Minimum center distance squared is zero or infinity. XB index may be invalid.")
        return np.inf

    xb = term1_sum / (n_samples * min_center_dist_sq)
    return xb

def visualize_clusters(X_orig, labels, plot_dir='plots', filename_suffix='', algorithm_name='FCM', k=None):
    """Visualize the clusters with matched style to reference plots."""
    print(f"\n--- Visualizing Clusters ({filename_suffix}) ---")
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    # Extract the first two PCA components for plotting
    X_values = X_orig.values if isinstance(X_orig, pd.DataFrame) else X_orig
    X_plot = X_values[:, :2]  # First two components
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels)
    
    # Create color map using viridis (same as reference)
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    
    # Main scatter plot with larger points (s=150 like in reference)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], 
                         c=labels, 
                         cmap='viridis',
                         s=200,  # Matched to reference
                         alpha=0.7,  # Same alpha as reference
                         edgecolor='w',  # White edge for better visibility
                         linewidth=0.5)
    
    # Add title and labels
    plt.title(f'{algorithm_name} Clustering (k={k})', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add grid with the same style as reference
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits to match the reference plot
    plt.xlim(-4.5, 6.5)
    plt.ylim(-4.5, 4.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Cluster')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, f'fcm_clusters_{filename_suffix}.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving visualization plot: {e}")
    plt.close()

def visualize_fuzzy_partition(X_orig, u, labels, plot_dir='plots', filename_suffix='', algorithm_name='FCM Fuzzy', k=None):
    """Visualize fuzzy partition with membership certainty as alpha."""
    print(f"\n--- Visualizing Fuzzy Partition ({filename_suffix}) ---")
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    # Extract the first two PCA components for plotting
    X_values = X_orig.values if isinstance(X_orig, pd.DataFrame) else X_orig
    X_plot = X_values[:, :2]  # First two components
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels)
    
    # Create color map using viridis (same as reference)
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    
    # Calculate max membership for each point (certainty)
    max_membership = np.max(u, axis=0)
    
    # Plot points colored by hard label, alpha by certainty
    for i, cluster_label in enumerate(sorted(unique_labels)):
        mask = (labels == cluster_label)
        if np.sum(mask) == 0: continue
        
        cluster_points = X_plot[mask]
        cluster_certainty = max_membership[mask]
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[colors[i]] * len(cluster_points),
                   alpha=cluster_certainty * 0.5 + 0.2,  # Scale alpha for visibility
                   s=200,  # Match reference size
                   edgecolor='w',
                   linewidth=0.5,
                   label=f'Cluster {cluster_label}')
    
    # Add title and labels
    plt.title(f'{algorithm_name} Clustering (k={k}, Alpha=Certainty)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add grid with the same style as reference
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits to match the reference plot
    plt.xlim(-4.5, 6.5)
    plt.ylim(-4.5, 4.5)
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, f'fcm_fuzzy_{filename_suffix}.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Fuzzy partition plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving fuzzy visualization plot: {e}")
    plt.close()

def main():
    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load PCA Data ---
    X_orig, X_fcm, full_df = load_pca_data(INPUT_DATA_FILE)
    if X_orig is None or X_fcm is None:
        return  # Exit if data loading failed

    # --- STEP 1: Run FCM with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Running FCM with fixed k={fixed_k} ===")
    fcm_results_fixed = perform_fcm(X_fcm, n_clusters=fixed_k, m=FUZZINESS_M, random_seed=RANDOM_STATE)
    
    if fcm_results_fixed is None:
        print(f"FCM with k={fixed_k} failed. Continuing to optimal k...")
    else:
        # Save model results
        fixed_model_file = MODEL_FILE_TEMPLATE.format(k=fixed_k)
        try:
            joblib.dump(fcm_results_fixed, fixed_model_file)
            print(f"FCM results for k={fixed_k} saved to {fixed_model_file}")
        except Exception as e:
            print(f"Error saving FCM results for k={fixed_k}: {e}")
        
        # Visualize the fixed k results
        hard_labels_fixed = fcm_results_fixed['hard_labels']
        membership_u_fixed = fcm_results_fixed['membership']
        visualize_clusters(X_orig, hard_labels_fixed, plot_dir=PLOT_DIR, 
                          filename_suffix=f"k{fixed_k}", k=fixed_k)
        visualize_fuzzy_partition(X_orig, membership_u_fixed, hard_labels_fixed, plot_dir=PLOT_DIR, 
                                 filename_suffix=f"k{fixed_k}", k=fixed_k)
        
        # Save clustered data
        fixed_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(k=fixed_k)
        df_fixed_output = full_df.copy()
        df_fixed_output['Cluster_FCM'] = hard_labels_fixed
        for i in range(fixed_k):
            df_fixed_output[f'Cluster_{i}_Membership'] = membership_u_fixed[i, :]
        try:
            df_fixed_output.to_csv(fixed_output_file, index=False)
            print(f"Clustered data for k={fixed_k} saved to {fixed_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Clusters ---
    print("\n=== Finding Optimal Number of Clusters ===")
    optimal_c = find_optimal_clusters_fcm(X_fcm, max_clusters=MAX_CLUSTERS, m=FUZZINESS_M, plot_dir=PLOT_DIR)

    # Skip this step if optimal_c is the same as fixed_k
    if optimal_c == fixed_k:
        print(f"Optimal number of clusters ({optimal_c}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Run FCM with Optimal k ---
        print(f"\n=== Running FCM with optimal k={optimal_c} ===")
        fcm_results_opt = perform_fcm(X_fcm, n_clusters=optimal_c, m=FUZZINESS_M, random_seed=RANDOM_STATE)
        
        if fcm_results_opt is None:
            print(f"FCM with optimal k={optimal_c} failed.")
        else:
            # Save model results
            opt_model_file = MODEL_FILE_TEMPLATE.format(k=optimal_c)
            try:
                joblib.dump(fcm_results_opt, opt_model_file)
                print(f"FCM results for optimal k={optimal_c} saved to {opt_model_file}")
            except Exception as e:
                print(f"Error saving FCM results for optimal k={optimal_c}: {e}")
            
            # Visualize the optimal k results
            hard_labels_opt = fcm_results_opt['hard_labels']
            membership_u_opt = fcm_results_opt['membership']
            visualize_clusters(X_orig, hard_labels_opt, plot_dir=PLOT_DIR, 
                              filename_suffix=f"optimal_k{optimal_c}", k=optimal_c)
            visualize_fuzzy_partition(X_orig, membership_u_opt, hard_labels_opt, plot_dir=PLOT_DIR, 
                                     filename_suffix=f"optimal_k{optimal_c}", k=optimal_c)
            
            # Save clustered data
            opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(k=optimal_c)
            df_opt_output = full_df.copy()
            df_opt_output['Cluster_FCM'] = hard_labels_opt
            for i in range(optimal_c):
                df_opt_output[f'Cluster_{i}_Membership'] = membership_u_opt[i, :]
            try:
                df_opt_output.to_csv(opt_output_file, index=False)
                print(f"Clustered data for optimal k={optimal_c} saved to {opt_output_file}")
            except Exception as e:
                print(f"Error saving clustered data for optimal k={optimal_c}: {e}")

    print("\nFuzzy C-Means clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_c} clustering")
    
    print("\nHow to adjust the plot sizes and styles yourself:")
    print("1. To change figure size: Modify 'figsize=(14, 12)' in the visualization functions")
    print("2. To change point size: Modify 's=150' in the scatter plot")
    print("3. To change transparency: Modify 'alpha=0.7' in the scatter plot")
    print("4. To adjust axis limits: Modify 'plt.xlim()' and 'plt.ylim()' values")
    print("5. For grid style: Adjust 'plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)'")

if __name__ == "__main__":
    main()