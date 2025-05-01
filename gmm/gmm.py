# Path: gmm/gmm.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Keep removed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler # Removed - Input is already processed
from sklearn.decomposition import PCA # Keep for potentially selecting components if needed later
import joblib
import os
import warnings
from mpl_toolkits.mplot3d import Axes3D # Keep if 3D plot is desired

# --- Configuration ---
# Input data file - updated to final.csv
INPUT_DATA_FILE = '../final.csv'
# Original unprocessed data file (not needed if final.csv has all needed columns)

OUTPUT_DIR = '.' # Save outputs in the current (gmm) directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'gmm_model_k{k}.joblib')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'gmm_clustered_data_k{k}.csv')

MAX_COMPONENTS = 15 # Max number of GMM components to test
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
        X = df[pca_cols]
        
        # Also return the full dataframe for later reference
        return X, df
    except FileNotFoundError:
        print(f"Error: Input data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading PCA data: {e}")
        return None, None

def find_optimal_components(X, max_components=15, plot_dir='plots'):
    """Find the optimal number of GMM components using BIC and AIC."""
    print("\n--- Finding Optimal Number of Components (BIC/AIC) ---")
    # Ensure X is a numpy array for GMM fitting
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    n_components_range = range(2, max_components + 1)
    bics = []
    aics = []

    for n_components in n_components_range:
        print(f"Calculating for n_components={n_components}...")
        try:
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type='full',
                                  random_state=RANDOM_STATE,
                                  n_init=5,
                                  init_params='kmeans')
            gmm.fit(X_np)
            bics.append(gmm.bic(X_np))
            aics.append(gmm.aic(X_np))
            print(f"  BIC: {bics[-1]:.2f}, AIC: {aics[-1]:.2f}")
        except Exception as e:
             print(f"  Error calculating BIC/AIC for {n_components} components: {e}")
             bics.append(np.inf)
             aics.append(np.inf)

    # Filter out failed calculations
    valid_indices = [(c, bic, aic) for c, bic, aic in zip(n_components_range, bics, aics) if not (np.isinf(bic) or np.isinf(aic))]
    if not valid_indices:
        print("Error: Could not calculate BIC/AIC for any component count. Defaulting to k=3.")
        return 3

    valid_k_range, valid_bics, valid_aics = zip(*valid_indices)

    # --- Plotting BIC and AIC ---
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_k_range, valid_bics, marker='o', linestyle='--', label='BIC')
    plt.plot(valid_k_range, valid_aics, marker='s', linestyle=':', label='AIC')
    plt.title('GMM BIC and AIC Scores')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Information Criterion Score (Lower is better)')
    plt.xticks(list(n_components_range))
    plt.legend()
    plt.grid(True)
    bic_aic_plot_path = os.path.join(plot_dir, 'gmm_bic_aic.png')
    try:
        plt.savefig(bic_aic_plot_path)
        print(f"BIC/AIC plot saved to {bic_aic_plot_path}")
    except Exception as e:
         print(f"Error saving BIC/AIC plot: {e}")
    plt.close()

    # Optimal k: Choose the minimum BIC
    optimal_k = valid_k_range[np.argmin(valid_bics)]
    print(f"Optimal number of components based on BIC: {optimal_k}")

    return optimal_k

def perform_gmm(X, n_components, random_state=42):
    """Perform GMM clustering with the specified number of components."""
    print(f"\n--- Performing GMM Clustering (n_components={n_components}) ---")
    # Ensure X is a numpy array for GMM fitting
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=random_state,
                          n_init=10,
                          init_params='kmeans')
    try:
        gmm.fit(X_np)
        labels = gmm.predict(X_np)
        probs = gmm.predict_proba(X_np)
    except Exception as e:
        print(f"Error during GMM fitting: {e}")
        return None, np.array([]), np.array([])

    silhouette_avg = np.nan
    if n_components > 1:
        try:
            silhouette_avg = silhouette_score(X_np, labels)
            print(f"GMM completed.")
            print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        except ValueError as e:
            print(f"Could not calculate Silhouette Score: {e}")
    else:
        print("GMM completed with k=1. Silhouette score not applicable.")

    return gmm, labels, probs

def visualize_clusters(X_orig, labels, plot_dir='plots', filename_suffix='', algorithm_name='GMM', k=None):
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
                         s=150,  # Matched to reference
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
    plot_path = os.path.join(plot_dir, f'gmm_clusters_{filename_suffix}.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving visualization plot: {e}")
    plt.close()

def main():
    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load PCA Data ---
    X, full_df = load_pca_data(INPUT_DATA_FILE)
    if X is None:
        return  # Exit if data loading failed

    # --- STEP 1: Run GMM with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Running GMM with fixed k={fixed_k} ===")
    gmm_fixed, labels_fixed, probs_fixed = perform_gmm(X, n_components=fixed_k, random_state=RANDOM_STATE)
    
    if gmm_fixed is None:
        print(f"GMM with k={fixed_k} failed. Continuing to optimal k...")
    else:
        # Save model results
        fixed_model_file = MODEL_FILE_TEMPLATE.format(k=fixed_k)
        try:
            joblib.dump(gmm_fixed, fixed_model_file)
            print(f"GMM model for k={fixed_k} saved to {fixed_model_file}")
        except Exception as e:
            print(f"Error saving GMM model for k={fixed_k}: {e}")
        
        # Visualize the fixed k results
        visualize_clusters(X, labels_fixed, plot_dir=PLOT_DIR, 
                          filename_suffix=f"k{fixed_k}", k=fixed_k)
        
        # Save clustered data
        fixed_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(k=fixed_k)
        df_fixed_output = full_df.copy()
        df_fixed_output['Cluster_GMM'] = labels_fixed
        for i in range(fixed_k):
            df_fixed_output[f'Cluster_{i}_Prob'] = probs_fixed[:, i]
        try:
            df_fixed_output.to_csv(fixed_output_file, index=False)
            print(f"Clustered data for k={fixed_k} saved to {fixed_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Components ---
    print("\n=== Finding Optimal Number of Components ===")
    optimal_k = find_optimal_components(X, max_components=MAX_COMPONENTS, plot_dir=PLOT_DIR)

    # Skip this step if optimal_k is the same as fixed_k
    if optimal_k == fixed_k:
        print(f"Optimal number of components ({optimal_k}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Run GMM with Optimal k ---
        print(f"\n=== Running GMM with optimal k={optimal_k} ===")
        gmm_opt, labels_opt, probs_opt = perform_gmm(X, n_components=optimal_k, random_state=RANDOM_STATE)
        
        if gmm_opt is None:
            print(f"GMM with optimal k={optimal_k} failed.")
        else:
            # Save model results
            opt_model_file = MODEL_FILE_TEMPLATE.format(k=optimal_k)
            try:
                joblib.dump(gmm_opt, opt_model_file)
                print(f"GMM model for optimal k={optimal_k} saved to {opt_model_file}")
            except Exception as e:
                print(f"Error saving GMM model for optimal k={optimal_k}: {e}")
            
            # Visualize the optimal k results
            visualize_clusters(X, labels_opt, plot_dir=PLOT_DIR, 
                              filename_suffix=f"optimal_k{optimal_k}", k=optimal_k)
            
            # Save clustered data
            opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(k=optimal_k)
            df_opt_output = full_df.copy()
            df_opt_output['Cluster_GMM'] = labels_opt
            for i in range(optimal_k):
                df_opt_output[f'Cluster_{i}_Prob'] = probs_opt[:, i]
            try:
                df_opt_output.to_csv(opt_output_file, index=False)
                print(f"Clustered data for optimal k={optimal_k} saved to {opt_output_file}")
            except Exception as e:
                print(f"Error saving clustered data for optimal k={optimal_k}: {e}")

    print("\nGMM clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_k} clustering")

if __name__ == "__main__":
    main()