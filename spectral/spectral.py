# Path: spectral/spectral.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Keep removed
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler # REMOVED - Input is PCA data
# from sklearn.decomposition import PCA # REMOVED - Input is PCA data, only used for visualization
import joblib
import os
import warnings
from mpl_toolkits.mplot3d import Axes3D # For optional 3D plot

# === Configuration ===
# Input data file - updated to final.csv
INPUT_DATA_FILE = '../final.csv'
# Original unprocessed data file (needed for merging final results)
ORIGINAL_UNPROCESSED_DATA_FILE = '../../dataset.csv' # Adjust path if needed

OUTPUT_DIR = '.' # Output in the current directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'spectral_clustered_data_{suffix}.csv')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'spectral_model_{suffix}.joblib')
MAX_K = 10 # Max number of clusters to test (adjust based on computation time)
RANDOM_STATE = 42

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
        
        # Also return the full dataframe for later reference
        return X, df
    except FileNotFoundError:
        print(f"Error: Input data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading PCA data: {e}")
        return None, None

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

def find_optimal_k_silhouette(X_pca_values, max_k=10, plot_dir='plots', random_state=42):
    """Find the optimal number of clusters using Silhouette scores on PCA data."""
    print("\n--- Finding Optimal K using Silhouette Score (on PCA data) ---")
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        print(f"Calculating for k={k}...")
        spectral = SpectralClustering(n_clusters=k,
                                    affinity='nearest_neighbors', # Often faster than 'rbf'
                                    assign_labels='kmeans',
                                    random_state=random_state,
                                    n_init=10, # For the kmeans step
                                    n_jobs=-1 # Use all available processors
                                    )
        try:
            # Fit on the PCA data
            labels = spectral.fit_predict(X_pca_values)

            # Check if multiple clusters were actually found
            if len(set(labels)) > 1:
                # Score on the PCA data
                score = silhouette_score(X_pca_values, labels)
                silhouette_scores.append(score)
                print(f"  Silhouette Score: {score:.4f}")
            else:
                print(f"  Warning: Only one cluster found for k={k}. Silhouette score not applicable. Appending NaN.")
                silhouette_scores.append(np.nan)
        except Exception as e:
            print(f"  Error during Spectral Clustering for k={k}: {e}. Appending NaN.")
            silhouette_scores.append(np.nan)

    # Filter out NaN values before plotting and finding max
    valid_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) if not np.isnan(score)]

    if not valid_scores:
        print("Error: Could not calculate silhouette scores for any k. Cannot determine optimal k.")
        return 2 # Default fallback

    valid_k_range, valid_silhouette_scores = zip(*valid_scores)

    # --- Plotting Silhouette Scores ---
    plt.figure(figsize=(10, 6))
    plt.plot(valid_k_range, valid_silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score for Optimal K (Spectral Clustering on PCA data)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(list(valid_k_range))
    plt.grid(True)
    silhouette_plot_path = os.path.join(plot_dir, 'spectral_silhouette_scores_pca.png')
    plt.savefig(silhouette_plot_path)
    plt.close()
    print(f"Silhouette score plot saved to {silhouette_plot_path}")

    # --- Determine Optimal K based on Silhouette Score ---
    optimal_k_silhouette = valid_k_range[np.argmax(valid_silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette}")

    return optimal_k_silhouette

def perform_spectral_clustering(X_pca_values, k, random_state=42):
    """Perform Spectral Clustering with the specified number of clusters on PCA data."""
    print(f"\n--- Performing Spectral Clustering with k={k} (on PCA data) ---")
    model = SpectralClustering(n_clusters=k,
                               affinity='nearest_neighbors',
                               assign_labels='kmeans',
                               random_state=random_state,
                               n_init=10,
                               n_jobs=-1)
    try:
        labels = model.fit_predict(X_pca_values)

        # Check if the expected number of clusters was found
        actual_clusters = len(set(labels))
        if actual_clusters != k:
            print(f"Warning: Found {actual_clusters} clusters, but requested {k}.")

        if actual_clusters > 1:
             silhouette_avg = silhouette_score(X_pca_values, labels)
             print(f"Spectral Clustering completed.")
             print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        else:
            print(f"Spectral Clustering completed, but only found {actual_clusters} cluster(s). Silhouette score not applicable.")
            silhouette_avg = np.nan

        return model, labels, silhouette_avg

    except Exception as e:
        print(f"Error during final Spectral Clustering fit/predict: {e}")
        return None, np.full(X_pca_values.shape[0], -1), np.nan

def visualize_clusters(X, labels, plot_dir='plots', filename_suffix='', algorithm_name='Spectral', k=None):
    """Visualize the clusters with matched style to reference plots."""
    print(f"\n--- Visualizing Clusters ({filename_suffix}) ---")
    
    # Handle error cases
    unique_labels = np.unique(labels)
    if -1 in unique_labels and len(unique_labels) == 1:
        print("Warning: Only noise/error labels found (-1). Cannot visualize clusters properly.")
        return
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    # Extract the first two PCA components for plotting
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    X_plot = X_values[:, :2]  # First two components
    
    # Get unique cluster labels
    if k is None:
        k = len(unique_labels)
        if -1 in unique_labels: 
            k -= 1  # Don't count -1 as a cluster
    
    # Create color map using viridis (same as reference)
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, k)))
    
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
    plot_path = os.path.join(plot_dir, f'spectral_clusters_{filename_suffix}.png')
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
    X_pca_df, full_df = load_pca_data(INPUT_DATA_FILE)
    if X_pca_df is None:
        return  # Exit if data loading failed
    
    X_pca_values = X_pca_df.values  # Use numpy array for clustering

    # --- STEP 1: Run Spectral Clustering with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Running Spectral Clustering with fixed k={fixed_k} ===")
    model_k3, labels_k3, silhouette_k3 = perform_spectral_clustering(X_pca_values, k=fixed_k, random_state=RANDOM_STATE)
    
    if model_k3 is None:
        print(f"Spectral Clustering with k={fixed_k} failed. Continuing to optimal k...")
    else:
        # Save model results
        k3_suffix = f"k{fixed_k}"
        model_k3_file = MODEL_FILE_TEMPLATE.format(suffix=k3_suffix)
        try:
            joblib.dump(model_k3, model_k3_file)
            print(f"Spectral Clustering model for k={fixed_k} saved to {model_k3_file}")
        except Exception as e:
            print(f"Error saving Spectral Clustering model for k={fixed_k}: {e}")
        
        # Visualize the fixed k results
        visualize_clusters(X_pca_df, labels_k3, plot_dir=PLOT_DIR, 
                          filename_suffix=k3_suffix, k=fixed_k)
        
        # Save clustered data
        k3_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=k3_suffix)
        df_k3_output = full_df.copy()
        df_k3_output['Cluster_Spectral'] = labels_k3
        try:
            df_k3_output.to_csv(k3_output_file, index=False)
            print(f"Clustered data for k={fixed_k} saved to {k3_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Clusters ---
    print("\n=== Finding Optimal Number of Clusters ===")
    optimal_k = find_optimal_k_silhouette(X_pca_values, max_k=MAX_K, plot_dir=PLOT_DIR, random_state=RANDOM_STATE)

    # Skip this step if optimal_k is the same as fixed_k
    if optimal_k == fixed_k:
        print(f"Optimal number of clusters ({optimal_k}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Run Spectral Clustering with Optimal k ---
        print(f"\n=== Running Spectral Clustering with optimal k={optimal_k} ===")
        model_opt, labels_opt, silhouette_opt = perform_spectral_clustering(X_pca_values, k=optimal_k, random_state=RANDOM_STATE)
        
        if model_opt is None:
            print(f"Spectral Clustering with optimal k={optimal_k} failed.")
        else:
            # Save model results
            opt_suffix = f"optimal_k{optimal_k}"
            model_opt_file = MODEL_FILE_TEMPLATE.format(suffix=opt_suffix)
            try:
                joblib.dump(model_opt, model_opt_file)
                print(f"Spectral Clustering model for optimal k={optimal_k} saved to {model_opt_file}")
            except Exception as e:
                print(f"Error saving Spectral Clustering model for optimal k={optimal_k}: {e}")
            
            # Visualize the optimal k results
            visualize_clusters(X_pca_df, labels_opt, plot_dir=PLOT_DIR, 
                              filename_suffix=opt_suffix, k=optimal_k)
            
            # Save clustered data
            opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=opt_suffix)
            df_opt_output = full_df.copy()
            df_opt_output['Cluster_Spectral'] = labels_opt
            try:
                df_opt_output.to_csv(opt_output_file, index=False)
                print(f"Clustered data for optimal k={optimal_k} saved to {opt_output_file}")
            except Exception as e:
                print(f"Error saving clustered data for optimal k={optimal_k}: {e}")

    print("\nSpectral clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_k} clustering")


if __name__ == "__main__":
    main()