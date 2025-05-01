# Path: kmeans/kmeans.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Keep removed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler # REMOVED - Input is PCA data
# from sklearn.decomposition import PCA # REMOVED - Input is PCA data, only used for visualization if needed
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
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'kmeans_model_{suffix}.joblib')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'kmeans_clustered_data_{suffix}.csv')
MAX_K = 15 # Max number of clusters to test
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

def find_optimal_k(X, max_k=15, plot_dir='plots', random_state=42):
    """Find optimal K using Elbow and Silhouette methods on PCA data."""
    print("\n--- Finding Optimal K (Elbow and Silhouette on PCA data) ---")
    # Input X is already PCA data (DataFrame)
    X_values = X.values # Use numpy array for KMeans

    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1) # Start from 2 clusters

    for k in k_range:
        print(f"Calculating for k={k}...")
        try:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
            kmeans.fit(X_values)
            inertias.append(kmeans.inertia_)
            score = silhouette_score(X_values, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"  Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {score:.4f}")
        except Exception as e:
            print(f"  Error calculating for k={k}: {e}")
            inertias.append(np.nan)
            silhouette_scores.append(np.nan)

    # Filter out failed calculations
    valid_indices = [i for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
    if not valid_indices:
        print("Error: Could not calculate Silhouette scores for any k > 1. Defaulting to k=3.")
        optimal_k = 3
    else:
        valid_k_range = [k_range[i] for i in valid_indices]
        valid_inertias = [inertias[i] for i in valid_indices]
        valid_silhouette_scores = [silhouette_scores[i] for i in valid_indices]

        # --- Plotting Elbow Method ---
        plt.figure(figsize=(10, 6))
        plt.plot(valid_k_range, valid_inertias, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal K (on PCA data)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.xticks(valid_k_range)
        plt.grid(True)
        elbow_plot_path = os.path.join(plot_dir, 'elbow_method_pca.png')
        plt.savefig(elbow_plot_path)
        plt.close()
        print(f"Elbow method plot saved to {elbow_plot_path}")

        # --- Plotting Silhouette Scores ---
        plt.figure(figsize=(10, 6))
        plt.plot(valid_k_range, valid_silhouette_scores, marker='o', linestyle='--')
        plt.title('Silhouette Score for Optimal K (on PCA data)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.xticks(valid_k_range)
        plt.grid(True)
        silhouette_plot_path = os.path.join(plot_dir, 'silhouette_scores_pca.png')
        plt.savefig(silhouette_plot_path)
        plt.close()
        print(f"Silhouette score plot saved to {silhouette_plot_path}")

        # --- Determine Optimal K based on Silhouette Score ---
        optimal_k = valid_k_range[np.argmax(valid_silhouette_scores)]
        print(f"Optimal K based on Silhouette Score: {optimal_k}")

    return optimal_k

def perform_kmeans(X, k, random_state=42):
    """Perform K-Means clustering on the input data (assumed PCA)."""
    print(f"\n--- Performing K-Means Clustering with k={k} (on PCA data) ---")
    X_values = X.values # Use numpy array for KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    try:
        kmeans.fit(X_values)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_ # Centers in the original PCA space (e.g., 10 dims)
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(X_values, labels)
        print(f"K-Means completed.")
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        return kmeans, labels, centers
    except Exception as e:
        print(f"Error during K-Means fitting: {e}")
        return None, None, None

def visualize_clusters(X, labels, centers, plot_dir='plots', filename_suffix='', k=None):
    """Visualize the clusters with matched style to reference plots."""
    print(f"\n--- Visualizing Clusters ({filename_suffix}) ---")
    
    # Create figure with the same square dimensions as the reference plot
    plt.figure(figsize=(14, 12))
    
    # Extract the first two PCA components for plotting
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    X_plot = X_values[:, :2]  # First two components
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels)
    
    # Main scatter plot with larger points (s=150 like in reference)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], 
                         c=labels, 
                         cmap='viridis',
                         s=150,  # Matched to reference
                         alpha=0.7,  # Same alpha as reference
                         edgecolor='w',  # White edge for better visibility
                         linewidth=0.5)
    
    # Plot centroids
    if centers is not None:
        centers_plot = centers[:, :2]  # First two components
        plt.scatter(centers_plot[:, 0], centers_plot[:, 1], 
                   s=250, marker='X', c='red', 
                   edgecolor='black', linewidth=1.5,
                   label='Centroids')
    
    # Add title and labels
    plt.title(f'KMeans Clustering (k={k})', fontsize=14)
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
    plot_path = os.path.join(plot_dir, f'kmeans_clusters_{filename_suffix}.png')
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

    # --- Load Original Data for Reference ---
    df_original_ref = load_original_data_for_reference(ORIGINAL_UNPROCESSED_DATA_FILE)

    # --- STEP 1: Run KMeans with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Running KMeans with fixed k={fixed_k} ===")
    kmeans_k3, labels_k3, centers_k3 = perform_kmeans(X, k=fixed_k, random_state=RANDOM_STATE)
    
    if kmeans_k3 is None:
        print(f"KMeans with k={fixed_k} failed. Continuing to optimal k...")
    else:
        # Save model results
        k3_model_file = MODEL_FILE_TEMPLATE.format(suffix=f"k{fixed_k}")
        try:
            joblib.dump(kmeans_k3, k3_model_file)
            print(f"KMeans model for k={fixed_k} saved to {k3_model_file}")
        except Exception as e:
            print(f"Error saving KMeans model for k={fixed_k}: {e}")
        
        # Visualize the fixed k results
        visualize_clusters(X, labels_k3, centers_k3, plot_dir=PLOT_DIR, 
                          filename_suffix=f"k{fixed_k}", k=fixed_k)
        
        # Save clustered data
        k3_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=f"k{fixed_k}")
        df_k3_output = full_df.copy()
        df_k3_output['Cluster_KMeans'] = labels_k3
        try:
            df_k3_output.to_csv(k3_output_file, index=False)
            print(f"Clustered data for k={fixed_k} saved to {k3_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Clusters ---
    print("\n=== Finding Optimal Number of Clusters ===")
    optimal_k = find_optimal_k(X, max_k=MAX_K, plot_dir=PLOT_DIR, random_state=RANDOM_STATE)

    # Skip this step if optimal_k is the same as fixed_k
    if optimal_k == fixed_k:
        print(f"Optimal number of clusters ({optimal_k}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Run KMeans with Optimal k ---
        print(f"\n=== Running KMeans with optimal k={optimal_k} ===")
        kmeans_opt, labels_opt, centers_opt = perform_kmeans(X, k=optimal_k, random_state=RANDOM_STATE)
        
        if kmeans_opt is None:
            print(f"KMeans with optimal k={optimal_k} failed.")
        else:
            # Save model results
            opt_model_file = MODEL_FILE_TEMPLATE.format(suffix=f"optimal_k{optimal_k}")
            try:
                joblib.dump(kmeans_opt, opt_model_file)
                print(f"KMeans model for optimal k={optimal_k} saved to {opt_model_file}")
            except Exception as e:
                print(f"Error saving KMeans model for optimal k={optimal_k}: {e}")
            
            # Visualize the optimal k results
            visualize_clusters(X, labels_opt, centers_opt, plot_dir=PLOT_DIR, 
                              filename_suffix=f"optimal_k{optimal_k}", k=optimal_k)
            
            # Save clustered data
            opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=f"optimal_k{optimal_k}")
            df_opt_output = full_df.copy()
            df_opt_output['Cluster_KMeans'] = labels_opt
            try:
                df_opt_output.to_csv(opt_output_file, index=False)
                print(f"Clustered data for optimal k={optimal_k} saved to {opt_output_file}")
            except Exception as e:
                print(f"Error saving clustered data for optimal k={optimal_k}: {e}")

    print("\nKMeans clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_k} clustering")

if __name__ == "__main__":
    main()