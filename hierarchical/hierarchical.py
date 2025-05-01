import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
import joblib

# --- Configuration ---
# Input data file - updated to final.csv
INPUT_DATA_FILE = '../final.csv'

OUTPUT_DIR = '.'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hierarchical_clustered_data_{suffix}.csv')
MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hierarchical_model_{suffix}.joblib')

LINKAGE_METHOD = 'ward'
RANDOM_STATE = 42
MAX_CLUSTERS = 10  # Maximum number of clusters to test for optimal k

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

def plot_dendrogram(X, method='ward', plot_dir='plots'):
    """Generate and save a dendrogram."""
    print(f"\n--- Generating Dendrogram (method: {method}) ---")
    
    # Convert to numpy array if it's a DataFrame
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    linked = linkage(X_values, method=method)
    
    plt.figure(figsize=(15, 8))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               truncate_mode='lastp', 
               p=30, 
               show_contracted=True 
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

def find_optimal_clusters(X, max_clusters=10, linkage_method='ward'):
    """Find the optimal number of clusters using silhouette scores."""
    print("\n--- Finding Optimal Number of Clusters (Silhouette Method) ---")
    
    # Convert to numpy array if it's a DataFrame
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    silhouette_scores = []
    n_clusters_range = range(2, min(max_clusters + 1, len(X_values)))
    
    for n_clusters in n_clusters_range:
        print(f"Testing k={n_clusters}...")
        
        # Create and fit the model
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X_values)
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(X_values, labels)
            silhouette_scores.append(silhouette_avg)
            print(f"  Silhouette score for k={n_clusters}: {silhouette_avg:.4f}")
        except Exception as e:
            print(f"  Error calculating silhouette score for k={n_clusters}: {e}")
            silhouette_scores.append(-1)  # Use -1 to indicate error
    
    # Determine optimal k (ignoring any -1 error values)
    valid_scores = [(k, score) for k, score in zip(n_clusters_range, silhouette_scores) if score > -1]
    if not valid_scores:
        print("Could not determine optimal k. Defaulting to k=3.")
        return 3
    
    optimal_k, best_score = max(valid_scores, key=lambda x: x[1])
    print(f"Optimal number of clusters based on silhouette score: {optimal_k} (score: {best_score:.4f})")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    silhouette_plot_path = os.path.join(PLOT_DIR, 'silhouette_scores.png')
    plt.savefig(silhouette_plot_path)
    plt.close()
    print(f"Silhouette scores plot saved to {silhouette_plot_path}")
    
    return optimal_k

def perform_hierarchical_clustering(X, n_clusters, linkage_method='ward'):
    """Perform Agglomerative Hierarchical Clustering."""
    print(f"\n--- Performing Hierarchical Clustering (k={n_clusters}, linkage={linkage_method}) ---")
    
    # Convert to numpy array if it's a DataFrame
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_values)
    
    silhouette_avg = silhouette_score(X_values, labels)
    print(f"Hierarchical Clustering completed.")
    print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
    
    return model, labels

def visualize_clusters(X, labels, plot_dir='plots', filename_suffix='', algorithm_name='Hierarchical', k=None):
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
    plot_path = os.path.join(plot_dir, f'hierarchical_clusters_{filename_suffix}.png')
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

    # --- Generate Dendrogram ---
    try:
        plot_dendrogram(X, method=LINKAGE_METHOD, plot_dir=PLOT_DIR)
    except MemoryError:
        print("MemoryError: Dataset might be too large for full dendrogram generation.")
        print("Skipping dendrogram and proceeding with clustering.")
    except Exception as e:
        print(f"Error generating dendrogram: {e}")
        print("Proceeding with clustering.")

    # --- STEP 1: Run Hierarchical Clustering with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Running Hierarchical Clustering with fixed k={fixed_k} ===")
    model_k3, labels_k3 = perform_hierarchical_clustering(X, n_clusters=fixed_k, linkage_method=LINKAGE_METHOD)
    
    # Save model and visualize results
    k3_suffix = f"k{fixed_k}"
    model_k3_file = MODEL_FILE_TEMPLATE.format(suffix=k3_suffix)
    try:
        joblib.dump(model_k3, model_k3_file)
        print(f"Hierarchical model (k={fixed_k}) saved to {model_k3_file}")
    except Exception as e:
        print(f"Error saving Hierarchical model (k={fixed_k}): {e}")
    
    # Visualize the fixed k results
    visualize_clusters(X, labels_k3, plot_dir=PLOT_DIR, 
                      filename_suffix=k3_suffix, k=fixed_k)
    
    # Save clustered data
    k3_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=k3_suffix)
    df_k3_output = full_df.copy()
    df_k3_output['Hierarchical_Cluster'] = labels_k3
    try:
        df_k3_output.to_csv(k3_output_file, index=False)
        print(f"Clustered data for k={fixed_k} saved to {k3_output_file}")
    except Exception as e:
        print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Clusters ---
    print("\n=== Finding Optimal Number of Clusters ===")
    optimal_k = find_optimal_clusters(X, max_clusters=MAX_CLUSTERS, linkage_method=LINKAGE_METHOD)

    # Skip this step if optimal_k is the same as fixed_k
    if optimal_k == fixed_k:
        print(f"Optimal number of clusters ({optimal_k}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Run Hierarchical Clustering with Optimal k ---
        print(f"\n=== Running Hierarchical Clustering with optimal k={optimal_k} ===")
        model_opt, labels_opt = perform_hierarchical_clustering(X, n_clusters=optimal_k, linkage_method=LINKAGE_METHOD)
        
        # Save model and visualize results
        opt_suffix = f"optimal_k{optimal_k}"
        model_opt_file = MODEL_FILE_TEMPLATE.format(suffix=opt_suffix)
        try:
            joblib.dump(model_opt, model_opt_file)
            print(f"Hierarchical model (optimal k={optimal_k}) saved to {model_opt_file}")
        except Exception as e:
            print(f"Error saving Hierarchical model (optimal k={optimal_k}): {e}")
        
        # Visualize the optimal k results
        visualize_clusters(X, labels_opt, plot_dir=PLOT_DIR, 
                          filename_suffix=opt_suffix, k=optimal_k)
        
        # Save clustered data
        opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=opt_suffix)
        df_opt_output = full_df.copy()
        df_opt_output['Hierarchical_Cluster'] = labels_opt
        try:
            df_opt_output.to_csv(opt_output_file, index=False)
            print(f"Clustered data for optimal k={optimal_k} saved to {opt_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for optimal k={optimal_k}: {e}")

    print("\nHierarchical clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_k} clustering")

if __name__ == "__main__":
    main() 