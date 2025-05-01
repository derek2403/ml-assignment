# Path: som/som.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# import seaborn as sns # Keep removed
# from sklearn.preprocessing import StandardScaler # REMOVED - Input is PCA data
# from sklearn.decomposition import PCA # REMOVED - Input is PCA data, only used for visualization
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from minisom import MiniSom
import joblib
import os
import math
import warnings
from mpl_toolkits.mplot3d import Axes3D # For 3D plot

# === Configuration ===
# Input data file - updated to final.csv
INPUT_DATA_FILE = '../final.csv'
# Original unprocessed data file (needed for merging final results)
ORIGINAL_UNPROCESSED_DATA_FILE = '../../dataset.csv' # Adjust path if needed

OUTPUT_DIR = '.' # Output in the current directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'som_clustered_data_{suffix}.csv')
SOM_MODEL_FILE = os.path.join(OUTPUT_DIR, 'som_model.joblib')
MAX_K_NODES = 10 # Max clusters for SOM nodes
RANDOM_STATE = 42
SOM_ITERATIONS = 10000 # Keep as example, might need tuning
SOM_GRID_SIZE = None # Auto-calculate based on data size

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

def train_som(X_pca_values, grid_size=None, sigma=1.5, learning_rate=0.5, num_iterations=5000, random_seed=42):
    """Initialize and train the Self-Organizing Map on PCA data."""
    print("\n--- Training Self-Organizing Map (on PCA data) --- ")
    n_features = X_pca_values.shape[1] # Number of PCA components
    n_samples = X_pca_values.shape[0]

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
    # Note: PCA init is still relevant even if input is already PCA,
    # it uses PCA on the input (PCA data) to initialize weights.
    som.pca_weights_init(X_pca_values)
    print(f"Training SOM for {num_iterations} iterations...")
    som.train_batch(X_pca_values, num_iterations, verbose=True) # Use train_batch for efficiency
    print("SOM training completed.")
    return som

def plot_som_distance_map(som, plot_dir='plots'):
    """Plot the SOM's distance map (U-Matrix)."""
    print("\n--- Plotting SOM Distance Map (U-Matrix) --- ")
    os.makedirs(plot_dir, exist_ok=True) # Ensure plot dir exists
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

def plot_som_node_clusters(som, node_cluster_labels, plot_dir='plots', suffix=''):
    """Visualize the clustered SOM nodes on the grid."""
    print(f"\n--- Plotting SOM Node Clusters Map ({suffix}) --- ")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    num_clusters = len(np.unique(node_cluster_labels))
    
    plt.pcolor(node_cluster_labels.T, cmap=plt.cm.get_cmap('viridis', max(1, num_clusters)))
    plt.colorbar(ticks=range(max(1, num_clusters)), label='Node Cluster ID')

    plt.title(f'SOM Grid Colored by Node Cluster (k={num_clusters})')
    plt.xticks(np.arange(som.get_weights().shape[0] + 1))
    plt.yticks(np.arange(som.get_weights().shape[1] + 1))
    plt.grid(True)
    plt.tight_layout()
    node_clusters_path = os.path.join(plot_dir, f'som_node_clusters_{suffix}.png')
    plt.savefig(node_clusters_path)
    plt.close()
    print(f"SOM Node Clusters plot saved to {node_clusters_path}")

def get_data_point_labels(som, X_pca_values, node_cluster_labels):
    """Assign cluster labels to data points based on their BMU's cluster."""
    print("\n--- Assigning cluster labels to data points --- ")
    data_labels = np.zeros(len(X_pca_values), dtype=int)
    for i, x in enumerate(X_pca_values):
        bmu_row, bmu_col = som.winner(x) # Find Best Matching Unit
        data_labels[i] = node_cluster_labels[bmu_row, bmu_col]
    print("Data point labels assigned.")
    return data_labels

def visualize_clusters(X, labels, plot_dir='plots', filename_suffix='', algorithm_name='SOM', k=None):
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
    plot_path = os.path.join(plot_dir, f'som_clusters_{filename_suffix}.png')
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
        return # Exit if data loading failed
    
    X_pca_values = X_pca_df.values

    # --- Load Original Data for Reference ---
    df_original_ref = load_original_data_for_reference(ORIGINAL_UNPROCESSED_DATA_FILE)

    # --- Scaling REMOVED ---
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X) # Not needed, input is PCA data
    # print("Data scaling confirmed/re-applied.")

    # --- Train SOM ---
    # Pass PCA numpy array
    som = train_som(X_pca_values, grid_size=SOM_GRID_SIZE, num_iterations=SOM_ITERATIONS, random_seed=RANDOM_STATE)

    # --- Visualize SOM U-Matrix ---
    plot_som_distance_map(som, plot_dir=PLOT_DIR)

    # --- Save SOM Model ---
    try:
        joblib.dump(som, SOM_MODEL_FILE) # Use joblib or pickle
        print(f"\nSOM model saved to {SOM_MODEL_FILE}")
    except Exception as e:
        print(f"Error saving SOM model: {e}")

    # --- STEP 1: Cluster SOM Nodes with fixed k=3 ---
    fixed_k = 3
    print(f"\n=== Clustering SOM Nodes with fixed k={fixed_k} ===")
    som_weights = som.get_weights()
    node_cluster_labels_k3 = cluster_som_nodes(som_weights, k=fixed_k, random_state=RANDOM_STATE)
    
    # Visualize k=3 node clustering
    plot_som_node_clusters(som, node_cluster_labels_k3, plot_dir=PLOT_DIR, suffix=f"k{fixed_k}")
    
    # Assign k=3 data labels
    data_labels_k3 = get_data_point_labels(som, X_pca_values, node_cluster_labels_k3)
    
    # Evaluate k=3 clustering
    if len(set(data_labels_k3)) > 1:
        try:
            silhouette_avg_k3 = silhouette_score(X_pca_values, data_labels_k3)
            print(f"K=3 Data Point Clustering Silhouette Score: {silhouette_avg_k3:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette score for k=3: {e}")
    
    # Visualize k=3 data clusters
    visualize_clusters(X_pca_df, data_labels_k3, plot_dir=PLOT_DIR, 
                      filename_suffix=f"k{fixed_k}", k=fixed_k)
    
    # Save k=3 clustered data
    k3_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=f"k{fixed_k}")
    df_k3_output = full_df.copy()
    df_k3_output['Cluster_SOM'] = data_labels_k3
    try:
        df_k3_output.to_csv(k3_output_file, index=False)
        print(f"Clustered data for k={fixed_k} saved to {k3_output_file}")
    except Exception as e:
        print(f"Error saving clustered data for k={fixed_k}: {e}")

    # --- STEP 2: Find Optimal Number of Clusters for SOM Nodes ---
    print("\n=== Finding Optimal Number of Clusters for SOM Nodes ===")
    optimal_k = find_optimal_k_for_nodes(som_weights, max_k=MAX_K_NODES, random_state=RANDOM_STATE)

    # Skip this step if optimal_k is the same as fixed_k
    if optimal_k == fixed_k:
        print(f"Optimal number of clusters ({optimal_k}) is the same as fixed k ({fixed_k}). Skipping duplicate clustering.")
    else:
        # --- STEP 3: Cluster SOM Nodes with Optimal k ---
        print(f"\n=== Clustering SOM Nodes with optimal k={optimal_k} ===")
        node_cluster_labels_opt = cluster_som_nodes(som_weights, k=optimal_k, random_state=RANDOM_STATE)
        
        # Visualize optimal node clustering
        plot_som_node_clusters(som, node_cluster_labels_opt, plot_dir=PLOT_DIR, suffix=f"optimal_k{optimal_k}")
        
        # Assign optimal data labels
        data_labels_opt = get_data_point_labels(som, X_pca_values, node_cluster_labels_opt)
        
        # Evaluate optimal clustering
        if len(set(data_labels_opt)) > 1:
            try:
                silhouette_avg_opt = silhouette_score(X_pca_values, data_labels_opt)
                print(f"Optimal k={optimal_k} Data Point Clustering Silhouette Score: {silhouette_avg_opt:.4f}")
            except Exception as e:
                print(f"Error calculating silhouette score for optimal k={optimal_k}: {e}")
        
        # Visualize optimal data clusters
        visualize_clusters(X_pca_df, data_labels_opt, plot_dir=PLOT_DIR, 
                          filename_suffix=f"optimal_k{optimal_k}", k=optimal_k)
        
        # Save optimal clustered data
        opt_output_file = CLUSTERED_DATA_FILE_TEMPLATE.format(suffix=f"optimal_k{optimal_k}")
        df_opt_output = full_df.copy()
        df_opt_output['Cluster_SOM'] = data_labels_opt
        try:
            df_opt_output.to_csv(opt_output_file, index=False)
            print(f"Clustered data for optimal k={optimal_k} saved to {opt_output_file}")
        except Exception as e:
            print(f"Error saving clustered data for optimal k={optimal_k}: {e}")

    print("\nSOM clustering process completed!")
    print("Summary:")
    print(f"1. Fixed k={fixed_k} clustering")
    print(f"2. Optimal k={optimal_k} clustering")

if __name__ == "__main__":
    main()