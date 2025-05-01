import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # For SOM node clustering
from sklearn.decomposition import PCA
import joblib
import glob
import os
from minisom import MiniSom # For SOM loading

# Updated to use final.csv for evaluation
EVAL_DATA_FILE = 'final.csv' 

# Directories to search for model files
MODEL_DIRS = [
    'dbscan',
    'fuzzy',
    'gmm', 
    'hdbscan',
    'hierarchical',
    'kmeans',
    'som',
    'spectral'
]

# Model type mapping (directory name to algorithm name for display)
MODEL_TYPE_MAPPING = {
    'dbscan': 'DBSCAN',
    'fuzzy': 'Fuzzy C-Means',
    'gmm': 'GMM',
    'hdbscan': 'HDBSCAN',
    'hierarchical': 'Hierarchical',
    'kmeans': 'K-Means',
    'som': 'SOM',
    'spectral': 'Spectral'
}

OUTPUT_DIR = 'comparison' 
PLOT_FILENAME = 'model_comparison_silhouette.png'
RANDOM_STATE = 42

def find_model_files():
    """Dynamically find all joblib model files in the specified directories."""
    print("Searching for model files...")
    model_paths = {}
    
    for model_dir in MODEL_DIRS:
        # Get all joblib files in this directory
        search_path = os.path.join(model_dir, '*.joblib')
        joblib_files = glob.glob(search_path)
        
        if joblib_files:
            # Get the algorithm name from the directory
            algorithm_name = MODEL_TYPE_MAPPING.get(model_dir, model_dir.capitalize())
            
            # Use the first found joblib file by default
            model_paths[algorithm_name] = joblib_files[0]
            
            # If there are multiple files, add them as separate models with version numbers
            if len(joblib_files) > 1:
                for i, file_path in enumerate(joblib_files[1:], 1):
                    # Extract some identifier from the filename, or just use index
                    file_name = os.path.basename(file_path)
                    model_paths[f"{algorithm_name} (variant {i})"] = file_path
            
            print(f"Found {len(joblib_files)} model file(s) for {algorithm_name}")
        else:
            print(f"No model files found in {model_dir}/")
    
    return model_paths

def load_eval_data(file_path):
    """Load and scale evaluation dataset features."""
    print(f"Loading evaluation data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Since we're using final.csv, we need to extract the PCA columns
        pca_cols = [col for col in df.columns if col.startswith('PC') and col[2:].isdigit()]
        if pca_cols:
            print(f"Found {len(pca_cols)} PCA columns: {pca_cols[:5]}...")
            X = df[pca_cols]
        else:
            # If no PCA columns found, use all columns except obvious cluster/label columns
            cols_to_drop = [col for col in df.columns if 'Cluster' in col or 'Label' in col or 'Potability' in col]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"No PCA columns found. Using {X.shape[1]} features for evaluation.")
        
        print(f"Data loaded with shape: {X.shape}")
        
        # Scale the data for metrics calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Evaluation data scaled.")
        return X_scaled, X.columns
    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading or scaling data: {e}")
        return None, None

# Function needed for processing SOM results
def get_som_data_point_labels(som, X_scaled, node_cluster_labels):
    """Assign cluster labels to data points based on their BMU's cluster."""
    data_labels = np.zeros(len(X_scaled), dtype=int)
    for i, x in enumerate(X_scaled):
        bmu_idx = som.winner(x) # Find Best Matching Unit index
        data_labels[i] = node_cluster_labels[bmu_idx] # Get cluster label of that node
    return data_labels

def get_labels_from_model(model_name, model_path, X_scaled):
    """Load a model and get cluster labels for the data."""
    if model_path is None or not os.path.exists(model_path):
        print(f"Model file not found for {model_name} at {model_path}. Skipping.")
        return None, None # Return None for labels and k
        
    print(f"\nLoading model and getting labels for: {model_name}")
    try:
        model = joblib.load(model_path)
        
        if 'K-Means' in model_name:
            labels = model.predict(X_scaled)
            k = model.n_clusters
        elif 'Hierarchical' in model_name:
            # Assumes model object saved after fit_predict stores labels
            labels = model.labels_
            k = model.n_clusters
        elif 'DBSCAN' in model_name:
            # Assumes model object saved after fit_predict stores labels
            labels = model.labels_
            # k is number of clusters excluding noise
            k = len(set(labels)) - (1 if -1 in labels else 0)
        elif 'HDBSCAN' in model_name:
            # For HDBSCAN model, similar handling as DBSCAN
            labels = model.labels_
            # k is number of clusters excluding noise
            k = len(set(labels)) - (1 if -1 in labels else 0)
        elif 'GMM' in model_name:
            labels = model.predict(X_scaled)
            k = model.n_components
        elif 'Spectral' in model_name:
             # Assumes model object saved after fit_predict stores labels
            labels = model.labels_
            k = model.n_clusters
        elif 'SOM' in model_name:
            # Need to re-cluster nodes and assign points
            som = model # Loaded model is the MiniSom object
            som_weights = som.get_weights()
            map_rows, map_cols, n_features = som_weights.shape
            num_nodes = map_rows * map_cols
            node_vectors = som_weights.reshape(num_nodes, n_features)
            
            # Determine optimal k for nodes (or use the one from the results table: k=2)
            k_nodes = 2 # Hardcoding based on previous results table
            print(f"  Re-clustering SOM nodes with k={k_nodes}...")
            kmeans_nodes = KMeans(n_clusters=k_nodes, random_state=RANDOM_STATE, n_init=10)
            node_labels_flat = kmeans_nodes.fit_predict(node_vectors)
            
            print(f"  Assigning data points to SOM node clusters...")
            winner_coordinates = np.array([som.winner(x) for x in X_scaled])
            labels = np.array([node_labels_flat[i[0] * map_cols + i[1]] for i in winner_coordinates])
            k = k_nodes
            
        elif 'Fuzzy C-Means' in model_name:
            # Load the results dictionary saved by fuzzy.py
            fcm_results = model 
            labels = fcm_results.get('hard_labels')
            k = fcm_results.get('n_clusters')
            if labels is None or k is None:
                 print(f"  Error: Could not extract labels/k from {model_name} results dict.")
                 return None, None
        else:
            print(f"  Label extraction logic not defined for {model_name}. Skipping.")
            return None, None
            
        print(f"  Found k={k}, Labels shape: {labels.shape}")
        return labels, k
        
    except Exception as e:
        print(f"  Error loading or processing model {model_name}: {e}")
        return None, None

def calculate_metrics(X_scaled, labels, model_name):
    """Calculate multiple clustering metrics, handling DBSCAN noise."""
    if labels is None:
        return np.nan, np.nan, np.nan
        
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters < 2:
        print(f"  Cannot calculate metrics for {model_name} (found {n_clusters} cluster(s)).")
        return np.nan, np.nan, np.nan
        
    try:
        # For DBSCAN, calculate only on non-noise points
        if -1 in unique_labels:
            print("  Calculating metrics excluding noise points...")
            non_noise_mask = (labels != -1)
            if np.sum(non_noise_mask) > 1:
                X_filtered = X_scaled[non_noise_mask]
                labels_filtered = labels[non_noise_mask]
                silhouette = silhouette_score(X_filtered, labels_filtered)
                calinski = calinski_harabasz_score(X_filtered, labels_filtered)
                davies = davies_bouldin_score(X_filtered, labels_filtered)
            else:
                print("  Not enough non-noise points for metrics calculation.")
                return np.nan, np.nan, np.nan
        else:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)
        
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Index: {calinski:.4f}")
        print(f"  Davies-Bouldin Index: {davies:.4f}")
        return silhouette, calinski, davies
    except Exception as e:
        print(f"  Error calculating metrics for {model_name}: {e}")
        return np.nan, np.nan, np.nan

def extract_params_from_filename(model_name, filename):
    """Extract parameter information from filename for display purposes."""
    basename = os.path.basename(filename)
    
    if 'DBSCAN' in model_name and 'eps' in basename:
        try:
            parts = basename.split('eps')[1].split('.joblib')[0]
            eps_str, ms_str = parts.split('_ms')
            return f"eps={float(eps_str):.2f}, ms={int(ms_str)}"
        except:
            pass
    
    if 'K-Means' in model_name or 'Hierarchical' in model_name or 'Spectral' in model_name:
        try:
            if 'k' in basename:
                k_val = basename.split('k')[1].split('.')[0].split('_')[0]
                return f"k={k_val}"
        except:
            pass
            
    if 'Fuzzy C-Means' in model_name:
        try:
            k_val = basename.split('k')[1].split('.')[0].split('_')[0]
            return f"c={k_val}, m=2.0"
        except:
            pass
    
    # Default - just return the filename
    return basename

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Model Comparison Script ---")
    print(f"Using {EVAL_DATA_FILE} for model evaluation")
    
    # 1. Find model files
    model_paths = find_model_files()
    
    if not model_paths:
        print("No model files found. Please check the directory structure.")
        exit()
    
    print(f"Found {len(model_paths)} model files for comparison:")
    for name, path in model_paths.items():
        print(f"  • {name}: {path}")
    
    # 2. Load and Prepare Data
    X_eval_scaled, feature_names = load_eval_data(EVAL_DATA_FILE)
    
    if X_eval_scaled is None:
        print("Halting script due to data loading error.")
        exit()
        
    results = []

    # 3. Load Models, Get Labels, Calculate Metrics
    for name, path in model_paths.items():
        labels, k = get_labels_from_model(name, path, X_eval_scaled)
        silhouette, calinski, davies = calculate_metrics(X_eval_scaled, labels, name)
        
        # Try to extract parameters from filename
        params = extract_params_from_filename(name, path)
        
        results.append({
            'Model': name,
            'File': os.path.basename(path),
            'Optimal k / Params': k if k is not None else params,
            'Silhouette Score': silhouette,
            'Calinski-Harabasz': calinski,
            'Davies-Bouldin': davies
        })
    
    # 4. Create Results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n--- Comparison Results ---")
    results_df_sorted = results_df.sort_values(by='Silhouette Score', ascending=False).reset_index(drop=True)
    print(results_df_sorted.to_string())

    # Save the comparison results to CSV
    results_csv_path = os.path.join(OUTPUT_DIR, 'clustering_comparison_results.csv')
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df_sorted.to_csv(results_csv_path, index=False)
        print(f"\nComparison results saved to {results_csv_path}")
    except Exception as e:
        print(f"Error saving comparison results to CSV: {e}")

    # 5. Visualization
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric
    metrics = ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin']
    n_metrics = len(metrics)
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, n_metrics, i)
        plot_df = results_df_sorted.dropna(subset=[metric])
        
        # For Davies-Bouldin, lower is better, so we'll invert the sorting
        if metric == 'Davies-Bouldin':
            plot_df = plot_df.sort_values(by=metric)
            title_suffix = '(Lower is Better)'
        else:
            title_suffix = '(Higher is Better)'
        
        sns.barplot(x=metric, y='Model', data=plot_df, color='skyblue')
        plt.title(f'{metric}\n{title_suffix}')
        plt.xlabel(metric)
        plt.ylabel('Clustering Algorithm' if i == 1 else '')
        
        # Add score labels to bars
        for index, value in enumerate(plot_df[metric]):
            plt.text(value + 0.01, index, f'{value:.4f}', va='center')
    
    plt.suptitle(f'Clustering Model Comparison\n(Evaluated on: {os.path.basename(EVAL_DATA_FILE)})', y=1.02)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    comparison_plot_path = os.path.join(OUTPUT_DIR, PLOT_FILENAME)
    try:
        plt.tight_layout()
        plt.savefig(comparison_plot_path, bbox_inches='tight')
        print(f"\nComparison plot saved to {comparison_plot_path}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")

    print("\n--- Comparison Script Finished ---") 