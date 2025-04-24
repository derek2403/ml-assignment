
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import warnings


# === Configuration ===
DATA_FILE = '../data.csv'
OUTPUT_DIR = '.' # Output in the current directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_FILE = os.path.join(OUTPUT_DIR, 'hdbscan_model.joblib') # Changed model name
CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'hdbscan_clustered_data.csv') # Simplified template
RANDOM_STATE = 42 # Used for PCA, not directly by HDBSCAN core algorithm


MIN_CLUSTER_SIZE = 15 
MIN_SAMPLES = None    
CLUSTER_SELECTION_EPSILON = 0.0 
METRIC = 'euclidean' 

# --- Ensure output directories exist ---
os.makedirs(PLOT_DIR, exist_ok=True)

# Set random seeds for reproducibility where applicable (like PCA)
np.random.seed(RANDOM_STATE)

# === Helper Functions ===

def load_data(file_path='../data.csv'):
    """Load the dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    if 'Potability' in df.columns:
        print("'Potability' column found.")
        y = df['Potability'].astype(int).values
        X = df.drop('Potability', axis=1)
        feature_names = X.columns.tolist()
    else:
        print("'Potability' column not found.")
        X = df
        y = None
        feature_names = X.columns.tolist()
    X_values = X.values # Get numpy array for processing
    return X_values, y, feature_names, df # Return original df too

def scale_data(X):
    """Scale data using StandardScaler."""
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling complete.")
    # Optionally save the scaler
    # joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'hdbscan_scaler.joblib'))
    return X_scaled, scaler

def perform_hdbscan(X_scaled):
    """Perform HDBSCAN clustering."""
    print("\n--- Performing HDBSCAN Clustering ---")
    print(f"Parameters: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}, metric='{METRIC}'")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                metric=METRIC,
                                cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                                core_dist_n_jobs=-1) # Use all available cores

    clusterer.fit(X_scaled)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_ # Probability of point belonging to assigned cluster
    outlier_scores = clusterer.outlier_scores_ # Higher score -> more likely an outlier

    # --- Basic Cluster Statistics ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nHDBSCAN completed.")
    print(f"  Number of clusters found: {n_clusters}")
    print(f"  Number of noise points: {n_noise} ({ (n_noise / len(labels) * 100) :.1f}%)")
    print(f"  Total points: {len(labels)}")

    if n_clusters > 1:
        # Create a mask to select only non-noise points for evaluation
        core_samples_mask = (labels != -1)
        X_core = X_scaled[core_samples_mask]
        labels_core = labels[core_samples_mask]

        if len(X_core) > 1 and len(set(labels_core)) > 1: # Check if enough points/clusters remain
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

def visualize_clusters_pca(X, labels, y, plot_dir, algorithm_name='HDBSCAN', random_state=42):
    """Visualize the clusters using PCA."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X) # Use scaled data for PCA

    plt.figure(figsize=(12, 8))

    unique_labels = np.unique(labels)
    n_plot_labels = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot_labels))

    # Plot data points with cluster colors
    for i, cluster_label in enumerate(unique_labels):
        cluster_points = X_pca[labels == cluster_label]
        label_text = f'Noise' if cluster_label == -1 else f'Cluster {cluster_label}'
        point_size = 10 if cluster_label == -1 else 50 # Smaller size for noise
        alpha_val = 0.1 if cluster_label == -1 else 0.7 # More transparent for noise
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=point_size, c=[colors[i]],
                    label=label_text, alpha=alpha_val)

    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    plt.title(f'{algorithm_name} Clustering Results (found {n_clusters_found} clusters, PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'hdbscan_clusters_pca.png') # Simplified filename
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
        # Create legend for Potability if y has unique values
        unique_y = np.unique(y)
        if len(unique_y) <= 10:
             handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
             legend_labels = [f'Potability {label}' for label in unique_y]
             plt.legend(handles, legend_labels, title="Original Labels")

        plt.grid(True)
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png')
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

# === Main Execution Logic ===
def main():
    # --- Load Data ---
    X_original, y_true, feature_names, df_original = load_data(DATA_FILE)

    # --- Scale Data ---
    X_scaled, scaler = scale_data(X_original)

    # --- Perform HDBSCAN Clustering ---
    hdbscan_model, labels, probabilities, outlier_scores = perform_hdbscan(X_scaled)

    # --- Save Model ---
    joblib.dump(hdbscan_model, MODEL_FILE)
    print(f"\nHDBSCAN model saved to {MODEL_FILE}")

    # --- Add Cluster Info to Original Data (Optional Saving) ---
    df_clustered = df_original.copy()
    df_clustered['HDBSCAN_Cluster'] = labels
    df_clustered['HDBSCAN_Prob'] = probabilities
    df_clustered['HDBSCAN_OutlierScore'] = outlier_scores

    clustered_file = CLUSTERED_DATA_FILE_TEMPLATE
    # df_clustered.to_csv(clustered_file, index=False) # Uncomment to save
    print(f"\nCluster labels and probabilities generated. Clustered data ready (not saved by default).")
    print(df_clustered.head())

    # --- Visualize Results ---
    visualize_clusters_pca(X_scaled, labels, y_true, plot_dir=PLOT_DIR, random_state=RANDOM_STATE)

    print("\nHDBSCAN clustering process completed!")


if __name__ == "__main__":
    main()