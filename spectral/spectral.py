
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler # Good practice
from sklearn.decomposition import PCA
import joblib 
import os
import warnings


def load_data(file_path='../data.csv'):
    """Load the scaled dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    # Separate features and target (if Potability exists)
    if 'Potability' in df.columns:
        print("'Potability' column found.")
        y = df['Potability'].astype(int)
        X = df.drop('Potability', axis=1)
    else:
        print("'Potability' column not found.")
        X = df
        y = None
    return X, y, df # Return original df too

def find_optimal_k_silhouette(X, max_k=10, plot_dir='spectral/plots', random_state=42):
    """Find the optimal number of clusters using Silhouette scores."""
    print("\n--- Finding Optimal K using Silhouette Score ---")
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
            labels = spectral.fit_predict(X) # Use X (or X_sample)
            
            # Check if multiple clusters were actually found (sometimes fails)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels) # Score on original X
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
    plt.title('Silhouette Score for Optimal K (Spectral Clustering)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(list(valid_k_range))
    plt.grid(True)
    silhouette_plot_path = os.path.join(plot_dir, 'spectral_silhouette_scores.png')
    plt.savefig(silhouette_plot_path)
    plt.close()
    print(f"Silhouette score plot saved to {silhouette_plot_path}")

    # --- Determine Optimal K based on Silhouette Score ---
    optimal_k_silhouette = valid_k_range[np.argmax(valid_silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette}")

    return optimal_k_silhouette

def perform_spectral_clustering(X, k, random_state=42):
    """Perform Spectral Clustering with the specified number of clusters."""
    print(f"\n--- Performing Spectral Clustering with k={k} ---")
    model = SpectralClustering(n_clusters=k,
                               affinity='nearest_neighbors',
                               assign_labels='kmeans',
                               random_state=random_state,
                               n_init=10,
                               n_jobs=-1)
    try:
        labels = model.fit_predict(X)
        
        # Check if the expected number of clusters was found
        actual_clusters = len(set(labels))
        if actual_clusters != k:
            print(f"Warning: Found {actual_clusters} clusters, but requested {k}.")
            
        if actual_clusters > 1:
             silhouette_avg = silhouette_score(X, labels)
             print(f"Spectral Clustering completed.")
             print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        else:
            print(f"Spectral Clustering completed, but only found {actual_clusters} cluster(s). Silhouette score not applicable.")
            silhouette_avg = np.nan
            
    except Exception as e:
        print(f"Error during final Spectral Clustering fit/predict: {e}")
        labels = np.full(X.shape[0], -1) # Assign dummy label on error
        silhouette_avg = np.nan
        
    return model, labels, silhouette_avg

def visualize_clusters_pca(X, labels, y, plot_dir='spectral/plots', algorithm_name='Spectral', k=None, random_state=42):
    """Visualize the clusters using PCA."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    # Handle case where clustering might have failed and assigned -1
    if -1 in unique_labels and len(unique_labels) == 1:
        print("Warning: Only noise/error labels found. Cannot visualize clusters properly.")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.5)
        k_title = "Error"
    else:
        if k is None:
            k = len(unique_labels) # Infer k if not provided (and not error state)
        k_title = str(k)
        colors = plt.cm.viridis(np.linspace(0, 1, k))

        # Plot data points with cluster colors
        for cluster_label, color in zip(unique_labels, colors):
            if cluster_label == -1: continue # Skip error labels if any other exist
            cluster_points = X_pca[labels == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[color],
                        label=f'Cluster {cluster_label}', alpha=0.6)

    plt.title(f'{algorithm_name} Clustering Results (k={k_title}, PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Only add legend if there are actual cluster labels
    if len(unique_labels) > 1 or (len(unique_labels) == 1 and -1 not in unique_labels):
        plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'spectral_clusters_pca_k{k_title}.png')
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
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
        legend_labels = [f'Potability {label}' for label in np.unique(y)]
        plt.legend(handles, legend_labels, title="Original Labels")
        plt.grid(True)
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png') # Overwrite is fine
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'spectral_clustered_data_k{k}.csv')
    MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'spectral_model_k{k}.joblib')
    MAX_K = 10 # Max number of clusters to test (adjust based on computation time)
    RANDOM_STATE = 42

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    X, y, df_original_with_target = load_data(DATA_FILE)

    # --- Ensure data is scaled ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling confirmed/re-applied.")

    # --- Find Optimal K ---
    # Warning: This step can be slow!
    optimal_k = find_optimal_k_silhouette(X_scaled, max_k=MAX_K, plot_dir=PLOT_DIR, random_state=RANDOM_STATE)

    # --- Perform Spectral Clustering ---
    model, labels, silhouette_avg = perform_spectral_clustering(X_scaled, k=optimal_k, random_state=RANDOM_STATE)


    k_str_model = str(optimal_k) if not np.isnan(silhouette_avg) else "Error"
    model_filename = MODEL_FILE_TEMPLATE.format(k=k_str_model)
    try:
        joblib.dump(model, model_filename)
        print(f"\nSpectral Clustering model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving Spectral Clustering model: {e}")

    # --- Add Cluster Labels to Original Data ---
    df_clustered = df_original_with_target.loc[X.index].copy()
    df_clustered['Cluster'] = labels

    print("\nCluster labels generated (not saving clustered data file).")

    # --- Visualize Results ---
    visualize_clusters_pca(X_scaled, labels, y, plot_dir=PLOT_DIR, k=optimal_k, random_state=RANDOM_STATE)

    print("\nSpectral clustering process completed!")


if __name__ == "__main__":
    main() 