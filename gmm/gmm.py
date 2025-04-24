
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
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

def find_optimal_components(X, max_components=10, plot_dir='gmm/plots'):
    """Find the optimal number of GMM components using BIC and AIC."""
    print("\n--- Finding Optimal Number of Components (BIC/AIC) ---")
    n_components_range = range(1, max_components + 1)
    bics = []
    aics = []

    for n_components in n_components_range:
        print(f"Calculating for n_components={n_components}...")
        gmm = GaussianMixture(n_components=n_components, 
                              covariance_type='full',
                              random_state=42,
                              n_init=5, 
                              init_params='kmeans' 
                             )
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        print(f"  BIC: {bics[-1]:.2f}, AIC: {aics[-1]:.2f}")

    # --- Plotting BIC and AIC ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bics, marker='o', linestyle='--', label='BIC')
    plt.plot(n_components_range, aics, marker='s', linestyle=':', label='AIC')
    plt.title('GMM BIC and AIC Scores')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion Score (Lower is better)')
    plt.xticks(n_components_range)
    plt.legend()
    plt.grid(True)
    bic_aic_plot_path = os.path.join(plot_dir, 'gmm_bic_aic.png')
    plt.savefig(bic_aic_plot_path)
    plt.close()
    print(f"BIC/AIC plot saved to {bic_aic_plot_path}")

    optimal_n_components_bic = n_components_range[np.argmin(bics)]
    print(f"Optimal number of components based on BIC: {optimal_n_components_bic}")

    if optimal_n_components_bic == 1:
        print("BIC suggests k=1, which is not suitable for silhouette scoring.")
        optimal_n_components_aic = n_components_range[np.argmin(aics)]
        optimal_n_components = max(2, optimal_n_components_aic) # Ensure at least k=2
        print(f"Falling back to AIC suggestion or min k=2. Using k = {optimal_n_components}")
    else:
        optimal_n_components = optimal_n_components_bic
        print(f"Using optimal k = {optimal_n_components} based on BIC.")

    return optimal_n_components

def perform_gmm(X, n_components, random_state=42):
    """Perform GMM clustering with the specified number of components."""
    print(f"\n--- Performing GMM Clustering (n_components={n_components}) ---")
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=random_state,
                          n_init=10, # More initializations for the final model
                          init_params='kmeans')
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    silhouette_avg = -999 # Default placeholder
    if n_components > 1:
        try:
            silhouette_avg = silhouette_score(X, labels)
            print(f"GMM completed.")
            print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
        except ValueError as e:
            print(f"Could not calculate Silhouette Score: {e}")
            silhouette_avg = -998 
    else:
        print("GMM completed with k=1. Silhouette score not applicable.")
        
    
    return gmm, labels, probs

def visualize_clusters_pca(X, labels, y, plot_dir='gmm/plots', algorithm_name='GMM', k=None, random_state=42):
    """Visualize the clusters using PCA."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels) # Infer k if not provided
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    # Plot data points with cluster colors
    for cluster_label, color in zip(unique_labels, colors):
        cluster_points = X_pca[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[color],
                    label=f'Cluster {cluster_label}', alpha=0.6)


    plt.title(f'{algorithm_name} Clustering Results (k={k}, PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'gmm_clusters_pca_k{k}.png')
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
    MODEL_FILE = os.path.join(OUTPUT_DIR, 'gmm_model.joblib')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'gmm_clustered_data_k{k}.csv')
    MAX_COMPONENTS = 15 # Max number of components to test
    RANDOM_STATE = 42

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    X, y, df_original_with_target = load_data(DATA_FILE)

    # --- Ensure data is scaled ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling confirmed/re-applied.")

    # --- Find Optimal Number of Components ---
    optimal_k = find_optimal_components(X_scaled, max_components=MAX_COMPONENTS, plot_dir=PLOT_DIR)

    # --- Perform GMM Clustering ---
    gmm_model, labels, probabilities = perform_gmm(X_scaled, n_components=optimal_k, random_state=RANDOM_STATE)

    # --- Save Model ---
    joblib.dump(gmm_model, MODEL_FILE)
    print(f"\nGMM model saved to {MODEL_FILE}")

    # --- Add Cluster Labels and Probabilities to Original Data ---
    df_clustered = df_original_with_target.loc[X.index].copy()
    df_clustered['Cluster'] = labels
    # Add probability columns for each component
    for i in range(optimal_k):
        df_clustered[f'Cluster_{i}_Prob'] = probabilities[:, i]
    print("\nCluster labels and probabilities generated (not saving clustered data file).")

    # --- Visualize Results ---
    visualize_clusters_pca(X_scaled, labels, y, plot_dir=PLOT_DIR, k=optimal_k, random_state=RANDOM_STATE)

    print("\nGMM clustering process completed!")


if __name__ == "__main__":
    main() 