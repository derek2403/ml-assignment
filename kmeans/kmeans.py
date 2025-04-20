#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler  # Though data is scaled, good practice to ensure
from sklearn.decomposition import PCA
import joblib
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")

def load_data(file_path='../data.csv'):
    """Load the scaled dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    # Separate features and target (if Potability exists)
    if 'Potability' in df.columns:
        print("'Potability' column found. Using it for visualization later.")
        y = df['Potability'].astype(int) # Ensure integer type for coloring
        X = df.drop('Potability', axis=1)
    else:
        print("'Potability' column not found. Proceeding without target variable.")
        X = df
        y = None
    return X, y, df # Return original df too for saving later

def find_optimal_k(X, max_k=10, plot_dir='kmeans/plot'):
    """Find the optimal number of clusters using Elbow and Silhouette methods."""
    print("\n--- Finding Optimal K (Elbow and Silhouette) ---")
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        print(f"Calculating for k={k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score only if more than 1 cluster
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"  Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {score:.4f}")
        else:
             print(f"  Inertia: {kmeans.inertia_:.2f}")


    # --- Plotting Elbow Method ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.xticks(k_range)
    plt.grid(True)
    elbow_plot_path = os.path.join(plot_dir, 'elbow_method.png')
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow method plot saved to {elbow_plot_path}")

    # --- Plotting Silhouette Scores ---
    if len(k_range) > 1 and len(silhouette_scores) > 0: # Ensure we have scores to plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
        plt.title('Silhouette Score for Optimal K')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.xticks(k_range)
        plt.grid(True)
        silhouette_plot_path = os.path.join(plot_dir, 'silhouette_scores.png')
        plt.savefig(silhouette_plot_path)
        plt.close()
        print(f"Silhouette score plot saved to {silhouette_plot_path}")

        # --- Determine Optimal K based on Silhouette Score ---
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette}")
        # You might also visually inspect the Elbow plot for an "elbow" point.
        # Let's tentatively use the silhouette score's recommendation.
        optimal_k = optimal_k_silhouette
    else:
        # Fallback if only k=1 was tested or no scores available
        print("Warning: Could not determine optimal k from Silhouette. Inspect Elbow plot manually.")
        # Ask user or default? For now, let's default to a common value like 3, but highlight this.
        optimal_k = 3 # Default or ask user
        print(f"Defaulting to k={optimal_k}. Please verify using the Elbow plot.")

    return optimal_k


def perform_kmeans(X, k, random_state=42):
    """Perform K-Means clustering with the specified number of clusters."""
    print(f"\n--- Performing K-Means Clustering with k={k} ---")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X, labels)

    print(f"K-Means completed.")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Average Silhouette Score: {silhouette_avg:.4f}")
    return kmeans, labels, centers

def visualize_clusters_pca(X, labels, centers, y, plot_dir='kmeans/plot', k=None, random_state=42):
    """Visualize the clusters using PCA for dimensionality reduction."""
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(centers) if centers is not None else None

    plt.figure(figsize=(12, 8))

    # Determine unique labels and create a color map
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels) # Infer k if not provided
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    # Plot data points with cluster colors
    for cluster_label, color in zip(unique_labels, colors):
        cluster_points = X_pca[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[color], # Use list for color
                    label=f'Cluster {cluster_label}', alpha=0.6)

    # Plot cluster centers
    if centers_pca is not None:
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=250, marker='*',
                    c='red', edgecolor='black', label='Centroids')

    plt.title('K-Means Clustering Results (PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, 'kmeans_clusters_pca.png')
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
        # Create a legend for Potability
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
        legend_labels = [f'Potability {label}' for label in np.unique(y)]
        plt.legend(handles, legend_labels, title="Original Labels")
        plt.grid(True)
        pca_potability_plot_path = os.path.join(plot_dir, 'pca_colored_by_potability.png')
        plt.savefig(pca_potability_plot_path)
        plt.close()
        print(f"PCA plot colored by Potability saved to {pca_potability_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plot')
    MODEL_FILE = os.path.join(OUTPUT_DIR, 'kmeans_model.joblib')
    CLUSTERED_DATA_FILE = os.path.join(OUTPUT_DIR, 'kmeans_clustered_data.csv')
    MAX_K = 15 # Range of k to test
    RANDOM_STATE = 42

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    X, y, df_original_with_target = load_data(DATA_FILE)

    # --- Find Optimal K ---
    # It's good practice to ensure data is scaled, even if loaded from scaled.csv
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    optimal_k = find_optimal_k(X_scaled, max_k=MAX_K, plot_dir=PLOT_DIR)

    # --- Perform K-Means ---
    kmeans_model, labels, centers = perform_kmeans(X_scaled, k=optimal_k, random_state=RANDOM_STATE)

    # --- Save Model ---
    joblib.dump(kmeans_model, MODEL_FILE)
    print(f"\nK-Means model saved to {MODEL_FILE}")

    # --- Add Cluster Labels to Original Data ---
    # Use the index from X to align labels correctly with the original DataFrame
    # (which might include the Potability column)
    df_clustered = df_original_with_target.loc[X.index].copy() # Ensure we use correct rows if some were dropped
    # df_clustered['Cluster'] = labels
    # df_clustered.to_csv(CLUSTERED_DATA_FILE, index=False)
    # print(f"Clustered data saved to {CLUSTERED_DATA_FILE}")
    print("\nCluster labels generated (not saving clustered data file).")

    # --- Visualize Results ---
    # Pass X_scaled for visualization as that's what clustering was done on
    visualize_clusters_pca(X_scaled, labels, centers, y, plot_dir=PLOT_DIR, k=optimal_k, random_state=RANDOM_STATE)

    print("\nK-Means clustering process completed!")


if __name__ == "__main__":
    main() 