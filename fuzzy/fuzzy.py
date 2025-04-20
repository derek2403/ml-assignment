#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Good practice
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import os
import warnings

# Suppress warnings if needed
# warnings.filterwarnings("ignore", category=FutureWarning)

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
    # FCM requires data in shape (n_features, n_samples)
    X_fcm = X.T
    return X, X_fcm, y, df # Return both X shapes, y, and original df

def find_optimal_clusters_fcm(X_fcm, max_clusters=10, m=2, plot_dir='fuzzy/plots'):
    """Find optimal number of clusters (c) for FCM using validity indices."""
    print(f"\n--- Finding Optimal Clusters (c) for FCM (m={m}) ---")
    fpcs = []
    pes = []
    cluster_range = range(2, max_clusters + 1)
    
    # Note: FPC requires the membership matrix U
    #       PE also requires U

    data = X_fcm # Use the (features, samples) shape
    
    for ncenters in cluster_range:
        print(f"Calculating for c={ncenters}...")
        try:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data, ncenters, m, error=0.005, maxiter=1000, init=None, seed=42)
            
            fpcs.append(fpc)
            # Calculate Partition Entropy (PE) - skfuzzy doesn't have a direct function
            # PE = -sum(u_ik * log(u_ik)) / N
            pe_val = -np.sum(u * np.log2(u + 1e-9)) / data.shape[1] # Add epsilon for log(0)
            pes.append(pe_val)
            print(f"  FPC: {fpc:.4f}, PE: {pe_val:.4f}")
            
        except Exception as e:
            print(f"  Error during FCM for c={ncenters}: {e}. Appending NaN.")
            fpcs.append(np.nan)
            pes.append(np.nan)

    # Filter out NaN values
    valid_fpcs = [(c, score) for c, score in zip(cluster_range, fpcs) if not np.isnan(score)]
    valid_pes = [(c, score) for c, score in zip(cluster_range, pes) if not np.isnan(score)]

    if not valid_fpcs or not valid_pes:
        print("Error: Could not calculate validity indices for any c. Cannot determine optimal c.")
        return 2 # Default fallback

    valid_c_range_fpc, valid_fpc_scores = zip(*valid_fpcs)
    valid_c_range_pe, valid_pe_scores = zip(*valid_pes)

    # --- Plotting Validity Indices ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (c)')
    ax1.set_ylabel('FPC (Higher is better)', color=color)
    ax1.plot(valid_c_range_fpc, valid_fpc_scores, marker='o', linestyle='--', color=color, label='FPC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('PE (Lower is better)', color=color)  
    ax2.plot(valid_c_range_pe, valid_pe_scores, marker='s', linestyle=':', color=color, label='PE')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('FCM Cluster Validity Indices')
    plt.xticks(list(cluster_range))
    # Add combined legend
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    validity_plot_path = os.path.join(plot_dir, 'fcm_validity_indices.png')
    plt.savefig(validity_plot_path)
    plt.close()
    print(f"Validity indices plot saved to {validity_plot_path}")

    # --- Determine Optimal c based on FPC ---
    # Often maximizing FPC is used, though minimizing PE is also valid.
    optimal_c = valid_c_range_fpc[np.argmax(valid_fpc_scores)]
    print(f"Optimal c based on maximizing FPC: {optimal_c}")
    # Note: The "best" c can be subjective, also consider PE minimum or domain knowledge.

    return optimal_c

def calculate_xie_beni(data_points, centers, membership, m):
    """Calculate Xie-Beni Index."""
    n_samples = data_points.shape[0]
    n_clusters = centers.shape[0]
    
    term1_sum = 0
    for k in range(n_clusters):
        for i in range(n_samples):
            # Use .iloc for integer-position row access in pandas DataFrame
            term1_sum += (membership[k, i] ** m) * (np.linalg.norm(data_points.iloc[i] - centers[k]) ** 2)
            
    min_center_dist_sq = np.inf
    for k1 in range(n_clusters):
        for k2 in range(k1 + 1, n_clusters):
            dist_sq = np.linalg.norm(centers[k1] - centers[k2]) ** 2
            if dist_sq < min_center_dist_sq:
                min_center_dist_sq = dist_sq
                
    if min_center_dist_sq == 0: # Avoid division by zero if centers are identical
        return np.inf
        
    xb = term1_sum / (n_samples * min_center_dist_sq)
    return xb

def perform_fcm(X_fcm, n_clusters, m=2, error=0.005, maxiter=1000, random_seed=42):
    """Perform Fuzzy C-Means clustering."""
    print(f"\n--- Performing Fuzzy C-Means (c={n_clusters}, m={m}) ---")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_fcm, n_clusters, m, error=error, maxiter=maxiter, init=None, seed=random_seed)
    
    # Calculate Partition Entropy (PE)
    pe = -np.sum(u * np.log2(u + 1e-9)) / X_fcm.shape[1] 
    
    # Calculate Xie-Beni (XB)
    # Need data in (samples, features) format for XB calculation distance
    X_orig_shape = X_fcm.T 
    xb = calculate_xie_beni(X_orig_shape, cntr, u, m)
    
    print(f"FCM completed.")
    print(f"  FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
    print(f"  PE (Partition Entropy): {pe:.4f}")
    print(f"  XB (Xie-Beni Index): {xb:.4f}")

    # Hard cluster assignments
    labels = np.argmax(u, axis=0)
    silhouette_avg = np.nan
    if len(set(labels)) > 1:
        try:
            # Silhouette needs original shape (samples, features)
            silhouette_avg = silhouette_score(X_orig_shape, labels)
            print(f"  Silhouette Score (hard labels): {silhouette_avg:.4f}")
        except Exception as e:
            print(f"  Could not calculate Silhouette Score: {e}")
    else:
        print("  Only one cluster assigned. Silhouette score not applicable.")
        
    # Package results for saving
    fcm_results = {
        'centers': cntr,
        'membership': u,
        'fpc': fpc,
        'pe': pe,
        'xb': xb,
        'hard_labels': labels,
        'silhouette': silhouette_avg,
        'n_clusters': n_clusters,
        'fuzziness_m': m
    }
    
    return fcm_results

def visualize_clusters_pca(X, labels, y, plot_dir='fuzzy/plots', algorithm_name='FCM (Hard Labels)', k=None, random_state=42):
    """Visualize the hard cluster assignments using PCA."""
    print("\n--- Visualizing Hard Clusters using PCA --- ")
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

    plt.title(f'{algorithm_name} Results (k={k}, PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'fcm_hard_clusters_pca_k{k}.png')
    plt.savefig(pca_plot_path)
    plt.close()
    print(f"Hard Cluster PCA plot saved to {pca_plot_path}")
    
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

def visualize_fuzzy_partition(X, u, labels, plot_dir='fuzzy/plots', algorithm_name='FCM Fuzzy Partition', k=None, random_state=42):
    """Visualize fuzzy partition using PCA, alpha represents membership certainty."""
    print("\n--- Visualizing Fuzzy Partition using PCA (Alpha=Certainty) --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    if k is None:
        k = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    
    # Calculate max membership for each point (certainty)
    max_membership = np.max(u, axis=0)
    
    # Plot points colored by hard label, alpha by certainty
    for cluster_label, color in zip(unique_labels, colors):
        mask = (labels == cluster_label)
        cluster_points_pca = X_pca[mask]
        cluster_certainty = max_membership[mask]
        
        # Use scatter with varying alpha
        plt.scatter(cluster_points_pca[:, 0], cluster_points_pca[:, 1], 
                    c=[color] * len(cluster_points_pca), # Assign color directly
                    alpha=cluster_certainty, # Use certainty for alpha
                    s=50,
                    label=f'Cluster {cluster_label}')

    plt.title(f'{algorithm_name} (k={k}, PCA Reduced, Alpha=Certainty)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Create custom legend as alpha varies
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 label=f'Cluster {i}', 
                                 markerfacecolor=colors[i], 
                                 markersize=10) for i in unique_labels]
    plt.legend(handles=legend_elements, title="Clusters")
    plt.grid(True)
    fuzzy_pca_plot_path = os.path.join(plot_dir, f'fcm_fuzzy_pca_k{k}.png')
    plt.savefig(fuzzy_pca_plot_path)
    plt.close()
    print(f"Fuzzy Partition PCA plot saved to {fuzzy_pca_plot_path}")

def main():
    # --- Configuration ---
    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    MODEL_FILE = os.path.join(OUTPUT_DIR, 'fuzzy_model.joblib') # Save results dict
    MAX_CLUSTERS = 10 # Max clusters to test for validity indices
    FUZZINESS_M = 2.0 # Standard fuzziness parameter
    RANDOM_STATE = 42 
    # Note: skfuzzy cmeans seed is handled internally if passed

    # --- Ensure output directories exist ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load Data ---
    # X_orig shape: (samples, features), X_fcm shape: (features, samples)
    X_orig, X_fcm, y, df_original_with_target = load_data(DATA_FILE)

    # --- Find Optimal Number of Clusters (c) ---
    optimal_c = find_optimal_clusters_fcm(X_fcm, max_clusters=MAX_CLUSTERS, m=FUZZINESS_M, plot_dir=PLOT_DIR)

    # --- Perform Final FCM Clustering ---
    fcm_results = perform_fcm(X_fcm, n_clusters=optimal_c, m=FUZZINESS_M, random_seed=RANDOM_STATE)

    # --- Save Model Results --- 
    try:
        joblib.dump(fcm_results, MODEL_FILE)
        print(f"\nFCM results (centers, membership, indices) saved to {MODEL_FILE}")
    except Exception as e:
        print(f"Error saving FCM results: {e}")

    # --- Visualize Results ---
    hard_labels = fcm_results['hard_labels']
    membership_u = fcm_results['membership']
    # Ensure we use X_orig (samples, features) for PCA visualization
    visualize_clusters_pca(X_orig, hard_labels, y, plot_dir=PLOT_DIR, k=optimal_c, random_state=RANDOM_STATE)
    visualize_fuzzy_partition(X_orig, membership_u, hard_labels, plot_dir=PLOT_DIR, k=optimal_c, random_state=RANDOM_STATE)

    print("\nFuzzy C-Means clustering process completed!")
    print("NOTE: Possibilistic C-Means, Kernel FCM, and Gustafson-Kessel require different implementations/libraries and are not included.")

if __name__ == "__main__":
    main() 