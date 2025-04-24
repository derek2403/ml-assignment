import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import warnings
import joblib



def load_data(file_path='../data.csv'):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    if 'Potability' in df.columns:
        print("'Potability' column found.")
        y = df['Potability'].astype(int)
        X = df.drop('Potability', axis=1)
    else:
        print("'Potability' column not found.")
        X = df
        y = None
    return X, y, df

def find_optimal_eps(X, min_samples, plot_dir='dbscan/plots'):
    print(f"\n--- Finding Optimal Eps using k-distance graph (min_samples={min_samples}) ---")

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    

    kth_distances = np.sort(distances[:, min_samples-1], axis=0)
    

    plt.figure(figsize=(10, 6))
    plt.plot(kth_distances)
    plt.title(f'k-Distance Graph (k = {min_samples})')
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.grid(True)
    k_dist_path = os.path.join(plot_dir, f'k_distance_graph_k{min_samples}.png')
    plt.savefig(k_dist_path)
    plt.close()
    print(f"k-Distance graph saved to {k_dist_path}")
    print("Inspect the graph: Look for the 'elbow' or point of maximum curvature.")
    print("This point's distance value on the y-axis is a good candidate for 'eps'.")
  
    try:
        diffs = np.diff(kth_distances, 2)
        elbow_index = np.argmax(diffs) + 1
        suggested_eps = kth_distances[elbow_index]
        print(f"--> Tentative suggested 'eps' based on max curvature heuristic: {suggested_eps:.4f}")
    except Exception as e:
        print(f"Could not automatically suggest eps: {e}")
        suggested_eps = 2.0
        print(f"Using default eps fallback: {suggested_eps}")
        
    return suggested_eps

def perform_dbscan(X, eps, min_samples):
    print(f"\n--- Performing DBSCAN (eps={eps:.4f}, min_samples={min_samples}) ---")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X)
    
   
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
   
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    
   
    if n_clusters_ > 0:
        non_noise_mask = (labels != -1)
        if np.sum(non_noise_mask) > 1:
            X_filtered = X[non_noise_mask]
            labels_filtered = labels[non_noise_mask]
            if len(set(labels_filtered)) > 1:
                silhouette_avg = silhouette_score(X_filtered, labels_filtered)
                print(f'Average Silhouette Score (excluding noise): {silhouette_avg:.4f}')
            else:
                 print("Silhouette Score cannot be calculated: Only one cluster found among non-noise points.")
        else:
             print("Silhouette Score cannot be calculated: Not enough non-noise points found.")
    else:
        print("Silhouette Score cannot be calculated: No clusters found (all noise?).")
        
    return dbscan, labels, core_samples_mask, n_clusters_, n_noise_

def visualize_clusters_pca(X, labels, core_samples_mask, y, plot_dir='dbscan/plots', algorithm_name='DBSCAN', n_clusters=None, n_noise=None, eps=None, min_samples=None, random_state=42):
    print("\n--- Visualizing Clusters using PCA --- ")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    
    unique_labels = set(labels)
    

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0)))
    color_map = {label: colors[i] for i, label in enumerate(sorted([l for l in unique_labels if l != -1]))}
    color_map[-1] = (0, 0, 0, 1)

    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color_map[k]), markeredgecolor='k', markersize=10, label=f'Cluster {k}' if k != -1 else None)
        
        xy = X_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color_map[k]), markeredgecolor='k', markersize=5, label='Noise' if k == -1 else None)

    handles, current_labels = plt.gca().get_legend_handles_labels()
    if 'Noise' not in current_labels and -1 in unique_labels:
        
        noise_handle = plt.Line2D([0], [0], marker='o', color='w', label='Noise', 
                                markerfacecolor=tuple(color_map[-1]), markersize=5, markeredgecolor='k')
        handles.append(noise_handle)
    if len([l for l in unique_labels if l != -1]) > 0 and not any(l.startswith('Cluster') for l in current_labels):
         
         cluster_handle = plt.Line2D([0], [0], marker='o', color='w', label=f'Clusters (Total: {n_clusters})', 
                                markerfacecolor='grey', markersize=10, markeredgecolor='k')
         handles.append(cluster_handle)

    plt.legend(handles=handles)
    title = f'{algorithm_name} Results (PCA Reduced)\neps={eps:.2f}, min_samples={min_samples} | Clusters: {n_clusters}, Noise: {n_noise}'
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    pca_plot_path = os.path.join(plot_dir, f'dbscan_clusters_pca_eps{eps:.2f}_ms{min_samples}.png')
    plt.savefig(pca_plot_path)
    plt.close()
    print(f"Cluster PCA plot saved to {pca_plot_path}")

def main():

    DATA_FILE = '../data.csv'
    OUTPUT_DIR = '.'
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    CLUSTERED_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_clustered_data_eps{eps:.2f}_ms{ms}.csv')
    NOISE_DATA_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_noise_points_eps{eps:.2f}_ms{ms}.csv')
    MODEL_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, 'dbscan_model_eps{eps:.2f}_ms{ms}.joblib')
    RANDOM_STATE = 42
    

    
    N_DIMS = 9
    MIN_SAMPLES = max(5, N_DIMS)
    
    
    print("\n--- Finding optimal parameters through grid search ---")
    eps_range = np.linspace(0.5, 3.0, 10)
    best_eps = None
    best_silhouette = -1
    best_noise_ratio = 1.0
    
    
    X, y, df_original_with_target = load_data(DATA_FILE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nPerforming grid search for optimal eps...")
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
            
        
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            try:
                silhouette = silhouette_score(X_scaled[non_noise_mask], 
                                            labels[non_noise_mask])
                
                
                
                if silhouette > best_silhouette and noise_ratio < 0.3:
                    best_silhouette = silhouette
                    best_eps = eps
                    print(f"New best eps: {eps:.2f} (Silhouette: {silhouette:.3f}, Noise ratio: {noise_ratio:.2%})")
            except:
                continue
    
    if best_eps is None:
        print("Could not find optimal eps, using default value.")
        best_eps = 1.5
    
    print(f"\nSelected parameters: eps={best_eps:.2f}, min_samples={MIN_SAMPLES}")
    
    
    model, labels, core_mask, n_clusters, n_noise = perform_dbscan(X_scaled, eps=best_eps, min_samples=MIN_SAMPLES)

    
    model_filename = MODEL_FILE_TEMPLATE.format(eps=best_eps, ms=MIN_SAMPLES)
    try:
        joblib.dump(model, model_filename)
        print(f"\nDBSCAN model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving DBSCAN model: {e}")

    
    visualize_clusters_pca(X_scaled, labels, core_mask, y, plot_dir=PLOT_DIR, 
                           n_clusters=n_clusters, n_noise=n_noise, eps=best_eps,
                           min_samples=MIN_SAMPLES, random_state=RANDOM_STATE)

    print("\nDBSCAN clustering process completed!")

if __name__ == "__main__":
    main() 