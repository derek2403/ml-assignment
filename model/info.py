#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DBSCAN model and data
print("Loading DBSCAN model and data...")
try:
    dbscan_model = joblib.load('dbscan.joblib')
    print("- DBSCAN model loaded.")
    # kmeans_model = joblib.load('kmeans.joblib') # Removed
    # print("- K-Means model loaded.") # Removed
    data = pd.read_csv('data.csv')
    print("- Data loaded.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure dbscan.joblib and data.csv are present.") # Removed kmeans reference
    exit(1)
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit(1)

# Print DBSCAN model parameters
print("\nDBSCAN Model Parameters:")
print(f"eps (epsilon): {dbscan_model.eps}")
print(f"min_samples: {dbscan_model.min_samples}")

# Removed K-Means parameter printing
# if hasattr(kmeans_model, 'n_clusters'):
#     print("\nK-Means Model Parameters:")
#     print(f"n_clusters: {kmeans_model.n_clusters}")
# else:
#     print("\nK-Means model details not available or model type is different.")

# Print data statistics
print("\nData Statistics (Original):")
print(data.describe())

# Print data ranges
print("\nData Ranges (Original):")
for col in data.columns:
    print(f"{col}: min={data[col].min():.2f}, max={data[col].max():.2f}")

# Print data distribution
print("\nData Distribution (first 5 rows):")
print(data.head())

# Scale the features (assuming DBSCAN expects scaled data)
print("\nScaling data using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
print("Data scaling complete.")

# Perform DBSCAN clustering
print("\nPerforming DBSCAN clustering...")
labels = dbscan_model.fit_predict(X_scaled)

# Add cluster labels to the dataframe
data['DBSCAN_Cluster'] = labels

# Basic cluster statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("\nDBSCAN Clustering Results:")
print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Total points: {len(labels)}")

# Print cluster sizes and proportions
print("\nDBSCAN Cluster Sizes:")
cluster_sizes = pd.Series(labels).value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    percentage = size / len(labels) * 100
    print(f"{label}: {size} points ({percentage:.1f}%)")

# Calculate cluster statistics
print("\nDetailed DBSCAN Cluster Statistics (Based on Original Data):")
for col in data.columns[:-1]:  # Exclude the Cluster column
    print(f"\n{col}:")
    # Use the original data for statistics, grouped by DBSCAN cluster label
    cluster_stats = data.groupby('DBSCAN_Cluster')[col].agg(['mean', 'std', 'min', 'max'])
    print(cluster_stats.round(3))

# Print sample points from each cluster
print("\nSample Points from Each DBSCAN Cluster (Original Data):")
for cluster_id in sorted(data['DBSCAN_Cluster'].unique()):
    print(f"\nCluster {cluster_id} Sample Points:")
    # Show original data values for samples
    sample_points = data[data['DBSCAN_Cluster'] == cluster_id].head(3)
    print(sample_points.drop(columns=['DBSCAN_Cluster']).round(3))

# Print cluster centroids
print("\nDBSCAN Cluster Centroids (Based on Original Data):")
# Calculate centroids using original data
centroids = data.groupby('DBSCAN_Cluster').mean()
print(centroids.round(3))

# Print cluster characteristics
print("\nDBSCAN Cluster Characteristics (Compared within Original Data):")
for cluster_id in sorted(data['DBSCAN_Cluster'].unique()):
    if cluster_id == -1:
        print(f"\nCluster {cluster_id} (Noise): Points not assigned to a core cluster.")
        continue
    print(f"\nCluster {cluster_id}:")
    cluster_data = data[data['DBSCAN_Cluster'] == cluster_id]
    # Compare against all non-noise points using original data
    all_data_non_noise = data[data['DBSCAN_Cluster'] != -1]
    
    characteristics = []
    for col in data.columns[:-1]: # Exclude cluster column
        cluster_mean = cluster_data[col].mean()
        overall_mean = all_data_non_noise[col].mean()
        overall_std = all_data_non_noise[col].std()
        
        # Check if the cluster mean is significantly different from overall mean
        # Avoid division by zero if std is 0
        if overall_std > 1e-6:
            z_score = (cluster_mean - overall_mean) / overall_std
            if abs(z_score) > 1:  # More than 1 standard deviation different
                direction = "higher" if z_score > 0 else "lower"
                characteristics.append(f"- {col}: {direction} than average ({cluster_mean:.2f} vs {overall_mean:.2f}) [z={z_score:.2f}]")
        else:
            if cluster_mean != overall_mean:
                 characteristics.append(f"- {col}: Different from average ({cluster_mean:.2f} vs {overall_mean:.2f}) [std=0]")

    if characteristics:
        print("Defining characteristics:")
        print("\n".join(characteristics))
    else:
        print("No strongly distinguishing characteristics based on z-score > 1.")

# Create visualization using scaled data
print("\nGenerating visualization...")
plt.figure(figsize=(12, 6))

# If we have more than 2 features, use PCA for visualization (using scaled data)
if X_scaled.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.title('DBSCAN Cluster Visualization (PCA on Scaled Data)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster ID')

# Plot cluster sizes
plt.subplot(1, 2, 2)
cluster_labels = ['Noise' if idx == -1 else f'Cluster {idx}' for idx in cluster_sizes.index]
plt.bar(cluster_labels, cluster_sizes.values)
plt.title('DBSCAN Cluster Sizes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Points')

plt.tight_layout()
plt.savefig('dbscan_cluster_analysis.png')
print("\nDBSCAN analysis complete. Visualization saved as 'dbscan_cluster_analysis.png'") 