#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and data
print("Loading DBSCAN model and data...")
model = joblib.load('dbscan.joblib')
data = pd.read_csv('data.csv')

# Scale the features (same as during training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Get cluster labels
labels = model.fit_predict(X_scaled)

# Add cluster labels to the dataframe
data['Cluster'] = labels

# Print model parameters
print("\nDBSCAN Model Parameters:")
print(f"eps (epsilon): {model.eps}")
print(f"min_samples: {model.min_samples}")

# Basic cluster statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("\nClustering Results:")
print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Total points: {len(labels)}")

# Print cluster sizes and proportions
print("\nCluster Sizes:")
cluster_sizes = pd.Series(labels).value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    percentage = size / len(labels) * 100
    print(f"{label}: {size} points ({percentage:.1f}%)")

# Calculate cluster statistics
print("\nCluster Statistics:")
for col in data.columns[:-1]:  # Exclude the Cluster column
    print(f"\n{col}:")
    cluster_stats = data.groupby('Cluster')[col].agg(['mean', 'std', 'min', 'max'])
    print(cluster_stats.round(3))

# Interpretation guide
print("\nCluster Interpretation Guide:")
print("----------------------------")
print("Cluster -1 (Noise): Points that don't belong to any cluster (outliers)")
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    # Get the defining characteristics of the cluster
    cluster_data = data[data['Cluster'] == i]
    all_data = data[data['Cluster'] != -1]  # Excluding noise points
    
    characteristics = []
    for col in data.columns[:-1]:
        cluster_mean = cluster_data[col].mean()
        overall_mean = all_data[col].mean()
        overall_std = all_data[col].std()
        
        # Check if the cluster mean is significantly different from overall mean
        z_score = (cluster_mean - overall_mean) / overall_std
        if abs(z_score) > 1:  # More than 1 standard deviation different
            direction = "higher" if z_score > 0 else "lower"
            characteristics.append(f"- {col}: {direction} than average ({cluster_mean:.2f} vs {overall_mean:.2f})")
    
    if characteristics:
        print("Defining characteristics:")
        print("\n".join(characteristics))
    else:
        print("No strongly distinguishing characteristics")

# Create visualization
plt.figure(figsize=(12, 6))

# If we have more than 2 features, use PCA for visualization
if X_scaled.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)

# Plot cluster sizes
plt.subplot(1, 2, 2)
cluster_sizes = pd.Series(labels).value_counts().sort_index()
cluster_labels = ['Noise' if idx == -1 else f'Cluster {idx}' for idx in cluster_sizes.index]
plt.bar(cluster_labels, cluster_sizes.values)
plt.title('Cluster Sizes')
plt.xticks(rotation=45)
plt.ylabel('Number of Points')

plt.tight_layout()
plt.savefig('cluster_analysis.png')
print("\nVisualization saved as 'cluster_analysis.png'") 