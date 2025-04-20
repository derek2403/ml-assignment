#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
N_SAMPLES_TOTAL = 3277
N_FEATURES = 9
RANDOM_STATE = 42
OUTPUT_FILENAME = 'mock_complex_structure_data.csv' # New filename

# Define column names
FEATURE_COLUMNS = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

# --- Component Configurations ---
# We'll aim for roughly 4 underlying groups + noise
n_cluster_1 = 800  # Clear blob
n_cluster_2 = 800  # Clear blob, well separated from 1
n_cluster_3 = 800  # Blob designed to overlap significantly with cluster 4
n_cluster_4 = 800  # Moons (2 groups) overlapping with cluster 3

n_noise = N_SAMPLES_TOTAL - (n_cluster_1 + n_cluster_2 + n_cluster_3) # Moons count as 1 group size here

# --- Generation ---
def generate_complex_data():
    """Generates synthetic data with mixed cluster structures."""
    print("Generating complex structure data...")
    np.random.seed(RANDOM_STATE) # Ensure reproducibility

    # Cluster 1: Clear Blob
    center1 = np.random.rand(N_FEATURES) * 5 # Spread centers
    X1, y1 = make_blobs(n_samples=n_cluster_1, n_features=N_FEATURES,
                        centers=[center1], cluster_std=0.6, random_state=RANDOM_STATE)
    y1[:] = 0 # Label as cluster 0
    print(f"Generated Cluster 1 (Blob): {X1.shape}")

    # Cluster 2: Clear Blob (separated from 1)
    center2 = center1 + np.random.rand(N_FEATURES)*3 + 4 # Ensure separation
    X2, y2 = make_blobs(n_samples=n_cluster_2, n_features=N_FEATURES,
                        centers=[center2], cluster_std=0.7, random_state=RANDOM_STATE+1)
    y2[:] = 1 # Label as cluster 1
    print(f"Generated Cluster 2 (Blob): {X2.shape}")

    # Cluster 3: Overlapping Blob
    # Position near where moons might be, higher std dev
    center3 = center1 + np.random.rand(N_FEATURES) * 2 - 1 # Position for overlap
    X3, y3 = make_blobs(n_samples=n_cluster_3, n_features=N_FEATURES,
                        centers=[center3], cluster_std=1.8, random_state=RANDOM_STATE+2)
    y3[:] = 2 # Label as cluster 2
    print(f"Generated Cluster 3 (Overlapping Blob): {X3.shape}")

    # Cluster 4: Moons (will be labeled as cluster 3 internally)
    # Generate in 2D, use first two features, place near Cluster 3 center projection
    X_moons_2d, y_moons = make_moons(n_samples=n_cluster_4, noise=0.1, random_state=RANDOM_STATE+3)
    # Scale and shift moons to overlap with projected center of cluster 3
    moon_scale = 3.0
    X_moons_2d = X_moons_2d * moon_scale + center3[:2] # Overlap with C3 projection

    X4 = np.random.randn(n_cluster_4, N_FEATURES) * 0.1 # Small noise in other dims
    X4[:, 0:2] = X_moons_2d # Place moons in first two dims
    y4 = y_moons + 3 # Labels 3 and 4 for the two moons
    print(f"Generated Cluster 4 (Moons): {X4.shape}")


    # Noise: Uniform noise across the approximate range of combined data
    # Combine first to estimate range, then generate noise (or just guess range)
    temp_X = np.vstack((X1, X2, X3, X4))
    min_vals = temp_X.min(axis=0)
    max_vals = temp_X.max(axis=0)

    X_noise = np.random.uniform(low=min_vals, high=max_vals, size=(n_noise, N_FEATURES))
    y_noise = np.full(n_noise, -1) # Label noise as -1
    print(f"Generated Noise: {X_noise.shape}")

    # Combine all parts
    X_combined = np.vstack((X1, X2, X3, X4, X_noise))
    # Internal true labels (not saved) - useful for potential evaluation later
    y_true_internal = np.concatenate((y1, y2, y3, y4, y_noise))

    print("Combined data components.")
    # Shuffle data rows
    indices = np.arange(X_combined.shape[0])
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_true_internal = y_true_internal[indices] # Keep internal labels aligned

    return X_combined, y_true_internal

def create_and_save_dataframe(X):
    """Scales features and saves ONLY features to CSV."""
    # Scale the combined features
    print("Scaling combined features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create final DataFrame with scaled features only
    df_scaled_features_only = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    # Save the scaled features WITHOUT any cluster labels
    try:
        df_scaled_features_only.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Successfully saved mock dataset (features only) to '{OUTPUT_FILENAME}'")
        print(f"Shape of saved data: {df_scaled_features_only.shape}")
        print("\nSample rows (scaled features):")
        print(df_scaled_features_only.head())
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

    return df_scaled_features_only

if __name__ == "__main__":
    print("--- Mock Complex Structure Data Generation Script ---")
    # Generate data
    features_combined, _ = generate_complex_data() # Ignore true labels for saving

    # Create and save DataFrame (features only)
    mock_df = create_and_save_dataframe(features_combined)

    print("\n--- Script Finished ---")