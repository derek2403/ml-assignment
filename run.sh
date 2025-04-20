#!/bin/bash

echo "--- Starting Clustering Runs ---"

# 1. K-Means
echo "Running K-Means (kmeans/kmeans.py)..."
(cd kmeans && python kmeans.py)
echo "K-Means finished."

# 2. Hierarchical
echo "Running Hierarchical Clustering (hierarchical/hierarchical.py)..."
(cd hierarchical && python hierarchical.py)
echo "Hierarchical finished."

# 3. DBSCAN
echo "Running DBSCAN (dbscan/dbscan.py)..."
# Remember to check/set MANUAL_EPS in dbscan.py if needed based on k-distance plot
(cd dbscan && python dbscan.py)
echo "DBSCAN finished."

# 4. GMM
echo "Running Gaussian Mixture Models (gmm/gmm.py)..."
(cd gmm && python gmm.py)
echo "GMM finished."

# 5. Spectral Clustering
echo "Running Spectral Clustering (spectral/spectral.py)..."
(cd spectral && python spectral.py)
echo "Spectral finished."

# 6. SOM
echo "Running Self-Organizing Maps (som/som.py)..."
(cd som && python som.py)
echo "SOM finished."

# 7. Fuzzy C-Means
echo "Running Fuzzy C-Means (fuzzy/fuzzy.py)..."
(cd fuzzy && python fuzzy.py)
echo "Fuzzy C-Means finished."

echo "--- All Clustering Runs Completed ---"

echo "--- Starting Comparison ---"
python compare.py
echo "Comparison finished."