import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
CORS(app)

model = None
X_train = None
X_train_scaled = None
column_order = [
    'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

def load_model():
    global model, X_train, X_train_scaled
    print("Loading DBSCAN model and data...")
    try:
        model = joblib.load('dbscan.joblib')
        print("- DBSCAN model loaded.")

        if os.path.exists('hdbscan.joblib'):
            print("- HDBSCAN model found (will be used as fallback 1)")
        else:
            print("- Warning: HDBSCAN model not found at 'hdbscan.joblib'")
            
        if os.path.exists('kmeans.joblib'):
            print("- K-Means model found (will be used as fallback 2)")
        else:
            print("- Warning: K-Means model not found at 'kmeans.joblib'")

        X_train = pd.read_csv('data.csv')
        
        if 'Potability' in X_train.columns:
            X_train = X_train.drop('Potability', axis=1)
            print("- Dropped Potability column from training data.")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("- Training data loaded and scaled.")
        
        if len(model.labels_) != len(X_train):
             print(f"Warning: Mismatch between DBSCAN model labels ({len(model.labels_)}) and training data rows ({len(X_train)}). Ensure model was trained on this data.")
             

        print("\nDBSCAN Model Parameters:")
        print(f"eps (epsilon): {model.eps}")
        print(f"min_samples: {model.min_samples}")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure dbscan.joblib and data.csv are present.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        exit(1)

def predict_cluster_dbscan(X_new):
    if X_train_scaled is None:
        raise RuntimeError("Scaled training data (X_train_scaled) not loaded.")
        
    nbrs = NearestNeighbors(n_neighbors=model.min_samples, radius=model.eps)
    nbrs.fit(X_train_scaled)
    
    distances, indices = nbrs.radius_neighbors(X_new)
    
    print(f"Number of neighbors found within eps radius: {len(indices[0])}")
    
    if len(distances[0]) == 0:
        print("No neighbors found within eps radius.")
        return -1
        
    try:
        neighbor_labels = model.labels_[indices[0]]
        print(f"Neighbor labels: {neighbor_labels}")
    except IndexError:
        print(f"Error: Indices {indices[0]} out of bounds for model labels (length {len(model.labels_)}).")
        return -1
        
    if all(label == -1 for label in neighbor_labels):
        print("All neighbors are noise points.")
        return -1
        
    non_noise_labels = [label for label in neighbor_labels if label != -1]
    if non_noise_labels:
        most_common_label = max(set(non_noise_labels), key=non_noise_labels.count)
        print(f"Found non-noise neighbors with labels: {non_noise_labels}. Most common: {most_common_label}")
        return most_common_label
        
    print("Found neighbors, but only noise labels.")
    return -1

def predict_cluster_ensemble(X_new):
    print("\nStarting ensemble prediction cascade...")
    dbscan_cluster = predict_cluster_dbscan(X_new)
    
    if dbscan_cluster != -1:
        print("Using DBSCAN classification.")
        return dbscan_cluster, "DBSCAN"
    
    print("DBSCAN returned noise (-1), trying HDBSCAN...")
    try:
        hdbscan_model = joblib.load('hdbscan.joblib')
        
        nbrs = NearestNeighbors(n_neighbors=5)
        nbrs.fit(X_train_scaled)
        distances, indices = nbrs.kneighbors(X_new)
        
        neighbor_labels = hdbscan_model.labels_[indices[0]]
        print(f"HDBSCAN nearest neighbor labels: {neighbor_labels}")
        
        non_noise_labels = [label for label in neighbor_labels if label != -1]
        
        if non_noise_labels:
            hdbscan_cluster = max(set(non_noise_labels), key=non_noise_labels.count)
            print(f"HDBSCAN assigned to cluster {hdbscan_cluster}")
            return hdbscan_cluster, "HDBSCAN (fallback)"
            
        print("HDBSCAN also returned noise for all nearest neighbors.")
            
    except FileNotFoundError:
        print("HDBSCAN model not found, skipping to K-Means.")
    except Exception as e:
        print(f"Error using HDBSCAN: {e}")
    
    print("Trying final fallback: K-Means...")
    try:
        kmeans_model = joblib.load('kmeans.joblib')
        kmeans_cluster = kmeans_model.predict(X_new)[0]
        print(f"K-Means fallback classified as cluster {kmeans_cluster}")
        return kmeans_cluster, "K-Means (fallback)"
    except Exception as e:
        print(f"K-Means fallback error: {e}")
        print("All ensemble models failed. Returning DBSCAN noise classification as last resort.")
        return -1, "DBSCAN (noise - all fallbacks failed)"

@app.route('/predict', methods=['POST', 'OPTIONS'])	
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        scaled_data = request.get_json()
        print(f"Received scaled data: {scaled_data}")

        expected_params = column_order
        
        missing_params = [param for param in expected_params if param not in scaled_data]
        if missing_params:
             print(f"Validation Error: Missing parameters {missing_params}")
             return jsonify({
                 "error": "Missing parameters",
                 "missing": missing_params
             }), 400
            
        input_list = []
        for col in column_order:
             try:
                 value = float(scaled_data[col])
                 input_list.append(value)
             except (TypeError, ValueError):
                 print(f"Validation Error: Non-numeric value received for {col}: {scaled_data[col]}")
                 return jsonify({
                     "error": "Invalid parameter value",
                     "message": f"Parameter '{col}' must be a number."
                 }), 400
                 
        X_new = np.array([input_list])
        print(f"Created NumPy array for prediction: {X_new}")

        cluster, prediction_method = predict_cluster_ensemble(X_new)

        cluster_meanings = {
            -1: "NOISE/OUTLIER - Unusual water quality",
            0: "HIGH - Higher than normal levels",
            1: "NORMAL - Typical water quality"
        }

        if cluster not in cluster_meanings:
             print(f"Warning: DBSCAN logic produced unexpected cluster label: {cluster}. Defaulting to outlier.")
             cluster = -1 
             
        meaning = cluster_meanings[cluster]

        print("\n=== Prediction Log ===")
        print(f"Prediction Method: {prediction_method}")
        print("\nScaled Input Values (received from frontend):")
        for key, value in scaled_data.items():
            print(f"{key}: {value}")

        print(f"\nPredicted Cluster: {cluster} ({meaning})")
        print("===================\n")

        return jsonify({
            "cluster": int(cluster),
            "meaning": meaning,
            "input_values": scaled_data
        })

    except Exception as e:
        print(f"Error during prediction: {e}") 
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

load_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)