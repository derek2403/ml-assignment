import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler  # Added for scaling
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None # DBSCAN model
X_train = None # Training data for neighbor search
X_train_scaled = None # Scaled version of training data
# Ensure column_order matches the order expected AFTER frontend scaling if keys change
# Using the likely frontend keys (camelCase/PascalCase)
column_order = [
    'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

def load_model():
    """Load the DBSCAN model and training data."""
    global model, X_train, X_train_scaled
    print("Loading DBSCAN model and data...")
    try:
        model = joblib.load('dbscan.joblib')
        print("- DBSCAN model loaded.")

        # Check if other models exist
        if os.path.exists('hdbscan.joblib'):
            print("- HDBSCAN model found (will be used as fallback 1)")
        else:
            print("- Warning: HDBSCAN model not found at 'hdbscan.joblib'")
            
        if os.path.exists('kmeans.joblib'):
            print("- K-Means model found (will be used as fallback 2)")
        else:
            print("- Warning: K-Means model not found at 'kmeans.joblib'")

        # Load training data
        X_train = pd.read_csv('data.csv')
        
        # Check if Potability column exists and drop it if present
        if 'Potability' in X_train.columns:
            X_train = X_train.drop('Potability', axis=1)
            print("- Dropped Potability column from training data.")
        
        # Scale the training data using the same method as during model training
        # This ensures X_train_scaled and the frontend data are on the same scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("- Training data loaded and scaled.")
        
        # Crucially, DBSCAN model.labels_ should correspond to the rows in X_train
        if len(model.labels_) != len(X_train):
             print(f"Warning: Mismatch between DBSCAN model labels ({len(model.labels_)}) and training data rows ({len(X_train)}). Ensure model was trained on this data.")
             # Consider exiting if this is critical
             # exit(1)

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
    """Predict cluster using DBSCAN logic (finding nearest neighbors). 
       Assumes X_new is already scaled as sent by frontend.
       Fits NearestNeighbors on the scaled X_train data."""
    # Ensure X_train_scaled is loaded
    if X_train_scaled is None:
        raise RuntimeError("Scaled training data (X_train_scaled) not loaded.")
        
    # Fit NearestNeighbors on the SCALED training data - this is the key fix
    # Now both X_train_scaled and X_new are on the same scale
    nbrs = NearestNeighbors(n_neighbors=model.min_samples, radius=model.eps)
    nbrs.fit(X_train_scaled)
    
    # Find neighbors within eps radius using the input X_new
    distances, indices = nbrs.radius_neighbors(X_new)
    
    print(f"Number of neighbors found within eps radius: {len(indices[0])}")
    
    # --- Rest of the logic remains the same ---
    if len(distances[0]) == 0:
        print("No neighbors found within eps radius.")
        return -1
        
    # Get the cluster labels of the neighbors from the loaded model
    try:
        neighbor_labels = model.labels_[indices[0]]
        print(f"Neighbor labels: {neighbor_labels}")
    except IndexError:
        print(f"Error: Indices {indices[0]} out of bounds for model labels (length {len(model.labels_)}).")
        return -1 # Treat as outlier if indices are invalid
        
    if all(label == -1 for label in neighbor_labels):
        print("All neighbors are noise points.")
        return -1
        
    non_noise_labels = [label for label in neighbor_labels if label != -1]
    if non_noise_labels:
        most_common_label = max(set(non_noise_labels), key=non_noise_labels.count)
        print(f"Found non-noise neighbors with labels: {non_noise_labels}. Most common: {most_common_label}")
        return most_common_label
        
    # This case should be covered by the all(label == -1) check, but for safety:
    print("Found neighbors, but only noise labels.")
    return -1

def predict_cluster_ensemble(X_new):
    """Three-model cascade ensemble: DBSCAN → HDBSCAN → K-Means
       This approach prioritizes density-based clustering but ensures
       every input gets a cluster assignment (no noise)."""
    
    # Step 1: Try DBSCAN first
    print("\nStarting ensemble prediction cascade...")
    dbscan_cluster = predict_cluster_dbscan(X_new)
    
    # If DBSCAN classifies as non-noise, return that result
    if dbscan_cluster != -1:
        print("Using DBSCAN classification.")
        return dbscan_cluster, "DBSCAN"
    
    # Step 2: If DBSCAN returns noise, try HDBSCAN 
    print("DBSCAN returned noise (-1), trying HDBSCAN...")
    try:
        # Load HDBSCAN model
        hdbscan_model = joblib.load('hdbscan.joblib')
        
        # Get distances to nearest neighbors to find the "soft" cluster assignment
        # HDBSCAN doesn't have a direct predict method, we need to find nearest points
        nbrs = NearestNeighbors(n_neighbors=5)
        nbrs.fit(X_train_scaled)
        distances, indices = nbrs.kneighbors(X_new)
        
        # Get the cluster labels of the 5 nearest neighbors
        neighbor_labels = hdbscan_model.labels_[indices[0]]
        print(f"HDBSCAN nearest neighbor labels: {neighbor_labels}")
        
        # Filter out noise points (-1)
        non_noise_labels = [label for label in neighbor_labels if label != -1]
        
        # If we have at least one non-noise neighbor, use the most common cluster
        if non_noise_labels:
            hdbscan_cluster = max(set(non_noise_labels), key=non_noise_labels.count)
            print(f"HDBSCAN assigned to cluster {hdbscan_cluster}")
            return hdbscan_cluster, "HDBSCAN (fallback)"
            
        print("HDBSCAN also returned noise for all nearest neighbors.")
            
    except FileNotFoundError:
        print("HDBSCAN model not found, skipping to K-Means.")
    except Exception as e:
        print(f"Error using HDBSCAN: {e}")
    
    # Step 3: Final fallback - use K-Means which always assigns a cluster
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
    """Endpoint to predict cluster based on water quality parameters.
       Assumes input data is pre-scaled by the frontend."""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        # Get JSON data from request (assumed to be scaled)
        scaled_data = request.get_json()
        print(f"Received scaled data: {scaled_data}")

        # Expected parameters (using names likely sent from frontend)
        expected_params = column_order
        
        # Validate presence of parameters
        missing_params = [param for param in expected_params if param not in scaled_data]
        if missing_params:
             print(f"Validation Error: Missing parameters {missing_params}")
             return jsonify({
                 "error": "Missing parameters",
                 "missing": missing_params
             }), 400
            
        # --- Input Validation Passed --- 
        
        # Create input array from the pre-scaled data in the correct order
        input_list = []
        for col in column_order:
             try:
                 # Attempt to convert to float, handle potential errors if not numeric
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

        # Predict cluster using DBSCAN only
        cluster, prediction_method = predict_cluster_ensemble(X_new)

        # Define cluster meanings
        cluster_meanings = {
            -1: "NOISE/OUTLIER - Unusual water quality",
            0: "HIGH - Higher than normal levels",
            1: "NORMAL - Typical water quality"
        }

        # Check if cluster is valid (-1, 0, or 1)
        if cluster not in cluster_meanings:
             print(f"Warning: DBSCAN logic produced unexpected cluster label: {cluster}. Defaulting to outlier.")
             cluster = -1 
             
        meaning = cluster_meanings[cluster]

        # Log the values
        print("\n=== Prediction Log ===")
        print(f"Prediction Method: {prediction_method}")
        print("\nScaled Input Values (received from frontend):")
        # Log received data directly
        for key, value in scaled_data.items():
            print(f"{key}: {value}")

        print(f"\nPredicted Cluster: {cluster} ({meaning})")
        print("===================\n")

        return jsonify({
            "cluster": int(cluster),
            "meaning": meaning,
            "input_values": scaled_data # Return the scaled input as received
        })

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Error during prediction: {e}") 
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# Initialize model when starting the server
load_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)