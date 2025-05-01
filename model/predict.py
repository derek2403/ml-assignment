import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import os
import skfuzzy as fuzz
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app)

fcm_model = None
pca_model = None
scaler = None
# Input parameters
required_columns = [
    'Max Fecal Coliform', 'Min Temperature', 'Max BOD', 'Max Temperature', 
    'Max Conductivity', 'Min Fecal Coliform', 'Max Total Coliform', 
    'Min BOD', 'Min Dissolved Oxygen', 'Min Nitrate N + Nitrite N', 
    'Min pH', 'Max Nitrate N + Nitrite N'
]

def load_model():
    global fcm_model, pca_model, scaler
    print("Loading Fuzzy C-Means model and data...")
    try:
        # Load FCM model
        fcm_model = joblib.load('fuzzy_model_results_k3.joblib')
        print("- FCM model loaded.")
        
        # Load the training data for PCA and scaling reference
        final_df = pd.read_csv('final.csv')
        print(f"- Training data loaded with shape: {final_df.shape}")
        
        # Extract original features
        orig_cols = [col for col in final_df.columns if 'orig_' in col or 
                     col in ['Temperature_Range', 'Min Nitrate N + Nitrite N', 'Max Nitrate N + Nitrite N', 'STN_Code', 'Water_Body_Type']]
        
        # Extract PCA components
        pca_cols = [col for col in final_df.columns if col.startswith('PC') and col[2:].isdigit()]
        print(f"- Found {len(pca_cols)} PCA components")
        
        # Create a basic dataset of original features (removing 'orig_' prefix)
        cleaned_orig_cols = [col.replace('orig_', '') for col in orig_cols if not col in ['STN_Code', 'Water_Body_Type']]
        
        # Create scaler for original features
        orig_features = final_df[[col for col in orig_cols if not col in ['STN_Code', 'Water_Body_Type']]]
        scaler = StandardScaler()
        scaler.fit(orig_features)
        print("- Scaler fitted to original features.")
        
        # Fit a PCA model to transform from original features to PCA space
        # First, scale the original features
        X_orig_scaled = scaler.transform(orig_features)
        pca_model = PCA(n_components=len(pca_cols))
        pca_model.fit(X_orig_scaled)
        print(f"- PCA model fitted with {len(pca_cols)} components.")
        
        # Print FCM model info
        print("\nFuzzy C-Means Model Parameters:")
        print(f"Number of clusters: {fcm_model['n_clusters']}")
        print(f"Fuzziness parameter (m): {fcm_model['fuzziness_m']}")
        print(f"FPC (model quality): {fcm_model['fpc']:.4f}")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure model files and final.csv are present.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def predict_cluster_fcm(X_pca_scaled):
    """Predict cluster using FCM model"""
    if fcm_model is None:
        raise RuntimeError("FCM model not loaded.")
    
    # Get the FCM centers from the model
    centers = fcm_model['centers']
    
    # Transpose for skfuzzy format (features x samples)
    X_pca_t = X_pca_scaled.T
    
    print(f"Centers shape: {centers.shape}")
    print(f"Input data shape: {X_pca_t.shape}")
    
    # Calculate membership using FCM formula
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_pca_t, centers, fcm_model['fuzziness_m'],
        error=0.005, maxiter=1000
    )
    
    # Get membership values for the sample
    memberships = u[:, 0]
    
    # Get the cluster with highest membership
    hard_label = np.argmax(memberships)
    
    print(f"FCM memberships: {memberships}")
    print(f"Assigned to cluster {hard_label} with membership value: {memberships[hard_label]:.4f}")
    
    return hard_label, memberships

@app.route('/predict', methods=['POST', 'OPTIONS'])	
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        input_data = request.get_json()
        print(f"Received input data: {input_data}")

        # Validate required input parameters
        missing_params = [param for param in required_columns if param not in input_data]
        if missing_params:
            print(f"Validation Error: Missing parameters {missing_params}")
            return jsonify({
                "error": "Missing parameters",
                "missing": missing_params
            }), 400
        
        # Create complete input data with calculated Temperature_Range
        complete_input_data = input_data.copy()
        
        # Calculate Temperature_Range from Max and Min Temperature
        try:
            max_temp = float(input_data['Max Temperature'])
            min_temp = float(input_data['Min Temperature'])
            temp_range = max_temp - min_temp
            
            # Add the calculated Temperature_Range to the input data
            complete_input_data['Temperature_Range'] = temp_range
            print(f"Calculated Temperature_Range: {temp_range}")
        except (TypeError, ValueError) as e:
            print(f"Error calculating Temperature_Range: {e}")
            return jsonify({
                "error": "Invalid temperature values",
                "message": "Could not calculate Temperature_Range from provided temperature values."
            }), 400
        
        # Create input list in the correct order for original features
        features_list = []
        
        # Get all original feature columns (match keys in complete_input_data to columns used when training)
        feature_keys = ['Max Fecal Coliform', 'Min Temperature', 'Max BOD', 'Max Temperature', 
                        'Max Conductivity', 'Temperature_Range', 'Min Fecal Coliform', 
                        'Max Total Coliform', 'Min BOD', 'Min Dissolved Oxygen', 
                        'Min Nitrate N + Nitrite N', 'Min pH', 'Max Nitrate N + Nitrite N']
        
        for key in feature_keys:
            try:
                value = float(complete_input_data[key])
                features_list.append(value)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error with feature {key}: {e}")
                return jsonify({
                    "error": f"Problem with feature {key}",
                    "message": str(e)
                }), 400
        
        # Convert to numpy array
        features_array = np.array([features_list])
        print(f"Original features array shape: {features_array.shape}")
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        print(f"Scaled features: {features_scaled}")
        
        # Transform to PCA space
        features_pca = pca_model.transform(features_scaled)
        print(f"PCA transformed features shape: {features_pca.shape}")
        
        # Predict using FCM
        cluster, memberships = predict_cluster_fcm(features_pca)

        # Define correct meanings for the clusters
        cluster_meanings = {
            0: "High Contamination - Poor water quality",
            1: "Moderate Contamination - Acceptable water quality",
            2: "Low Contamination - Good water quality"
        }
            
        meaning = cluster_meanings.get(cluster, "Unknown cluster type")

        # Calculate certainty based on the highest membership value
        certainty = float(memberships[cluster])

        print("\n=== Prediction Log ===")
        print("Prediction Method: Fuzzy C-Means")
        print("\nInput Values:")
        for key, value in input_data.items():
            print(f"{key}: {value}")
        print(f"Temperature_Range (calculated): {temp_range}")

        print(f"\nPredicted Cluster: {cluster} ({meaning})")
        print(f"Certainty: {certainty:.4f}")
        print("===================\n")

        # Include both original inputs and derived features in response
        response_data = input_data.copy()
        response_data['Temperature_Range'] = temp_range

        return jsonify({
            "cluster": int(cluster),
            "meaning": meaning,
            "certainty": certainty,
            "memberships": memberships.tolist(),
            "input_values": response_data,
            "calculated_values": {
                "Temperature_Range": temp_range
            }
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