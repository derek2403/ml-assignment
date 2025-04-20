import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
X_train_scaled = None

def load_model_and_scaler():
    """Load the DBSCAN model and create scaler using original data."""
    global model, scaler, X_train_scaled
    print("Loading model and preparing scaler...")
    model = joblib.load('dbscan.joblib')
    
    # Load original data to fit the scaler
    original_data = pd.read_csv('../data.csv')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(original_data)

def interpret_cluster(cluster_label, values, standardized_values):
    """Provide interpretation of the cluster assignment."""
    interpretation = {
        "cluster": int(cluster_label),
        "classification": "",
        "description": "",
        "characteristics": "",
        "standardized_values": {}
    }
    
    if cluster_label == -1:
        interpretation["classification"] = "NOISE/OUTLIER"
        interpretation["description"] = "This sample has unusual characteristics that don't fit well into any established cluster."
        interpretation["characteristics"] = "May require individual investigation."
    elif cluster_label == 0:
        interpretation["classification"] = "High Measurements"
        interpretation["description"] = "This sample shows higher-than-normal levels across parameters."
        interpretation["characteristics"] = "Higher values in pH, solids, chloramines, and turbidity."
    elif cluster_label == 1:
        interpretation["classification"] = "Normal/Baseline"
        interpretation["description"] = "This sample represents typical water quality conditions."
        interpretation["characteristics"] = "Values close to normal ranges."

    # Add standardized values
    for param, std_value in zip(values.keys(), standardized_values):
        interpretation["standardized_values"][param] = float(std_value)
    
    return interpretation

def predict_cluster(X_new_scaled):
    """Predict cluster for new data point using DBSCAN."""
    # Combine new point with training data
    X_combined = np.vstack([X_train_scaled, X_new_scaled])
    
    # Fit and predict on combined data
    labels = model.fit_predict(X_combined)
    
    # Return the label for the new point (last point)
    return labels[-1]

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict cluster based on water quality parameters."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Expected parameters
        expected_params = [
            'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        # Validate input parameters
        if not all(param in data for param in expected_params):
            missing_params = [param for param in expected_params if param not in data]
            return jsonify({
                "error": "Missing parameters",
                "missing": missing_params
            }), 400
        
        # Create input array in correct order
        values = {param: float(data[param]) for param in expected_params}
        X = np.array(list(values.values())).reshape(1, -1)
        
        # Scale input
        X_scaled = scaler.transform(X)
        
        # Get cluster prediction
        cluster = predict_cluster(X_scaled)
        
        # Get interpretation
        result = interpret_cluster(cluster, values, X_scaled[0])
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid parameter value",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# Initialize model and scaler when starting the server
load_model_and_scaler()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050) 