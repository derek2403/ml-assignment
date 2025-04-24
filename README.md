# Water Quality Clustering Analysis and Prediction

This project analyzes water quality data using various clustering algorithms and provides a web application for users to get real-time cluster predictions based on input parameters.

## Overview

The system consists of:
1.  **Data Cleaning & Preparation**: Scripts to clean the raw water quality data, handle missing values and outliers.
2.  **Clustering Model Training**: Individual scripts for training 8 different clustering algorithms (K-Means, Hierarchical, DBSCAN, HDBSCAN, GMM, Spectral, SOM, Fuzzy C-Means) on the dataset. Each script performs parameter tuning (where applicable), saves the trained model, and generates visualizations.
3.  **Model Comparison**: A script to evaluate and compare the performance of all trained models using standard clustering metrics.
4.  **Prediction API**: A Flask-based backend API that loads selected models (DBSCAN, HDBSCAN, K-Means) and uses an ensemble approach to predict the cluster (NORMAL or HIGH) for new water quality inputs.
5.  **Web Frontend**: A Next.js application providing an interactive interface with sliders for users to input water parameters and see the predicted quality cluster.

## Technology Stack

*   **Frontend**: Next.js (React Framework), Tailwind CSS - Deployed on Vercel.
*   **Backend**: Python, Flask, scikit-learn, Pandas, NumPy, Joblib, HDBSCAN - Deployed via Docker (using DockerHub image) on a Cloud Virtual Machine (CVM).
*   **Containerization**: Docker, Docker Compose.

## Directory Structure

```
├── compare.py                # Script to compare model performance
├── data.csv                  # Original dataset
├── clean.py                  # Script for data cleaning
├── check.py                  # Script to check data ranges/missing values
├── frontend/                 # Next.js frontend application
│   ├── pages/                # Next.js pages and API routes
│   ├── public/
│   ├── styles/
│   └── ...                   # Other Next.js config files (package.json, etc.)
├── model/                    # Flask backend API
│   ├── predict.py            # Main Flask API script for prediction
│   ├── requirements.txt      # Backend Python dependencies
│   ├── Dockerfile            # Dockerfile for building the backend image
│   ├── docker-compose.yml    # Docker Compose file (optional, for local testing)
│   ├── dbscan.joblib         # Saved models used by the prediction API
│   ├── hdbscan.joblib
│   └── kmeans.joblib
├── dbscan/                   # DBSCAN specific training files
│   ├── dbscan.py             # Script to train DBSCAN
│   ├── dbscan_model*.joblib  # Saved model(s) from training
│   └── plots/                # Generated plots (k-distance, PCA)
├── fuzzy/                    # Fuzzy C-Means specific training files
│   ├── fuzzy.py              # Script to train Fuzzy C-Means
│   ├── fuzzy_model.joblib    # Saved model results
│   └── plots/                # Generated plots (validity indices, PCA)
├── gmm/                      # Gaussian Mixture Model specific training files
│   ├── gmm.py                # Script to train GMM
│   ├── gmm_model.joblib      # Saved model
│   └── plots/                # Generated plots (BIC/AIC, PCA)
├── hdbscan/                  # HDBSCAN specific training files
│   ├── train.py              # Script to train HDBSCAN
│   ├── hdbscan_model.joblib  # Saved model
│   └── plots/                # Generated plots (PCA)
├── hierarchical/             # Hierarchical Clustering specific files
│   ├── hierarchical.py       # Script to train Hierarchical Clustering
│   ├── hierarchical*.joblib  # Saved model(s)
│   └── plots/                # Generated plots (dendrogram, PCA)
├── kmeans/                   # K-Means specific training files
│   ├── kmeans.py             # Script to train K-Means
│   ├── kmeans_model.joblib   # Saved model
│   └── plots/                # Generated plots (elbow, silhouette, PCA)
├── som/                      # Self-Organizing Map specific files
│   ├── som.py                # Script to train SOM
│   ├── som_model.joblib      # Saved SOM object
│   └── plots/                # Generated plots (U-Matrix, PCA)
├── spectral/                 # Spectral Clustering specific files
│   ├── spectral.py           # Script to train Spectral Clustering
│   ├── spectral_model*.joblib # Saved model(s)
│   └── plots/                # Generated plots (silhouette, PCA)
└── plots/                    # Preprocessing output graphs
```

**Key Points:**
*   Each algorithm used for training resides in its own dedicated directory (e.g., `/kmeans`, `/dbscan`).
*   Inside each algorithm directory, you'll find the Python script (`*.py`) used for training, the resulting saved model (`*.joblib`), and a `/plots` subdirectory containing visualizations generated during training (like PCA plots, evaluation metrics plots, etc.).
*   The `/model` directory contains the backend Flask API code (`predict.py`), its dependencies (`requirements.txt`), and the specific model files (`dbscan.joblib`, `hdbscan.joblib`, `kmeans.joblib`) it loads for the ensemble prediction.
*   The `/frontend` directory contains the Next.js web application code.

## Running the Application

**Prerequisites:**
*   Python 3.10+ and Pip
*   Node.js and npm (or yarn)
*   Docker and Docker Compose (or Colima/Docker Desktop)

**Terminal 1: Backend API Setup (Flask)**

1.  Navigate to the backend directory:
    ```bash
    cd model
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Flask prediction server:
    ```bash
    python predict.py
    ```
    The server will start, typically on `http://0.0.0.0:5050`. It will load the necessary models upon startup. *Note: Ensure `dbscan.joblib`, `hdbscan.joblib`, and `kmeans.joblib` are present in this directory.*

**Terminal 2: Frontend Setup (Next.js)**

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install Node.js dependencies:
    ```bash
    npm install
    # or: yarn install
    ```
3.  Start the Next.js development server:
    ```bash
    npm run dev
    # or: yarn dev
    ```
    The frontend application will usually be available at `http://localhost:3000`.

**Accessing the Application:**
Open your web browser and go to `http://localhost:3000`. You should see the water quality analysis interface. Use the sliders to input parameters and click "Analyze Water Quality" to get a prediction from the backend API.