import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

# Suppress warnings if needed
# warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'data.csv' # Data file relative to this script's location (model/)
ORIGINAL_MODEL_FILE = 'dbscan.joblib' # Original model to load for reference
TUNED_MODEL_FILE = 'tuned.joblib' # Output file for the best model found

# --- Parameter Search Ranges ---
# Adjust these ranges based on your data exploration (e.g., k-distance plot)
# More points or finer steps increase runtime.
EPS_RANGE = np.linspace(0.5, 3.0, 10) # Example range for epsilon
MIN_SAMPLES_RANGE = range(5, 21, 5)   # Example range for min_samples

# --- Evaluation Criteria ---
MIN_CLUSTERS = 2         # Must find at least this many clusters (excluding noise)
MAX_NOISE_RATIO = 0.50 # Maximum allowed ratio of noise points (e.g., 50%)

# === Helper Functions ===

def load_data(file_path):
    """Load the dataset."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        exit(1)
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    # Assume data is features only for clustering
    if 'Potability' in df.columns:
        print("Note: 'Potability' column found, dropping for clustering.")
        X = df.drop('Potability', axis=1)
    else:
        X = df
    return X.values # Return as numpy array

def scale_data(X):
    """Scale data using StandardScaler."""
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling complete.")
    return X_scaled, scaler

def find_best_dbscan_params(X_scaled, eps_range, min_samples_range):
    """Perform grid search to find best DBSCAN parameters based on a combined score
       that balances Silhouette score and low noise ratio."""
    print("\n--- Starting DBSCAN Parameter Search ---")
    print(f"Optimizing for: High Silhouette Score * (1 - Noise Ratio)") # Clarify goal
    print(f"Constraints: Min Clusters={MIN_CLUSTERS}, Max Noise Ratio={MAX_NOISE_RATIO:.0%}")
    print(f"Searching eps in [{eps_range.min():.2f}, {eps_range.max():.2f}] (steps: {len(eps_range)})")
    print(f"Searching min_samples in {list(min_samples_range)}")

    best_combined_score = -1.0  # Initialize best combined score (can be negative)
    best_silhouette_at_best_combined = -1.0 # Track silhouette corresponding to best combined
    best_noise_at_best_combined = 1.0 # Track noise corresponding to best combined
    best_eps = None
    best_min_samples = None
    best_labels = None

    total_combinations = len(eps_range) * len(min_samples_range)
    current_combination = 0

    for eps in eps_range:
        for min_samples in min_samples_range:
            current_combination += 1
            print(f"  Testing ({current_combination}/{total_combinations}): eps={eps:.3f}, min_samples={min_samples}...", end="")

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
            labels = dbscan.fit_predict(X_scaled)

            # --- Evaluate Results ---
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            # Check basic validity
            if n_clusters < MIN_CLUSTERS:
                print(" Skipping (Too few clusters)")
                continue
            if noise_ratio > MAX_NOISE_RATIO:
                print(f" Skipping (Noise ratio {noise_ratio:.1%} > {MAX_NOISE_RATIO:.0%})")
                continue

            # Calculate Silhouette Score (on non-noise points)
            non_noise_mask = (labels != -1)
            n_non_noise_points = np.sum(non_noise_mask)

            score = -1.0 # Default silhouette if calculation fails
            if n_non_noise_points > 1 and len(set(labels[non_noise_mask])) > 1:
                try:
                    score = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                    # Calculate combined score only if silhouette is valid
                    combined_score = score * (1.0 - noise_ratio) # Balance score and noise
                    print(f" Clusters={n_clusters}, Noise={noise_ratio:.1%}, Silhouette={score:.4f}, Combined={combined_score:.4f}")

                    # Check if this is the best *combined* score so far
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_silhouette_at_best_combined = score # Store corresponding silhouette
                        best_noise_at_best_combined = noise_ratio # Store corresponding noise
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels # Keep track of labels for info
                        print(f"    *** New Best Found (Combined Score)! ***")

                except ValueError as e:
                    print(f" Error calculating silhouette: {e}")
            else:
                print(f" Skipping Evaluation (Clusters={n_clusters}, Non-Noise Points={n_non_noise_points})")

    print("\n--- Parameter Search Complete ---")
    if best_eps is not None:
        print(f"Best parameters found (based on maximizing Silhouette * (1 - Noise Ratio)):")
        print(f"  eps          : {best_eps:.4f}")
        print(f"  min_samples  : {best_min_samples}")
        print(f"  Best Combined Score: {best_combined_score:.4f}")
        print(f"    (Achieved Silhouette: {best_silhouette_at_best_combined:.4f})")
        print(f"    (Achieved Noise Ratio: {best_noise_at_best_combined:.1%})")
        # Info about the best clustering found
        best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        best_n_noise = list(best_labels).count(-1)
        print(f"  Resulting Clusters: {best_n_clusters}")
        print(f"  Resulting Noise   : {best_n_noise} ({best_n_noise/len(best_labels):.1%})")
    else:
        print("No suitable parameters found within the specified ranges and criteria.")

    # Return parameters that achieved the best combined score
    return best_eps, best_min_samples, best_combined_score # Return combined score

# === Main Execution Logic ===
def main():
    # --- Load Original Model (Optional Reference) ---
    try:
        original_model = joblib.load(ORIGINAL_MODEL_FILE)
        print(f"Original model '{ORIGINAL_MODEL_FILE}' loaded for reference:")
        print(f"  Original eps: {original_model.eps}")
        print(f"  Original min_samples: {original_model.min_samples}")
    except FileNotFoundError:
        print(f"Warning: Original model file '{ORIGINAL_MODEL_FILE}' not found. Skipping reference.")
    except Exception as e:
         print(f"Warning: Could not load original model '{ORIGINAL_MODEL_FILE}'. Error: {e}")

    # --- Load and Scale Data ---
    X_original = load_data(DATA_FILE)
    X_scaled, scaler = scale_data(X_original) # Using a new scaler fit on the data

    # --- Find Best Parameters ---
    best_eps, best_min_samples, best_score = find_best_dbscan_params(
        X_scaled,
        EPS_RANGE,
        MIN_SAMPLES_RANGE
    )

    # --- Fit and Save Final Tuned Model ---
    if best_eps is not None and best_min_samples is not None:
        print(f"\nFitting final DBSCAN model with eps={best_eps:.4f}, min_samples={best_min_samples}...")
        final_tuned_model = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean', n_jobs=-1)
        final_tuned_model.fit(X_scaled) # Fit on the entire scaled dataset

        # Double check the fit results (optional)
        final_labels = final_tuned_model.labels_
        final_n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        final_n_noise = list(final_labels).count(-1)
        print(f"Final model fit complete. Found {final_n_clusters} clusters and {final_n_noise} noise points.")

        print(f"\nSaving tuned model to '{TUNED_MODEL_FILE}'...")
        try:
            joblib.dump(final_tuned_model, TUNED_MODEL_FILE)
            print("Tuned model saved successfully.")
        except Exception as e:
            print(f"Error saving tuned model: {e}")
    else:
        print("\nNo best parameters found. No tuned model was saved.")

    print("\nFine-tuning script finished.")


if __name__ == "__main__":
    main()