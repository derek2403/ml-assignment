import pandas as pd
import os

# Define the path to the data file relative to the script's location
DATA_FILE = 'data.csv'

def check_missing_data(file_path):
    """
    Checks a CSV file for missing data (NaN values) and reports counts and percentages.

    Args:
        file_path (str): The path to the CSV file.
    """
    print(f"\n--- Verifying Missing Data in: {file_path} ---")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded file. Shape: {df.shape}")

        # Calculate total missing values
        total_missing = df.isnull().sum().sum()

        if total_missing == 0:
            print("\nResult: No missing values found in the dataset. Verification successful!")
        else:
            print(f"\nResult: Found {total_missing} missing value(s) in the dataset.")
            print("\nMissing values per column:")
            missing_per_column = df.isnull().sum()
            missing_percentage = (missing_per_column / len(df)) * 100
            missing_info = pd.DataFrame({
                'Missing Values': missing_per_column,
                'Percentage': missing_percentage
            })
            # Only display columns that actually have missing values
            print(missing_info[missing_info['Missing Values'] > 0].to_string())

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    check_missing_data(DATA_FILE)