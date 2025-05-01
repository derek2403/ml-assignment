import pandas as pd
import csv # Needed for quoting options

# --- Configuration ---
file_path = 'dataset.csv'
output_path = 'dataset.csv' # Overwrite the original file
# Columns where internal newlines should be replaced
columns_to_clean = ['Name of Monitoring Location', 'State Name']
# What to replace the newline character with (e.g., a space, a comma-space)
replacement_char = ' '
# --- End Configuration ---

print(f"Attempting to clean file: {file_path}")
print(f"Newlines in columns {columns_to_clean} will be replaced with '{replacement_char}'")

try:
    # Read the CSV, pandas handles the quoted newlines automatically
    df = pd.read_csv(file_path)
    print(f"Successfully read {len(df)} rows.")

    # Replace '\n' with the specified character in the target columns
    for col in columns_to_clean:
        if col in df.columns:
            # Ensure column is string type to use .str accessor, fill NA to avoid errors
            df[col] = df[col].fillna('').astype(str).str.replace('\n', replacement_char, regex=False)
            print(f"Cleaned column: {col}")
        else:
            print(f"Warning: Column '{col}' not found in {file_path}")

    # Save the cleaned data back to the CSV
    # quoting=csv.QUOTE_MINIMAL ensures quotes are only added when necessary
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nFile '{output_path}' has been updated successfully.")
    print("\nFirst 5 rows of the cleaned data:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred during processing: {e}")
