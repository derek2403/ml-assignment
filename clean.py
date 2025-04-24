
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os

def load_data(file_path='water_potability.csv'):
    """
    Load the dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """
    Basic data exploration
    """
    print("\n--- Data Exploration ---")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nSummary statistics:")
    print(df.describe())
    
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_info)
    
    return missing_info

def handle_missing_values(df, strategy='knn'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values ('remove', 'mean', 'median', 'knn')
        
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with handled missing values
    """
    print(f"\n--- Handling Missing Values (strategy: {strategy}) ---")
    
    # Make a copy of the dataframe
    df_clean = df.copy()
    
    if strategy == 'remove':
        # Remove rows with missing values
        df_clean = df_clean.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")
    
    elif strategy == 'mean':
        # Impute missing values with mean
        df_clean = df_clean.fillna(df_clean.mean())
        print("Imputed missing values with mean")
    
    elif strategy == 'median':
        # Impute missing values with median
        df_clean = df_clean.fillna(df_clean.median())
        print("Imputed missing values with median")
    
    elif strategy == 'knn':
        # Impute missing values with KNN
        print("Imputing missing values with KNN...")
        imputer = KNNImputer(n_neighbors=5)
        df_clean_array = imputer.fit_transform(df_clean)
        df_clean = pd.DataFrame(df_clean_array, columns=df_clean.columns)
        print("Imputed missing values with KNN")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Check if there are still missing values
    missing_after = df_clean.isnull().sum().sum()
    if missing_after > 0:
        print(f"Warning: There are still {missing_after} missing values!")
    else:
        print("No missing values remaining")
    
    return df_clean

def detect_outliers(df, method='zscore', threshold=3.0):
    """
    Detect outliers in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    method : str
        Method for outlier detection ('zscore', 'iqr')
    threshold : float
        Threshold for z-score method
        
    Returns:
    --------
    outliers_mask : pandas Series
        Boolean mask indicating outliers (True for outliers)
    """
    print(f"\n--- Detecting Outliers (method: {method}) ---")
    
    outliers_mask = pd.Series(False, index=df.index)
    
    if method == 'zscore':
        # Z-score method
        for column in df.columns:
            if df[column].dtype != 'object' and column != 'Potability':  # Skip non-numeric and target columns
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                column_outliers = z_scores > threshold
                print(f"Column {column}: {column_outliers.sum()} outliers detected")
                outliers_mask = outliers_mask | column_outliers
    
    elif method == 'iqr':
        # IQR method
        for column in df.columns:
            if df[column].dtype != 'object' and column != 'Potability':  # Skip non-numeric and target columns
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                print(f"Column {column}: {column_outliers.sum()} outliers detected")
                outliers_mask = outliers_mask | column_outliers
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    total_outliers = outliers_mask.sum()
    print(f"Total records with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)")
    
    return outliers_mask

def handle_outliers(df, outliers_mask, strategy='cap'):
    """
    Handle outliers in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    outliers_mask : pandas Series
        Boolean mask indicating outliers (True for outliers)
    strategy : str
        Strategy for handling outliers ('remove', 'cap')
        
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with handled outliers
    """
    print(f"\n--- Handling Outliers (strategy: {strategy}) ---")
    
    # Make a copy of the dataframe
    df_clean = df.copy()
    
    if strategy == 'remove':
        # Remove outliers
        df_clean = df_clean[~outliers_mask]
        print(f"Removed {outliers_mask.sum()} outliers")
    
    elif strategy == 'cap':
        # Cap outliers at a threshold
        for column in df_clean.columns:
            if df_clean[column].dtype != 'object' and column != 'Potability':  # Skip non-numeric and target columns
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap lower values
                df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
                
                # Cap upper values
                df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        
        print("Capped outliers to 1.5*IQR from quartiles")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def scale_features(df, method='standard'):
    """
    Scale features in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    method : str
        Method for feature scaling ('standard', 'minmax')
        
    Returns:
    --------
    df_scaled : pandas DataFrame
        DataFrame with scaled features
    """
    print(f"\n--- Scaling Features (method: {method}) ---")
    
    # Make a copy of the dataframe
    df_scaled = df.copy()
    
    # Extract target variable (if present)
    if 'Potability' in df_scaled.columns:
        y = df_scaled['Potability'].copy()
        X = df_scaled.drop('Potability', axis=1)
    else:
        X = df_scaled.copy()
        y = None
    
    if method == 'standard':
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print("Features standardized (mean=0, std=1)")
    
    elif method == 'minmax':
        # Min-Max scaling
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print("Features scaled to range [0, 1]")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Re-add target variable (if originally present)
    if y is not None:
        df_scaled['Potability'] = y
    
    return df_scaled

def analyze_correlations(df):
    """
    Analyze feature correlations
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
        
    Returns:
    --------
    corr_matrix : pandas DataFrame
        Correlation matrix
    """
    print("\n--- Analyzing Feature Correlations ---")
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Print correlations with target (if present)
    if 'Potability' in corr_matrix.columns:
        print("\nCorrelations with target variable (Potability):")
        target_corr = corr_matrix['Potability'].sort_values(ascending=False)
        print(target_corr)
    
    # Find highly correlated features
    threshold = 0.7
    high_corr_features = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_features:
        print("\nHighly correlated features (|r| > 0.7):")
        for f1, f2, corr in high_corr_features:
            print(f"  {f1} -- {f2}: {corr:.3f}")
    else:
        print("\nNo highly correlated features found (|r| > 0.7)")
    
    return corr_matrix

def plot_distributions(df, output_dir):
    """
    Plot distributions of features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    output_dir : str
        Directory to save the plots
    """
    print("\n--- Plotting Feature Distributions ---")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms for each feature
    for column in df.columns:
        if df[column].dtype != 'object':  # Skip non-numeric columns
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'dist_{column}.png'))
            plt.close()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # If target variable is present, plot boxplots by target
    if 'Potability' in df.columns:
        for column in df.columns:
            if column != 'Potability' and df[column].dtype != 'object':
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Potability', y=column, data=df)
                plt.title(f'{column} by Potability')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'boxplot_{column}_by_potability.png'))
                plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/plots', exist_ok=True)
    
    # Load the data
    df = load_data()
    
    # Explore the data
    explore_data(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df, strategy='knn')
    
    # Detect outliers
    outliers_mask = detect_outliers(df_clean, method='iqr')
    
    # Handle outliers
    df_clean = handle_outliers(df_clean, outliers_mask, strategy='cap')
    
    # Analyze correlations
    analyze_correlations(df_clean)
    
    # Scale features
    df_scaled = scale_features(df_clean, method='standard')
    
    # Plot distributions
    plot_distributions(df_clean, 'data/plots')
    
    # Save the cleaned data
    df_clean.to_csv('data/cleaned.csv', index=False)
    print("\nCleaned data saved to data/cleaned.csv")
    
    # Save the scaled data
    df_scaled.to_csv('data/scaled.csv', index=False)
    print("Scaled data saved to data/scaled.csv")
    
    print("\nData cleaning completed!")


if __name__ == "__main__":
    main() 