import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the AQI dataset."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Cleans the AQI dataset.
    Handles NaN values and drops unnecessary rows if any.
    """
    # PM 2.5 is our target variable. Let's check for nulls.
    # The view of the file showed some nan values in PM 2.5 and others.
    # We will drop rows where the target 'PM 2.5' is NaN
    
    if 'PM 2.5' in df.columns:
        df = df.dropna(subset=['PM 2.5'])
        
    # Fill remaining NaNs in features with median
    # For a robust approach, we use median to avoid outlier effects
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    return df

def preprocess_features(df):
    """
    Separates features and targets, and prepares for modeling.
    """
    # Features (X) and Target (y)
    X = df.drop('PM 2.5', axis=1)
    y = df['PM 2.5']
    return X, y

if __name__ == "__main__":
    df = load_data('../data/raw/AQI_Data.csv')
    df_clean = clean_data(df)
    print("Data cleaned successfully. Shape:", df_clean.shape)
    
