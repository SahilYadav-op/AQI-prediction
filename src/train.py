import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from data_preprocessing import load_data, clean_data, preprocess_features

def run_eda_and_training():
    # Ensure directories exist
    os.makedirs('../reports/figures', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    df = load_data('../data/raw/AQI_Data.csv')
    df = clean_data(df)
    
    # 2. EDA & Visualization
    print("Generating EDA visualizations...")
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../reports/figures/correlation_heatmap.png')
    plt.close()
    
    # Target Variable Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['PM 2.5'], kde=True, bins=30, color='blue')
    plt.title('Distribution of PM 2.5')
    plt.xlabel('PM 2.5')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('../reports/figures/pm25_distribution.png')
    plt.close()
    
    # 3. Model Training
    print("Preparing data for modeling...")
    X, y = preprocess_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    # Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted PM 2.5')
    plt.xlabel('Actual PM 2.5')
    plt.ylabel('Predicted PM 2.5')
    plt.tight_layout()
    plt.savefig('../reports/figures/actual_vs_predicted.png')
    plt.close()
    
    # 5. Save Model
    joblib.dump(model, '../models/linear_regression_model.pkl')
    print("Model saved to models/linear_regression_model.pkl")

if __name__ == "__main__":
    run_eda_and_training()
