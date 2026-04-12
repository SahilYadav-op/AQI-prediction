import joblib
import pandas as pd
import argparse

def predict(input_data):
    """Predicts PM 2.5 giving input data dictionary."""
    model = joblib.load('../models/linear_regression_model.pkl')
    
    # Ensure expected columns
    expected_cols = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Reorder or fill columns just in case
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0 # fallback
            
    df_input = df_input[expected_cols]
    
    prediction = model.predict(df_input)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PM 2.5 AQI")
    parser.add_argument('--T', type=float, default=20.0, help='Average Temperature')
    parser.add_argument('--TM', type=float, default=30.0, help='Maximum Temperature')
    parser.add_argument('--Tm', type=float, default=15.0, help='Minimum Temperature')
    parser.add_argument('--SLP', type=float, default=1010.0, help='Sea Level Pressure')
    parser.add_argument('--H', type=float, default=60.0, help='Humidity')
    parser.add_argument('--VV', type=float, default=1.0, help='Visibility')
    parser.add_argument('--V', type=float, default=5.0, help='Average Wind Speed')
    parser.add_argument('--VM', type=float, default=15.0, help='Maximum Sustained Wind Speed')
    
    args = parser.parse_args()
    
    input_features = {
        'T': args.T,
        'TM': args.TM,
        'Tm': args.Tm,
        'SLP': args.SLP,
        'H': args.H,
        'VV': args.VV,
        'V': args.V,
        'VM': args.VM
    }
    
    pred_result = predict(input_features)
    print(f"Predicted PM 2.5: {pred_result:.2f}")

