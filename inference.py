import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = 'models/polynomial_model.pkl'
DATA_PATH = 'new_data.csv'

def preprocess_input_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # One-hot encode Store (assuming model was trained on Stores 1 to 45)
    store_dummies = pd.get_dummies(df['Store'], prefix='Store')
    
    # Add missing store columns with 0 (for all stores from 1 to 45)
    all_stores = [f'Store_{i}' for i in range(1, 46)]
    for store_col in all_stores:
        if store_col not in store_dummies.columns:
            store_dummies[store_col] = 0
    
    # Reorder columns so they are always the same order
    store_dummies = store_dummies[all_stores]
    
    # Similarly for Holiday_Flag
    holiday_dummies = pd.get_dummies(df['Holiday_Flag'], prefix='Holiday_Flag')
    if 'Holiday_Flag_1' not in holiday_dummies.columns:
        holiday_dummies['Holiday_Flag_1'] = 0
    
    # Assemble final dataframe
    X = pd.concat([
        df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Year']],
        store_dummies,
        holiday_dummies[['Holiday_Flag_1']]
    ], axis=1)
    
    return X


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)

def make_predictions():
    new_df = pd.read_csv(DATA_PATH)
    X_new = preprocess_input_data(new_df)

    model = load_model(MODEL_PATH)

    # Align columns exactly
    expected_features = model.feature_names_in_


    # Add missing columns if any (should be none now)
    for col in expected_features:
        if col not in X_new.columns:
            X_new[col] = 0
    
    X_new = X_new[expected_features]

    y_log_pred = model.predict(X_new)
    y_pred = np.exp(y_log_pred)
    new_df['Predicted_Weekly_Sales'] = y_pred

    print(new_df[['Predicted_Weekly_Sales']])



if __name__ == '__main__':
    
    make_predictions()
