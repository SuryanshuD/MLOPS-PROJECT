import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# === Ensure necessary directories exist ===
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# === Setup logging (force=True fixes Streamlit issue) ===
logging.basicConfig(
    filename=os.path.join('logs', 'app.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    force=True  # <-- Required for Streamlit to respect logging config
)

def train_and_evaluate_models():
    df = pd.read_csv('Walmart_Sales.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = pd.get_dummies(df, columns=['Store', 'Holiday_Flag'], drop_first=True)

    y = np.log(df['Weekly_Sales'])
    X = df.drop(columns=['Weekly_Sales', 'Date'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    trained_models = {}

    def evaluate(name, model, X_val, X_test):
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        results[name] = (test_mse, test_r2)
        trained_models[name] = model
        log_msg = f"{name} - Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}"
        logging.info(log_msg)
        print(log_msg)  # also print in Streamlit
        plot_model_performance(name, y_test, test_pred)

    # === Train Models ===
    normal_lr = LinearRegression().fit(X_train_scaled, y_train)
    evaluate('Normal LR', normal_lr, X_val_scaled, X_test_scaled)

    ridge = Ridge()
    ridge_grid = GridSearchCV(Pipeline([
        ('scaler', StandardScaler()), 
        ('ridge', ridge)
    ]), {'ridge__alpha': [0.1, 1.0, 10.0]}, cv=5)
    ridge_grid.fit(X_train, y_train)
    evaluate('Ridge', ridge_grid, X_val, X_test)

    lasso = Lasso()
    lasso_grid = GridSearchCV(Pipeline([
        ('scaler', StandardScaler()), 
        ('lasso', lasso)
    ]), {'lasso__alpha': [0.001, 0.01, 0.1]}, cv=5)
    lasso_grid.fit(X_train, y_train)
    evaluate('Lasso', lasso_grid, X_val, X_test)

    elastic = ElasticNet()
    elastic_grid = GridSearchCV(Pipeline([
        ('scaler', StandardScaler()), 
        ('elastic', elastic)
    ]), {
        'elastic__alpha': [0.01, 0.1],
        'elastic__l1_ratio': [0.5, 0.9]
    }, cv=5)
    elastic_grid.fit(X_train, y_train)
    evaluate('ElasticNet', elastic_grid, X_val, X_test)

    # === Polynomial Regression ===
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    poly_pipeline.fit(X_train, y_train)
    evaluate('Polynomial', poly_pipeline, X_val, X_test)

    # ✅ Save polynomial model with absolute path
    poly_model_path = os.path.join(os.getcwd(), 'models', 'polynomial_model.pkl')
    joblib.dump(poly_pipeline, poly_model_path)
    logging.info(f"Polynomial model saved to {poly_model_path}")
    print(f"✅ Polynomial model saved to {poly_model_path}")

    # === Bayesian Ridge ===
    bayes = BayesianRidge().fit(X_train_scaled, y_train)
    evaluate('Bayesian Ridge', bayes, X_val_scaled, X_test_scaled)

    # ✅ Save best model (by R²)
    save_best_model(results, trained_models)

    # ✅ Auto-generate new_data.csv for inference testing
    generate_new_data_sample(X_test, df)

    return results

def save_best_model(results, models_dict):
    best_model_name = max(results, key=lambda k: results[k][1])
    best_model = models_dict[best_model_name]
    filename = os.path.join('models', f"{best_model_name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(best_model, filename)
    logging.info(f"Best model '{best_model_name}' saved to {filename}")
    print(f"✅ Best model '{best_model_name}' saved to {filename}")

def plot_model_performance(model_name, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, color='blue', edgecolors='black', alpha=0.7)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_name} - True vs Predicted")
    plt.tight_layout()
    plt.show()

def generate_new_data_sample(X_test, _):
    try:
        # Reload raw data to get original column names (not encoded)
        raw_df = pd.read_csv('Walmart_Sales.csv')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=True)

        cols_needed = ['Date', 'Temperature', 'Fuel_Price', 'Store', 'Holiday_Flag', 'CPI', 'Unemployment']
        sample = raw_df[cols_needed].iloc[:len(X_test)].copy()
        sample['Date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(len(sample)), unit='W')
        sample.to_csv('new_data.csv', index=False)

        logging.info("✅ new_data.csv generated successfully.")
        print("📦 Generated sample new_data.csv")

    except Exception as e:
        logging.error(f"Failed to generate new_data.csv: {e}")
        print(f"❌ Failed to generate new_data.csv: {e}")


# Run manually (for CLI debug)
if __name__ == "__main__":
    print("🔁 Training models...")
    train_and_evaluate_models()
    print("✅ Training complete.")
