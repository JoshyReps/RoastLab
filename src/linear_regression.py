"""
Algorithm 1: Ordinary Least Squares (OLS) Linear Regression
Business Task: Cold Brew Daily Batch Demand Predictor
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def generate_cold_brew_data(n_samples: int = 180, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    temps = np.random.uniform(18.0, 38.0, size=(n_samples, 1))
    liters = 0.95 * temps[:, 0] - 4.5 + np.random.normal(0, 1.8, size=n_samples)
    liters = np.clip(liters, 8.0, 45.0)
    
    return pd.DataFrame({
        "forecast_peak_temp_c": np.round(temps[:, 0], 1),
        "cold_brew_liters_demanded": np.round(liters, 1)
    })

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df = generate_cold_brew_data()
    df.to_csv(os.path.join(data_dir, "cold_brew_demand.csv"), index=False)
    
    X = df[["forecast_peak_temp_c"]].values
    y = df["cold_brew_liters_demanded"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    joblib.dump(model, os.path.join(models_dir, "linear_regression.pkl"))
    return {
        "algorithm": "Linear Regression",
        "formula": f"Y = {float(model.coef_[0]):.3f} * X + {float(model.intercept_):.3f}",
        "slope": round(float(model.coef_[0]), 4),
        "intercept": round(float(model.intercept_), 4),
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 3),
        "samples": len(df)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Linear Regression trained successfully:", metrics)
