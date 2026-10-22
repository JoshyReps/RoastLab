"""
Algorithm 6: Artificial Neural Network (Multi-Layer Perceptron Regressor)
Business Task: Roaster Batch Telemetry to SCA Cupping Score Quality Predictor
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score

def generate_roast_telemetry_data(n_samples: int = 350, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    charge_temp = np.random.uniform(190.0, 215.0, n_samples)
    drop_temp = np.random.uniform(205.0, 228.0, n_samples)
    roast_time = np.random.uniform(9.0, 14.5, n_samples)
    crack_time = np.clip(roast_time * np.random.uniform(0.70, 0.82, n_samples), 6.5, 12.0)
    ror_avg = np.random.uniform(7.0, 14.0, n_samples)
    
    # Cupping Score (80 - 95 SCA scale)
    score = (
        89.0
        - 1.5 * (roast_time - 11.5) ** 2
        - 0.08 * (drop_temp - 214.0) ** 2
        - 0.45 * (ror_avg - 9.5) ** 2
        + 0.05 * (charge_temp - 200.0)
        + np.random.normal(0, 0.35, n_samples)
    )
    score = np.clip(score, 78.0, 94.5)
    
    tiers = []
    for s in score:
        if s >= 88.0:
            tiers.append("Presidential Specialty")
        elif s >= 84.0:
            tiers.append("Specialty Grade")
        elif s >= 80.0:
            tiers.append("Premium Commercial")
        else:
            tiers.append("Underdeveloped / Reprocess")
            
    return pd.DataFrame({
        "charge_temp_c": np.round(charge_temp, 1),
        "drop_temp_c": np.round(drop_temp, 1),
        "roast_time_min": np.round(roast_time, 2),
        "first_crack_time_min": np.round(crack_time, 2),
        "avg_ror_c_per_min": np.round(ror_avg, 2),
        "sca_cupping_score": np.round(score, 2),
        "quality_tier": tiers
    })

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_roast = generate_roast_telemetry_data()
    df_roast.to_csv(os.path.join(data_dir, "roast_quality_telemetry.csv"), index=False)
    
    X = df_roast[["charge_temp_c", "drop_temp_c", "roast_time_min", "first_crack_time_min", "avg_ror_c_per_min"]]
    y = df_roast["sca_cupping_score"].values
    
    reg = TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=2000,
            random_state=42
        ),
        transformer=StandardScaler()
    )
    
    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp_reg", reg)
    ])
    mlp_pipeline.fit(X, y)
    
    preds = mlp_pipeline.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    joblib.dump(mlp_pipeline, os.path.join(models_dir, "ann_roast_qc.pkl"))
    return {
        "algorithm": "Artificial Neural Network (Multi-Layer Perceptron Regressor)",
        "architecture": "Input [5] -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear Output)",
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 3),
        "iterations": int(reg.regressor_.n_iter_),
        "samples": len(df_roast)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("ANN trained successfully:", metrics)
