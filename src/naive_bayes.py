"""
Algorithm 2: Multinomial Naïve Bayes Classifier with Laplace Smoothing
Business Task: Hot vs. Iced Order Preference Predictor
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def generate_preference_data(n_samples: int = 320, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    time_of_day_opts = ["Morning", "Afternoon", "Evening"]
    order_type_opts = ["Dine-in", "Takeaway"]
    weather_opts = ["Warm", "Cool"]
    
    rows = []
    for _ in range(n_samples):
        tod = np.random.choice(time_of_day_opts, p=[0.5, 0.35, 0.15])
        ot = np.random.choice(order_type_opts, p=[0.4, 0.6])
        w = np.random.choice(weather_opts, p=[0.65, 0.35])
        
        p_iced = 0.5
        if tod == "Morning":
            p_iced -= 0.25
        elif tod == "Afternoon":
            p_iced += 0.30
        
        if w == "Warm":
            p_iced += 0.30
        else:
            p_iced -= 0.35
            
        if ot == "Takeaway":
            p_iced += 0.10
            
        p_iced = np.clip(p_iced, 0.05, 0.95)
        label = "Iced" if np.random.rand() < p_iced else "Hot"
        rows.append({"time_of_day": tod, "order_type": ot, "weather": w, "preference": label})
        
    return pd.DataFrame(rows)

def train_and_save(data_dir: str, models_dir: str) -> dict:
    
    df = generate_preference_data()
    df.to_csv(os.path.join(data_dir, "drink_preference.csv"), index=False)
    
    X = df[["time_of_day", "order_type", "weather"]]
    y = df["preference"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("nb", MultinomialNB(alpha=1.0)) # Laplace smoothing alpha=1.0
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(models_dir, "naive_bayes.pkl"))
    return {
        "algorithm": "Multinomial Naive Bayes (Laplace alpha=1.0)",
        "accuracy": round(float(acc), 4),
        "samples": len(df),
        "classes": [str(c) for c in pipeline.named_steps["nb"].classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Naïve Bayes trained successfully:", metrics)
