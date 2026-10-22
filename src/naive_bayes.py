"""
Algorithm 2: Multi-Class Multinomial Naïve Bayes Classifier with Laplace Smoothing
Business Task: Granular Beverage Category Predictor based on Contextual Operational Conditions

Features:
- day_type: Weekday, Weekend / Holiday
- weather_condition: Hot & Humid, Cool & Dry, Rainy, Freezing
- customer_segment: Student, Morning Commuter, Remote Worker, Evening Social
- companion_purchase: Food Item Ordered, Beverage Only

Target (Y):
- Hot Espresso
- Iced Specialty Milk
- Blended Frappé
- Cold Brew / Iced Tea
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

DAY_TYPES = ["Weekday", "Weekend / Holiday"]
WEATHER_CONDITIONS = ["Hot & Humid", "Cool & Dry", "Rainy", "Freezing"]
CUSTOMER_SEGMENTS = ["Morning Commuter", "Student", "Remote Worker", "Evening Social"]
COMPANION_PURCHASES = ["Food Item Ordered", "Beverage Only"]
CATEGORIES = ["Hot Espresso", "Iced Specialty Milk", "Blended Frappé", "Cold Brew / Iced Tea"]

def generate_enhanced_preference_data(n_samples: int = 480, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    
    for _ in range(n_samples):
        dt = np.random.choice(DAY_TYPES, p=[0.70, 0.30])
        w = np.random.choice(WEATHER_CONDITIONS, p=[0.35, 0.30, 0.25, 0.10])
        cs = np.random.choice(CUSTOMER_SEGMENTS, p=[0.40, 0.25, 0.20, 0.15])
        cp = np.random.choice(COMPANION_PURCHASES, p=[0.45, 0.55])
        
        # Base log-odds / weights for categories:
        # [Hot Espresso, Iced Specialty Milk, Blended Frappé, Cold Brew / Iced Tea]
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Weather influence
        if w == "Hot & Humid":
            scores += np.array([-0.6, 0.5, 0.8, 0.9])
        elif w == "Freezing":
            scores += np.array([1.2, -0.4, -0.9, -0.8])
        elif w == "Rainy":
            scores += np.array([0.9, 0.3, -0.8, -0.5])
        elif w == "Cool & Dry":
            scores += np.array([0.5, 0.4, -0.3, 0.1])
            
        # Day Type influence
        if dt == "Weekday":
            scores += np.array([0.5, 0.2, -0.3, 0.3])
        else:
            scores += np.array([-0.2, 0.4, 0.7, 0.1])
            
        # Segment influence
        if cs == "Morning Commuter":
            scores += np.array([0.9, 0.1, -0.6, 0.4])
        elif cs == "Student":
            scores += np.array([-0.4, 0.6, 0.8, 0.2])
        elif cs == "Remote Worker":
            scores += np.array([0.3, 0.5, -0.2, 0.4])
        elif cs == "Evening Social":
            scores += np.array([-0.1, 0.5, 0.5, 0.1])
            
        # Companion Purchase influence (warm pastry/food strongly pairs with warm drinks)
        if cp == "Food Item Ordered":
            scores += np.array([0.8, 0.3, -0.6, -0.4])
        else:
            scores += np.array([-0.2, 0.1, 0.4, 0.5])
            
        # Softmax into probability distribution
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        chosen_cat = np.random.choice(CATEGORIES, p=probs)
        rows.append({
            "day_type": dt,
            "weather_condition": w,
            "customer_segment": cs,
            "companion_purchase": cp,
            "beverage_category": chosen_cat
        })
        
    return pd.DataFrame(rows)

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df = generate_enhanced_preference_data()
    df.to_csv(os.path.join(data_dir, "drink_preference.csv"), index=False)
    
    X = df[["day_type", "weather_condition", "customer_segment", "companion_purchase"]]
    y = df["beverage_category"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("nb", MultinomialNB(alpha=1.0)) # Laplace smoothing alpha=1.0
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(models_dir, "naive_bayes.pkl"))
    return {
        "algorithm": "Multinomial Naive Bayes (Multi-Class with Laplace alpha=1.0)",
        "accuracy": round(float(acc), 4),
        "samples": len(df),
        "categories": [str(c) for c in pipeline.named_steps["nb"].classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Enhanced Naïve Bayes trained successfully:", metrics)
