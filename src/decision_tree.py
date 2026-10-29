"""
Algorithm 5: ID3 Decision Tree Classifier (Entropy / Information Gain)
Business Task: Coffee Shop Location Feasibility Evaluator (Yes / No)
Predicts whether to open a new coffee shop at a target location
based on 4 environmental site criteria:
- foot_traffic: High, Moderate, Low
- rent_cost: Affordable, Moderate, Expensive
- competition: Low, High
- accessibility: Good, Poor

Target Output:
- open_coffee_shop: Yes / No
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 17 Environmental Site Evaluation Observations
SITE_DATA = [
    ("High", "Affordable", "Low", "Good", "Yes"),
    ("High", "Affordable", "High", "Good", "Yes"),
    ("High", "Moderate", "Low", "Good", "Yes"),
    ("High", "Moderate", "High", "Good", "Yes"),
    ("High", "Expensive", "Low", "Good", "Yes"),
    ("High", "Expensive", "High", "Good", "No"),
    ("High", "Affordable", "Low", "Poor", "No"),
    ("Moderate", "Affordable", "Low", "Good", "Yes"),
    ("Moderate", "Affordable", "High", "Good", "No"),
    ("Moderate", "Moderate", "Low", "Good", "Yes"),
    ("Moderate", "Moderate", "High", "Good", "No"),
    ("Moderate", "Expensive", "Low", "Good", "No"),
    ("Moderate", "Affordable", "Low", "Poor", "No"),
    ("Low", "Affordable", "Low", "Good", "No"),
    ("Low", "Affordable", "High", "Poor", "No"),
    ("Low", "Moderate", "Low", "Good", "No"),
    ("Low", "Expensive", "Low", "Poor", "No")
]

FEATURE_COLS = ["foot_traffic", "rent_cost", "competition", "accessibility"]
TARGET_COL = "open_coffee_shop"

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_sites = pd.DataFrame(SITE_DATA, columns=FEATURE_COLS + [TARGET_COL])
    df_sites.to_csv(os.path.join(data_dir, "coffee_shop_location_decision.csv"), index=False)
    
    X = df_sites[FEATURE_COLS]
    y = df_sites[TARGET_COL]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("dt", DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42))
    ])
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    dt_model = pipeline.named_steps["dt"]
    
    joblib.dump(pipeline, os.path.join(models_dir, "decision_tree.pkl"))
    return {
        "algorithm": "ID3 Decision Tree (Entropy / Information Gain)",
        "task": "Coffee Shop Location Feasibility (Yes / No)",
        "accuracy": round(float(acc), 4),
        "tree_depth": int(dt_model.get_depth()),
        "leaves_count": int(dt_model.get_n_leaves()),
        "samples": len(df_sites),
        "classes": [str(c) for c in dt_model.classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Simple Binary Decision Tree trained successfully:", metrics)
