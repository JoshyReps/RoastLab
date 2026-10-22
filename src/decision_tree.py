"""
Algorithm 5: ID3 Decision Tree Classifier (Entropy / Information Gain)
Business Task: Single-Origin Coffee Bean Recommendation Quiz
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

QUIZ_DATA = [
    # Ethiopian Yirgacheffe: Light, High Acidity, Fruity/Floral
    ("Light/Tea-like", "High/Bright", "Fruity/Floral", "Ethiopian Yirgacheffe"),
    ("Light/Tea-like", "Balanced", "Fruity/Floral", "Ethiopian Yirgacheffe"),
    ("Medium/Smooth", "High/Bright", "Fruity/Floral", "Ethiopian Yirgacheffe"),
    
    # Kenyan AA: Medium, High Acidity, Earthy/Spicy
    ("Medium/Smooth", "High/Bright", "Earthy/Spicy", "Kenyan AA"),
    ("Light/Tea-like", "High/Bright", "Earthy/Spicy", "Kenyan AA"),
    
    # Colombian Supremo: Medium, Balanced, Chocolatey/Nutty
    ("Medium/Smooth", "Balanced", "Chocolatey/Nutty", "Colombian Supremo"),
    ("Medium/Smooth", "Low/Mild", "Chocolatey/Nutty", "Colombian Supremo"),
    ("Light/Tea-like", "Balanced", "Chocolatey/Nutty", "Colombian Supremo"),
    
    # Guatemalan Antigua: Medium, High, Chocolatey/Nutty
    ("Medium/Smooth", "High/Bright", "Chocolatey/Nutty", "Guatemalan Antigua"),
    ("Heavy/Syrupy", "High/Bright", "Chocolatey/Nutty", "Guatemalan Antigua"),
    
    # Sumatra Mandheling: Heavy, Low Acidity, Earthy/Spicy
    ("Heavy/Syrupy", "Low/Mild", "Earthy/Spicy", "Sumatra Mandheling"),
    ("Heavy/Syrupy", "Balanced", "Earthy/Spicy", "Sumatra Mandheling"),
    ("Medium/Smooth", "Low/Mild", "Earthy/Spicy", "Sumatra Mandheling"),
    
    # Brazilian Santos: Heavy, Low Acidity, Chocolatey/Nutty
    ("Heavy/Syrupy", "Low/Mild", "Chocolatey/Nutty", "Brazilian Santos"),
    ("Heavy/Syrupy", "Balanced", "Chocolatey/Nutty", "Brazilian Santos"),
]

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_beans = pd.DataFrame(QUIZ_DATA, columns=["body", "acidity", "flavor_profile", "recommended_bean"])
    df_beans.to_csv(os.path.join(data_dir, "coffee_beans_quiz.csv"), index=False)
    
    X = df_beans[["body", "acidity", "flavor_profile"]]
    y = df_beans["recommended_bean"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("dt", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42))
    ])
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(models_dir, "decision_tree.pkl"))
    return {
        "algorithm": "ID3 Decision Tree (Information Gain / Entropy Criterion)",
        "accuracy": round(float(acc), 4),
        "tree_depth": int(pipeline.named_steps["dt"].get_depth()),
        "leaves_count": int(pipeline.named_steps["dt"].get_n_leaves()),
        "samples": len(df_beans)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Decision Tree trained successfully:", metrics)
