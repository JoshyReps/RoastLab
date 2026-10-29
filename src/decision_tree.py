"""
Algorithm 5: ID3 Decision Tree Classifier (Entropy / Information Gain)
Business Task: Retail Site Acquisition & Expansion Feasibility Classifier (YES / NO)
Evaluates whether RoastLab should open a new coffee shop at a candidate location
based on 4 environmental, demographic, financial, and infrastructural site criteria:
- foot_traffic: High Volume, Moderate Volume, Low Volume
- lease_overhead: Favorable (< 10%), Moderate Market Rate (10-18%), Prime Overhead (> 18%)
- competition_density: Low Saturation, Moderate Saturation, Heavy Saturation
- transit_access: Prime Accessibility, Standard Accessibility, Restricted

Target Output:
- expansion_verdict: YES / NO
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

SITE_RECORDS = [
    # High Foot Traffic Sites
    ("High Volume (> 15k pedestrians/day)", "Favorable (< 10% revenue)", "Low Saturation (0-2 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Favorable (< 10% revenue)", "Moderate Saturation (3-5 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Favorable (< 10% revenue)", "Heavy Saturation (> 5 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Low Saturation (0-2 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Moderate Saturation (3-5 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Heavy Saturation (> 5 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Prime / Premium Overhead (> 18%)", "Low Saturation (0-2 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("High Volume (> 15k pedestrians/day)", "Prime / Premium Overhead (> 18%)", "Heavy Saturation (> 5 cafes)", "Prime Accessibility (Transit Hub)", "NO"),
    ("High Volume (> 15k pedestrians/day)", "Favorable (< 10% revenue)", "Low Saturation (0-2 cafes)", "Restricted (Poor Walkability)", "NO"),

    # Moderate Foot Traffic Sites
    ("Moderate Volume (5k-15k pedestrians/day)", "Favorable (< 10% revenue)", "Low Saturation (0-2 cafes)", "Prime Accessibility (Transit Hub)", "YES"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Favorable (< 10% revenue)", "Moderate Saturation (3-5 cafes)", "Standard Accessibility (Streetflow)", "YES"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Favorable (< 10% revenue)", "Heavy Saturation (> 5 cafes)", "Standard Accessibility (Streetflow)", "NO"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Low Saturation (0-2 cafes)", "Standard Accessibility (Streetflow)", "YES"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Moderate Saturation (3-5 cafes)", "Standard Accessibility (Streetflow)", "YES"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Heavy Saturation (> 5 cafes)", "Standard Accessibility (Streetflow)", "NO"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Prime / Premium Overhead (> 18%)", "Low Saturation (0-2 cafes)", "Standard Accessibility (Streetflow)", "NO"),
    ("Moderate Volume (5k-15k pedestrians/day)", "Moderate Market Rate (10-18%)", "Low Saturation (0-2 cafes)", "Restricted (Poor Walkability)", "NO"),

    # Low Foot Traffic Sites
    ("Low Volume (< 5k pedestrians/day)", "Favorable (< 10% revenue)", "Low Saturation (0-2 cafes)", "Standard Accessibility (Streetflow)", "NO"),
    ("Low Volume (< 5k pedestrians/day)", "Favorable (< 10% revenue)", "Heavy Saturation (> 5 cafes)", "Restricted (Poor Walkability)", "NO"),
    ("Low Volume (< 5k pedestrians/day)", "Moderate Market Rate (10-18%)", "Low Saturation (0-2 cafes)", "Standard Accessibility (Streetflow)", "NO"),
    ("Low Volume (< 5k pedestrians/day)", "Prime / Premium Overhead (> 18%)", "Low Saturation (0-2 cafes)", "Restricted (Poor Walkability)", "NO")
]

FEATURE_COLS = ["foot_traffic", "lease_overhead", "competition_density", "transit_access"]
TARGET_COL = "expansion_verdict"

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_sites = pd.DataFrame(SITE_RECORDS, columns=FEATURE_COLS + [TARGET_COL])
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
        "task": "Retail Site Acquisition Feasibility (YES / NO)",
        "accuracy": round(float(acc), 4),
        "tree_depth": int(dt_model.get_depth()),
        "leaves_count": int(dt_model.get_n_leaves()),
        "samples": len(df_sites),
        "classes": [str(c) for c in dt_model.classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Professional Binary Decision Tree trained successfully:", metrics)
