"""
Algorithm 5: ID3 Decision Tree Classifier (Entropy / Information Gain)
Business Task: Cafe Branch Location & Expansion Site Selection Optimizer
Recommends the ideal coffee shop format and location archetype based on
environmental, demographic, financial, and spatial site characteristics.
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

LOCATION_DATA = [
    # 1. Downtown CBD Express Kiosk (High commuter density, grab-and-go, prime rent, transit/compact)
    ("Heavy Transit & Office Commuters", "High / Prime Downtown ($$$$)", "Grab-&-Go (< 10 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Micro Space (< 40 sqm)", "Downtown CBD Express Kiosk"),
    ("Heavy Transit & Office Commuters", "High / Prime Downtown ($$$$)", "Grab-&-Go (< 10 mins)", "Street Parking & Shared Lot", "Micro Space (< 40 sqm)", "Downtown CBD Express Kiosk"),
    ("Heavy Transit & Office Commuters", "Moderate / Urban Commercial ($$)", "Grab-&-Go (< 10 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Micro Space (< 40 sqm)", "Downtown CBD Express Kiosk"),
    ("Heavy Transit & Office Commuters", "High / Prime Downtown ($$$$)", "Grab-&-Go (< 10 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Standard Lounge (80 - 150 sqm)", "Downtown CBD Express Kiosk"),
    
    # 2. University Campus Study Lounge (Academic demographic, extended dwell, moderate rent, standard lounge)
    ("Moderate Student & Academic", "Moderate / Urban Commercial ($$)", "Extended Study & Work (> 60 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Standard Lounge (80 - 150 sqm)", "University Campus Study Lounge"),
    ("Moderate Student & Academic", "Moderate / Urban Commercial ($$)", "Extended Study & Work (> 60 mins)", "Street Parking & Shared Lot", "Standard Lounge (80 - 150 sqm)", "University Campus Study Lounge"),
    ("Moderate Student & Academic", "Economical / Low Overhead ($)", "Extended Study & Work (> 60 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Standard Lounge (80 - 150 sqm)", "University Campus Study Lounge"),
    ("Moderate Student & Academic", "Moderate / Urban Commercial ($$)", "Extended Study & Work (> 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "University Campus Study Lounge"),
    ("Moderate Student & Academic", "Economical / Low Overhead ($)", "Extended Study & Work (> 60 mins)", "Street Parking & Shared Lot", "Standard Lounge (80 - 150 sqm)", "University Campus Study Lounge"),
    
    # 3. Suburban Community Roastery & Patio (Residential families, weekend brunch, spacious, parking)
    ("Neighborhood Residents & Families", "Moderate / Urban Commercial ($$)", "Social Dine-In & Brunch (30 - 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "Suburban Community Roastery & Patio"),
    ("Neighborhood Residents & Families", "Economical / Low Overhead ($)", "Social Dine-In & Brunch (30 - 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "Suburban Community Roastery & Patio"),
    ("Neighborhood Residents & Families", "Moderate / Urban Commercial ($$)", "Social Dine-In & Brunch (30 - 60 mins)", "Street Parking & Shared Lot", "Standard Lounge (80 - 150 sqm)", "Suburban Community Roastery & Patio"),
    ("Neighborhood Residents & Families", "Moderate / Urban Commercial ($$)", "Extended Study & Work (> 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "Suburban Community Roastery & Patio"),
    ("Neighborhood Residents & Families", "Economical / Low Overhead ($)", "Social Dine-In & Brunch (30 - 60 mins)", "Dedicated Drive-Thru & Ample Parking", "Spacious Destination (> 200 sqm)", "Suburban Community Roastery & Patio"),
    
    # 4. Highway Commuter Drive-Thru Express (High vehicular flow, rapid window service, dedicated drive-thru lane)
    ("Low Pedestrian / High Vehicular Flow", "Economical / Low Overhead ($)", "Grab-&-Go (< 10 mins)", "Dedicated Drive-Thru & Ample Parking", "Standard Lounge (80 - 150 sqm)", "Highway Commuter Drive-Thru Express"),
    ("Low Pedestrian / High Vehicular Flow", "Moderate / Urban Commercial ($$)", "Grab-&-Go (< 10 mins)", "Dedicated Drive-Thru & Ample Parking", "Standard Lounge (80 - 150 sqm)", "Highway Commuter Drive-Thru Express"),
    ("Low Pedestrian / High Vehicular Flow", "Economical / Low Overhead ($)", "Grab-&-Go (< 10 mins)", "Dedicated Drive-Thru & Ample Parking", "Micro Space (< 40 sqm)", "Highway Commuter Drive-Thru Express"),
    ("Low Pedestrian / High Vehicular Flow", "Economical / Low Overhead ($)", "Social Dine-In & Brunch (30 - 60 mins)", "Dedicated Drive-Thru & Ample Parking", "Standard Lounge (80 - 150 sqm)", "Highway Commuter Drive-Thru Express"),
    
    # 5. Arts & Heritage Flagship Tasting Room (Tourists & cultural leisure, tasting bar, high ticket, spacious)
    ("Weekend Leisure & Tourists", "High / Prime Downtown ($$$$)", "Social Dine-In & Brunch (30 - 60 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Spacious Destination (> 200 sqm)", "Arts & Heritage Flagship Tasting Room"),
    ("Weekend Leisure & Tourists", "Moderate / Urban Commercial ($$)", "Social Dine-In & Brunch (30 - 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "Arts & Heritage Flagship Tasting Room"),
    ("Weekend Leisure & Tourists", "High / Prime Downtown ($$$$)", "Extended Study & Work (> 60 mins)", "Street Parking & Shared Lot", "Spacious Destination (> 200 sqm)", "Arts & Heritage Flagship Tasting Room"),
    ("Weekend Leisure & Tourists", "Moderate / Urban Commercial ($$)", "Extended Study & Work (> 60 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Spacious Destination (> 200 sqm)", "Arts & Heritage Flagship Tasting Room"),
    
    # 6. Corporate Business Park Micro-Lobby (Captive office workers, low overhead, M-F weekday rush)
    ("Heavy Transit & Office Commuters", "Economical / Low Overhead ($)", "Grab-&-Go (< 10 mins)", "Street Parking & Shared Lot", "Micro Space (< 40 sqm)", "Corporate Business Park Micro-Lobby"),
    ("Heavy Transit & Office Commuters", "Economical / Low Overhead ($)", "Grab-&-Go (< 10 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Micro Space (< 40 sqm)", "Corporate Business Park Micro-Lobby"),
    ("Moderate Student & Academic", "Economical / Low Overhead ($)", "Grab-&-Go (< 10 mins)", "Transit Hub / Walk-Up Only (Zero Parking)", "Micro Space (< 40 sqm)", "Corporate Business Park Micro-Lobby"),
    ("Heavy Transit & Office Commuters", "Economical / Low Overhead ($)", "Moderate Stay (15 - 45 mins)", "Street Parking & Shared Lot", "Micro Space (< 40 sqm)", "Corporate Business Park Micro-Lobby")
]

FEATURE_COLS = ["foot_traffic", "rent_budget", "dwell_time", "parking_access", "space_capacity"]
TARGET_COL = "store_format"

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_locations = pd.DataFrame(LOCATION_DATA, columns=FEATURE_COLS + [TARGET_COL])
    df_locations.to_csv(os.path.join(data_dir, "store_location_expansion.csv"), index=False)
    
    X = df_locations[FEATURE_COLS]
    y = df_locations[TARGET_COL]
    
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
        "algorithm": "ID3 Decision Tree (Information Gain / Entropy Criterion)",
        "accuracy": round(float(acc), 4),
        "tree_depth": int(dt_model.get_depth()),
        "leaves_count": int(dt_model.get_n_leaves()),
        "samples": len(df_locations),
        "classes": [str(c) for c in dt_model.classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Decision Tree Location Model trained successfully:", metrics)
