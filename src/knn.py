"""
Algorithm 3: Multi-Dimensional K-Nearest Neighbors (KNN) Recommender
Features: 5 Continuous Sensory Attributes (Normalized Scale 1-10)
- Sweetness (1 - 10)
- Caffeine Strength (1 - 10)
- Bitterness / Roast Intensity (1 - 10)
- Creaminess / Dairy Body (1 - 10)
- Acidity / Citrus Brightness (1 - 10)

Feature Normalization:
x_norm = (x - x_min) / (x_max - x_min)
Distance: Euclidean metric d(P, Q) = sqrt(sum (p_i - q_i)^2)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

MENU_ITEMS_5D = [
    {
        "drink_name": "Signature Cold Brew",
        "category": "Cold Coffee",
        "sweetness": 1.5,
        "caffeine": 9.2,
        "bitterness": 6.5,
        "creaminess": 1.0,
        "acidity": 5.5,
        "price": "$5.50",
        "desc": "Steeped 20 hours in cold water. Clean, bold cacao finish with crisp clarity and zero dairy."
    },
    {
        "drink_name": "Nitro Draft Velvet",
        "category": "Cold Coffee",
        "sweetness": 2.0,
        "caffeine": 9.5,
        "bitterness": 5.8,
        "creaminess": 4.5,
        "acidity": 4.0,
        "price": "$6.25",
        "desc": "Draft tap infused with pure nitrogen. Micro-foam cascade creating a natural creamy texture."
    },
    {
        "drink_name": "Double Ristretto Espresso",
        "category": "Hot Espresso",
        "sweetness": 1.0,
        "caffeine": 9.8,
        "bitterness": 8.8,
        "creaminess": 1.5,
        "acidity": 7.2,
        "price": "$4.25",
        "desc": "Short, restricted extraction capturing dense caramel crema and intense essential oils."
    },
    {
        "drink_name": "Artisanal Americano",
        "category": "Hot Espresso",
        "sweetness": 1.5,
        "caffeine": 8.2,
        "bitterness": 6.8,
        "creaminess": 1.0,
        "acidity": 6.0,
        "price": "$4.50",
        "desc": "Double espresso pulled over hot mountain spring water. Balanced, aromatic, and light-bodied."
    },
    {
        "drink_name": "Single-Origin Pour-Over (Ethiopian)",
        "category": "Pour-Over",
        "sweetness": 3.0,
        "caffeine": 8.0,
        "bitterness": 3.2,
        "creaminess": 1.0,
        "acidity": 9.2,
        "price": "$6.50",
        "desc": "Hand-poured Chemex. Vivid floral jasmine, bergamot brightness, and sweet Meyer lemon notes."
    },
    {
        "drink_name": "Sumatra Dark Roast French Press",
        "category": "Pour-Over",
        "sweetness": 1.0,
        "caffeine": 8.5,
        "bitterness": 9.2,
        "creaminess": 1.8,
        "acidity": 2.0,
        "price": "$5.75",
        "desc": "Heavy body, intense dark roast depth, earthy cedar, pipe tobacco, and dark bittersweet cocoa."
    },
    {
        "drink_name": "Artisanal Flat White",
        "category": "Milk Coffee",
        "sweetness": 3.5,
        "caffeine": 7.5,
        "bitterness": 5.2,
        "creaminess": 7.5,
        "acidity": 4.0,
        "price": "$5.25",
        "desc": "Double ristretto enveloped in free-poured, velvety steamed microfoam milk with silky mouthfeel."
    },
    {
        "drink_name": "Spanish Cortado",
        "category": "Milk Coffee",
        "sweetness": 3.0,
        "caffeine": 8.2,
        "bitterness": 6.5,
        "creaminess": 5.5,
        "acidity": 5.0,
        "price": "$4.75",
        "desc": "1:1 ratio of rich espresso to warm steamed milk. Cuts bitterness while maintaining punch."
    },
    {
        "drink_name": "Caffe Velvet Latte",
        "category": "Milk Coffee",
        "sweetness": 4.5,
        "caffeine": 6.5,
        "bitterness": 4.0,
        "creaminess": 8.5,
        "acidity": 3.2,
        "price": "$5.50",
        "desc": "Smooth steamed whole milk over espresso with a delicate, creamy froth top."
    },
    {
        "drink_name": "Brown Sugar Oat Latte",
        "category": "Plant-Based",
        "sweetness": 6.8,
        "caffeine": 5.8,
        "bitterness": 4.2,
        "creaminess": 7.8,
        "acidity": 2.8,
        "price": "$6.75",
        "desc": "Rolled oat milk hand-shaken with espresso, molasses brown sugar reduction, and cinnamon."
    },
    {
        "drink_name": "Caramel Macchiato Reserve",
        "category": "Sweet Coffee",
        "sweetness": 8.5,
        "caffeine": 6.0,
        "bitterness": 3.8,
        "creaminess": 8.2,
        "acidity": 2.5,
        "price": "$6.25",
        "desc": "Vanilla-infused steamed milk marked with fresh espresso shots and artisanal butter caramel drizzle."
    },
    {
        "drink_name": "Valrhona Mocha Truffle",
        "category": "Sweet Coffee",
        "sweetness": 7.5,
        "caffeine": 7.0,
        "bitterness": 5.8,
        "creaminess": 8.5,
        "acidity": 2.0,
        "price": "$6.50",
        "desc": "Melted French dark cacao ganache folded into espresso and warm velvety textured milk."
    },
    {
        "drink_name": "Madagascar Vanilla Bean Frappé",
        "category": "Blended",
        "sweetness": 9.2,
        "caffeine": 3.5,
        "bitterness": 1.8,
        "creaminess": 9.0,
        "acidity": 1.2,
        "price": "$6.75",
        "desc": "Blended crystal ice, fresh dairy cream, real Madagascar vanilla seeds, and light cold-brew syrup."
    },
    {
        "drink_name": "Ceremonial Matcha Espresso Fusion",
        "category": "Specialty",
        "sweetness": 5.2,
        "caffeine": 6.2,
        "bitterness": 5.5,
        "creaminess": 6.8,
        "acidity": 2.8,
        "price": "$6.50",
        "desc": "Stone-ground Uji matcha green tea layered over iced oat milk and a floating single-origin espresso shot."
    }
]

FEATURE_COLS = ["sweetness", "caffeine", "bitterness", "creaminess", "acidity"]

def generate_5d_taste_profiles(n_per_item: int = 30, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_menu = pd.DataFrame(MENU_ITEMS_5D)
    np.random.seed(seed)
    rows = []
    
    for item in MENU_ITEMS_5D:
        for _ in range(n_per_item):
            s = np.clip(item["sweetness"] + np.random.normal(0, 0.4), 1.0, 10.0)
            c = np.clip(item["caffeine"] + np.random.normal(0, 0.4), 1.0, 10.0)
            b = np.clip(item["bitterness"] + np.random.normal(0, 0.4), 1.0, 10.0)
            cr = np.clip(item["creaminess"] + np.random.normal(0, 0.4), 1.0, 10.0)
            a = np.clip(item["acidity"] + np.random.normal(0, 0.4), 1.0, 10.0)
            
            rows.append({
                "sweetness_pref": round(float(s), 2),
                "caffeine_pref": round(float(c), 2),
                "bitterness_pref": round(float(b), 2),
                "creaminess_pref": round(float(cr), 2),
                "acidity_pref": round(float(a), 2),
                "recommended_drink": item["drink_name"]
            })
            
    return df_menu, pd.DataFrame(rows)

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_menu, df_train = generate_5d_taste_profiles()
    df_menu.to_csv(os.path.join(data_dir, "drinks_menu.csv"), index=False)
    df_train.to_csv(os.path.join(data_dir, "customer_taste_profiles.csv"), index=False)
    
    X = df_train[["sweetness_pref", "caffeine_pref", "bitterness_pref", "creaminess_pref", "acidity_pref"]]
    y = df_train["recommended_drink"]
    
    # Min-Max Scaling Pipeline to normalize features to [0, 1]
    pipeline = Pipeline([
        ("scaler", MinMaxScaler(feature_range=(0, 1))),
        ("knn", KNeighborsClassifier(n_neighbors=3, metric="euclidean"))
    ])
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, os.path.join(models_dir, "knn.pkl"))
    return {
        "algorithm": "K-Nearest Neighbors (5-Dimensional with Min-Max Scaling)",
        "features": FEATURE_COLS,
        "menu_items_count": len(df_menu),
        "samples": len(df_train)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("5D Normalized KNN trained successfully:", metrics)
