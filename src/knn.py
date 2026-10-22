"""
Algorithm 3: K-Nearest Neighbors (KNN) Recommender
Business Task: Personalized Drink Menu Recommendation via 2D Euclidean Taste Coordinates
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

MENU_ITEMS = [
    {"drink_name": "Signature Cold Brew", "sweetness": 1.5, "caffeine": 9.2, "category": "Cold Coffee", "desc": "Steeped for 20h, bold, chocolate notes, zero sugar"},
    {"drink_name": "Nitro Draft Coffee", "sweetness": 2.0, "caffeine": 9.5, "category": "Cold Coffee", "desc": "Velvety head, creamy texture without dairy"},
    {"drink_name": "Double Espresso", "sweetness": 1.0, "caffeine": 9.8, "category": "Hot Espresso", "desc": "Rich crema, intense caramelized body"},
    {"drink_name": "Artisanal Americano", "sweetness": 1.8, "caffeine": 8.0, "category": "Hot Espresso", "desc": "Hot water pulled over double espresso shots"},
    {"drink_name": "Flat White", "sweetness": 3.5, "caffeine": 7.5, "category": "Milk Coffee", "desc": "Microfoam milk with double ristretto"},
    {"drink_name": "Cortado", "sweetness": 2.8, "caffeine": 8.2, "category": "Milk Coffee", "desc": "Equal parts espresso and warm steamed milk"},
    {"drink_name": "Caffe Latte", "sweetness": 4.5, "caffeine": 6.5, "category": "Milk Coffee", "desc": "Smooth steamed milk with rich espresso base"},
    {"drink_name": "Spanish Latte", "sweetness": 7.8, "caffeine": 6.8, "category": "Sweet Coffee", "desc": "Espresso infused with sweetened condensed milk"},
    {"drink_name": "Caramel Macchiato", "sweetness": 8.5, "caffeine": 6.0, "category": "Sweet Coffee", "desc": "Vanilla-infused milk marked with espresso & caramel drizzle"},
    {"drink_name": "Mocha Truffle", "sweetness": 7.5, "caffeine": 7.0, "category": "Sweet Coffee", "desc": "Artisan dark cacao melted with espresso and whole milk"},
    {"drink_name": "Vanilla Bean Frappe", "sweetness": 9.2, "caffeine": 4.0, "category": "Blended", "desc": "Blended ice, real Madagascar vanilla, whipped cream"},
    {"drink_name": "Brown Sugar Oat Latte", "sweetness": 6.8, "caffeine": 5.8, "category": "Plant-based", "desc": "Rolled oat milk with spiced brown sugar reduction"},
    {"drink_name": "Matcha Espresso Fusion", "sweetness": 5.5, "caffeine": 6.2, "category": "Specialty", "desc": "Ceremonial grade Uji matcha layered with single-origin espresso"},
    {"drink_name": "Decaf Honey Cinnamon Cappuccino", "sweetness": 5.0, "caffeine": 1.2, "category": "Low Caffeine", "desc": "Swiss water decaf with wild honey and Ceylon cinnamon"}
]

def generate_taste_profiles(n_per_item: int = 25, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_menu = pd.DataFrame(MENU_ITEMS)
    np.random.seed(seed)
    rows = []
    for item in MENU_ITEMS:
        for _ in range(n_per_item):
            s = np.clip(item["sweetness"] + np.random.normal(0, 0.4), 1.0, 10.0)
            c = np.clip(item["caffeine"] + np.random.normal(0, 0.4), 1.0, 10.0)
            rows.append({
                "sweetness_pref": round(float(s), 2),
                "caffeine_pref": round(float(c), 2),
                "recommended_drink": item["drink_name"]
            })
    return df_menu, pd.DataFrame(rows)

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_menu, df_train = generate_taste_profiles()
    df_menu.to_csv(os.path.join(data_dir, "drinks_menu.csv"), index=False)
    df_train.to_csv(os.path.join(data_dir, "customer_taste_profiles.csv"), index=False)
    
    X = df_train[["sweetness_pref", "caffeine_pref"]]
    y = df_train["recommended_drink"]
    
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(X, y)
    
    joblib.dump(knn, os.path.join(models_dir, "knn.pkl"))
    return {
        "algorithm": "K-Nearest Neighbors (k=3, Euclidean metric)",
        "menu_items_count": len(df_menu),
        "samples": len(df_train)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("KNN trained successfully:", metrics)
