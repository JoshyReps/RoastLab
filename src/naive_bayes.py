"""
Algorithm 2: Multinomial Naïve Bayes Classifier with Laplace Smoothing
Business Task: Single-Origin Whole Bean Recommendation Engine
Predicts the optimal whole-bean coffee origin for home brewing and retail purchase
based on customer brew equipment, roast preference, flavor desires, and drinking style.

Features:
- brew_method: Pour-Over (V60 / Chemex), French Press / Immersion, Espresso Machine / Moka Pot, Cold Drip / Immersion
- roast_preference: Light Roast (Bright & Acidic), Medium Roast (Sweet & Balanced), Dark Roast (Smoky & Bold)
- flavor_notes_desired: Floral & Citrus, Milk Chocolate & Nutty, Dark Cacao & Earthy Spice, Winey Berry & Sweet Fruit
- drinking_style: Drank Black / Neat, With Milk / Cream, Iced / Chilled

Target (Y) - 6 Single-Origin Beans:
- Ethiopian Yirgacheffe
- Colombian Huila Supremo
- Sumatra Mandheling
- Guatemalan Antigua
- Kenyan AA
- Brazilian Cerrado Santos
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

BREW_METHODS = [
    "Pour-Over (V60 / Chemex)",
    "French Press / Immersion",
    "Espresso Machine / Moka Pot",
    "Cold Drip / Immersion"
]

ROAST_PREFERENCES = [
    "Light Roast (Bright & Acidic)",
    "Medium Roast (Sweet & Balanced)",
    "Dark Roast (Smoky & Bold)"
]

FLAVOR_NOTES = [
    "Floral & Citrus",
    "Milk Chocolate & Nutty",
    "Dark Cacao & Earthy Spice",
    "Winey Berry & Sweet Fruit"
]

DRINKING_STYLES = [
    "Drank Black / Neat",
    "With Milk / Cream",
    "Iced / Chilled"
]

BEAN_ORIGINS = [
    "Ethiopian Yirgacheffe",
    "Colombian Huila Supremo",
    "Sumatra Mandheling",
    "Guatemalan Antigua",
    "Kenyan AA",
    "Brazilian Cerrado Santos"
]

def generate_bean_recommendation_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    
    for _ in range(n_samples):
        bm = np.random.choice(BREW_METHODS, p=[0.35, 0.25, 0.25, 0.15])
        rp = np.random.choice(ROAST_PREFERENCES, p=[0.30, 0.45, 0.25])
        fn = np.random.choice(FLAVOR_NOTES, p=[0.25, 0.35, 0.20, 0.20])
        ds = np.random.choice(DRINKING_STYLES, p=[0.50, 0.35, 0.15])
        
        # Log-odds weights across the 6 bean origins:
        # [Ethiopian, Colombian, Sumatra, Guatemalan, Kenyan, Brazilian]
        scores = np.zeros(6)
        
        # Brew method alignment
        if bm == "Pour-Over (V60 / Chemex)":
            scores += np.array([2.5, 0.5, -2.0, 0.8, 2.2, -1.5])
        elif bm == "French Press / Immersion":
            scores += np.array([-1.8, 0.8, 3.0, 1.2, -1.0, 1.5])
        elif bm == "Espresso Machine / Moka Pot":
            scores += np.array([-1.5, 1.5, 1.2, 1.8, -0.8, 2.8])
        elif bm == "Cold Drip / Immersion":
            scores += np.array([1.5, 0.8, 1.0, 0.5, 1.8, 0.5])
            
        # Roast preference alignment
        if rp == "Light Roast (Bright & Acidic)":
            scores += np.array([3.2, -0.5, -3.5, -0.2, 2.8, -2.0])
        elif rp == "Medium Roast (Sweet & Balanced)":
            scores += np.array([0.2, 2.5, 0.0, 2.0, 0.6, 1.8])
        elif rp == "Dark Roast (Smoky & Bold)":
            scores += np.array([-3.0, 0.2, 3.5, 1.0, -2.5, 2.2])
            
        # Flavor notes alignment
        if fn == "Floral & Citrus":
            scores += np.array([3.8, -1.0, -3.2, -0.5, 2.4, -2.5])
        elif fn == "Milk Chocolate & Nutty":
            scores += np.array([-1.5, 3.0, 0.5, 1.8, -1.2, 2.8])
        elif fn == "Dark Cacao & Earthy Spice":
            scores += np.array([-2.5, 0.5, 3.5, 2.0, -1.8, 1.2])
        elif fn == "Winey Berry & Sweet Fruit":
            scores += np.array([1.8, 0.2, -1.6, 0.4, 3.5, -1.0])
            
        # Drinking style alignment
        if ds == "Drank Black / Neat":
            scores += np.array([2.2, 0.8, 0.2, 0.6, 2.0, -0.8])
        elif ds == "With Milk / Cream":
            scores += np.array([-2.4, 1.6, 2.2, 1.4, -1.8, 2.6])
        elif ds == "Iced / Chilled":
            scores += np.array([1.8, 0.2, 0.8, -0.4, 2.2, 0.0])
            
        # Add slight realistic stochastic noise
        scores += np.random.normal(0, 0.5, size=6)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        chosen_bean = np.random.choice(BEAN_ORIGINS, p=probs)
        rows.append({
            "brew_method": bm,
            "roast_preference": rp,
            "flavor_notes_desired": fn,
            "drinking_style": ds,
            "recommended_bean": chosen_bean
        })
        
    return pd.DataFrame(rows)

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df = generate_bean_recommendation_data()
    df.to_csv(os.path.join(data_dir, "drink_preference.csv"), index=False)
    
    X = df[["brew_method", "roast_preference", "flavor_notes_desired", "drinking_style"]]
    y = df["recommended_bean"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("nb", MultinomialNB(alpha=1.0)) # Laplace smoothing alpha=1.0
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(models_dir, "naive_bayes.pkl"))
    return {
        "algorithm": "Multinomial Naive Bayes (Single-Origin Bean Origin Recommender)",
        "accuracy": round(float(acc), 4),
        "samples": len(df),
        "beans": [str(c) for c in pipeline.named_steps["nb"].classes_]
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("Bean Recommender Naive Bayes trained successfully:", metrics)
