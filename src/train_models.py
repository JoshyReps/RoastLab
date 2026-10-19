"""
RoastLab Coffee Co. - Core Machine Learning Training Pipeline
Generates representative domain datasets and trains all 6 required models for ELEC026.23.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score
)

# Output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

metrics_report = {}

def convert_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ==============================================================================
# 1. LINEAR REGRESSION: Cold Brew Batch Volume Demand Predictor
# ==============================================================================
def train_linear_regression():
    print("[1/6] Training Linear Regression (Cold Brew Demand)...")
    np.random.seed(42)
    n_samples = 180
    
    # Temperature between 18°C and 38°C
    temps = np.random.uniform(18.0, 38.0, size=(n_samples, 1))
    # Batch volume: roughly 0.95 * temp - 4.5 + noise
    liters = 0.95 * temps[:, 0] - 4.5 + np.random.normal(0, 1.8, size=n_samples)
    liters = np.clip(liters, 8.0, 45.0)
    
    df = pd.DataFrame({
        "forecast_peak_temp_c": np.round(temps[:, 0], 1),
        "cold_brew_liters_demanded": np.round(liters, 1)
    })
    df.to_csv(os.path.join(DATA_DIR, "cold_brew_demand.csv"), index=False)
    
    model = LinearRegression()
    model.fit(temps, liters)
    
    preds = model.predict(temps)
    r2 = r2_score(liters, preds)
    rmse = np.sqrt(mean_squared_error(liters, preds))
    
    joblib.dump(model, os.path.join(MODELS_DIR, "linear_regression.pkl"))
    metrics_report["linear_regression"] = {
        "algorithm": "Linear Regression",
        "formula": f"Y = {float(model.coef_[0]):.3f} * X + {float(model.intercept_):.3f}",
        "slope": round(float(model.coef_[0]), 4),
        "intercept": round(float(model.intercept_), 4),
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 3),
        "samples": int(n_samples)
    }
    print(f"      R2: {r2:.3f}, RMSE: {rmse:.2f}L, slope: {model.coef_[0]:.3f}")

# ==============================================================================
# 2. NAÏVE BAYES: Hot vs. Iced Drink Preference Classifier
# ==============================================================================
def train_naive_bayes():
    print("[2/6] Training Naïve Bayes (Hot vs Iced Classifier)...")
    np.random.seed(42)
    n_samples = 320
    
    time_of_day_opts = ["Morning", "Afternoon", "Evening"]
    order_type_opts = ["Dine-in", "Takeaway"]
    weather_opts = ["Warm", "Cool"]
    
    rows = []
    for _ in range(n_samples):
        tod = np.random.choice(time_of_day_opts, p=[0.5, 0.35, 0.15])
        ot = np.random.choice(order_type_opts, p=[0.4, 0.6])
        w = np.random.choice(weather_opts, p=[0.65, 0.35])
        
        # Ground truth probability for Iced
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
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "drink_preference.csv"), index=False)
    
    X = df[["time_of_day", "order_type", "weather"]]
    y = df["preference"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("nb", MultinomialNB(alpha=1.0)) # Laplace smoothing alpha=1.0
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    metrics_report["naive_bayes"] = {
        "algorithm": "Multinomial Naive Bayes (Laplace alpha=1.0)",
        "accuracy": round(float(acc), 4),
        "samples": int(n_samples),
        "classes": [str(c) for c in pipeline.named_steps["nb"].classes_]
    }
    print(f"      Accuracy: {acc:.3f}")

# ==============================================================================
# 3. K-NEAREST NEIGHBORS: Personalized Drink Menu Recommendation
# ==============================================================================
def train_knn():
    print("[3/6] Training K-Nearest Neighbors (Drink Recommender)...")
    menu = [
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
    df_menu = pd.DataFrame(menu)
    df_menu.to_csv(os.path.join(DATA_DIR, "drinks_menu.csv"), index=False)
    
    np.random.seed(42)
    rows = []
    for item in menu:
        for _ in range(25):
            s = np.clip(item["sweetness"] + np.random.normal(0, 0.4), 1.0, 10.0)
            c = np.clip(item["caffeine"] + np.random.normal(0, 0.4), 1.0, 10.0)
            rows.append({
                "sweetness_pref": round(float(s), 2),
                "caffeine_pref": round(float(c), 2),
                "recommended_drink": item["drink_name"]
            })
            
    df_train = pd.DataFrame(rows)
    df_train.to_csv(os.path.join(DATA_DIR, "customer_taste_profiles.csv"), index=False)
    
    X = df_train[["sweetness_pref", "caffeine_pref"]]
    y = df_train["recommended_drink"]
    
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(X, y)
    
    joblib.dump(knn, os.path.join(MODELS_DIR, "knn.pkl"))
    metrics_report["knn"] = {
        "algorithm": "K-Nearest Neighbors (k=3, Euclidean metric)",
        "menu_items_count": int(len(menu)),
        "samples": int(len(df_train))
    }
    print(f"      Trained KNN on {len(df_train)} customer taste points across {len(menu)} drinks.")

# ==============================================================================
# 4. SUPPORT VECTOR MACHINE: Customer Review Sentiment Classifier
# ==============================================================================
def train_svm():
    print("[4/6] Training SVM (Review Sentiment Analyzer)...")
    reviews = [
        # Positive
        ("The cold brew was incredibly smooth, floral, and gave me the cleanest energy boost!", "Positive"),
        ("Best artisanal latte in town! Barista made stunning latte art and the froth was silky.", "Positive"),
        ("Exceptional specialty coffee beans. The Ethiopian pour-over had vivid strawberry notes.", "Positive"),
        ("Cozy atmosphere, fast and friendly service, and delicious freshly baked pastries.", "Positive"),
        ("Staff was super accommodating with my oat milk request. Highly recommended cafe!", "Positive"),
        ("RoastLab is my everyday sanctuary. Fast Wi-Fi, great ambient music, and world-class roast.", "Positive"),
        ("The caramel macchiato was balanced, not overly sweet, and tasted like real handcrafted coffee.", "Positive"),
        ("Prompt service, clean counter, and the cashier smiled warmly. Excellent experience.", "Positive"),
        ("Loved the nitro brew! Refreshing, cold, and velvety without needing any sweetener.", "Positive"),
        ("Outstanding single-origin selection and knowledgeable baristas who explain the tasting notes.", "Positive"),
        ("Always consistent quality across visits. Best cold drip I've had in years.", "Positive"),
        ("Terrific cafe vibes and top-notch customer hospitality.", "Positive"),
        
        # Negative
        ("The cold brew was sour, lukewarm, and the service was frustratingly slow.", "Negative"),
        ("Waited over 25 minutes for a simple black americano, and the counter was completely dirty.", "Negative"),
        ("Terrible experience. Milk was burnt, scalded my tongue, and the cashier was rude.", "Negative"),
        ("Overpriced mediocre coffee. The espresso tasted stale, burnt, and had a chemical bitter bite.", "Negative"),
        ("Forgot my order twice and refused to apologize. Will never visit RoastLab again.", "Negative"),
        ("The iced latte tasted watered down like ice cubes with milk. Extremely disappointed.", "Negative"),
        ("Dirty tables everywhere, noisy grinders, and unattentive staff on their phones.", "Negative"),
        ("Ordered sweet vanilla latte but it tasted bitter and sour. Completely botched.", "Negative"),
        ("Horrible service and stale pastry. Not worth the specialty price tag at all.", "Negative"),
        ("Rude barista rolled their eyes when I asked about bean origins. Worst customer experience.", "Negative"),
        ("Cup was leaking coffee everywhere and they had run out of cup sleeves.", "Negative"),
        ("Bitter, burnt espresso and unwelcoming vibe. Complete waste of money.", "Negative")
    ]
    
    df_reviews = pd.DataFrame(reviews, columns=["review_text", "sentiment"])
    df_reviews.to_csv(os.path.join(DATA_DIR, "customer_reviews.csv"), index=False)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)),
        ("svm", SVC(kernel="linear", C=1.0, random_state=42))
    ])
    pipeline.fit(df_reviews["review_text"], df_reviews["sentiment"])
    
    preds = pipeline.predict(df_reviews["review_text"])
    acc = accuracy_score(df_reviews["sentiment"], preds)
    
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "svm_sentiment.pkl"))
    metrics_report["svm"] = {
        "algorithm": "Support Vector Classifier (Linear Kernel Boundary)",
        "accuracy": round(float(acc), 4),
        "support_vectors": int(pipeline.named_steps["svm"].support_vectors_.shape[0]),
        "samples": int(len(df_reviews))
    }
    print(f"      Accuracy: {acc:.3f}, Support vectors: {pipeline.named_steps['svm'].support_vectors_.shape[0]}")

# ==============================================================================
# 5. DECISION TREE: Single-Origin Coffee Bean Selector (Entropy / ID3)
# ==============================================================================
def train_decision_tree():
    print("[5/6] Training Decision Tree (Bean Selector Quiz)...")
    data = [
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
    df_beans = pd.DataFrame(data, columns=["body", "acidity", "flavor_profile", "recommended_bean"])
    df_beans.to_csv(os.path.join(DATA_DIR, "coffee_beans_quiz.csv"), index=False)
    
    X = df_beans[["body", "acidity", "flavor_profile"]]
    y = df_beans["recommended_bean"]
    
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("dt", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42))
    ])
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "decision_tree.pkl"))
    metrics_report["decision_tree"] = {
        "algorithm": "ID3 Decision Tree (Information Gain / Entropy Criterion)",
        "accuracy": round(float(acc), 4),
        "tree_depth": int(pipeline.named_steps["dt"].get_depth()),
        "leaves_count": int(pipeline.named_steps["dt"].get_n_leaves()),
        "samples": int(len(df_beans))
    }
    print(f"      Accuracy: {acc:.3f}, Depth: {pipeline.named_steps['dt'].get_depth()}, Leaves: {pipeline.named_steps['dt'].get_n_leaves()}")

# ==============================================================================
# 6. ARTIFICIAL NEURAL NETWORK (ANN / MLP): Roaster QC & Cupping Score Predictor
# ==============================================================================
def train_ann():
    print("[6/6] Training Artificial Neural Network (Roaster QC & Cupping Score)...")
    np.random.seed(42)
    n_samples = 350
    
    # Roasting Telemetry
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
            
    df_roast = pd.DataFrame({
        "charge_temp_c": np.round(charge_temp, 1),
        "drop_temp_c": np.round(drop_temp, 1),
        "roast_time_min": np.round(roast_time, 2),
        "first_crack_time_min": np.round(crack_time, 2),
        "avg_ror_c_per_min": np.round(ror_avg, 2),
        "sca_cupping_score": np.round(score, 2),
        "quality_tier": tiers
    })
    df_roast.to_csv(os.path.join(DATA_DIR, "roast_quality_telemetry.csv"), index=False)
    
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
    
    joblib.dump(mlp_pipeline, os.path.join(MODELS_DIR, "ann_roast_qc.pkl"))
    metrics_report["ann"] = {
        "algorithm": "Artificial Neural Network (Multi-Layer Perceptron Regressor)",
        "architecture": "Input [5] -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear Output)",
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 3),
        "iterations": int(reg.regressor_.n_iter_),
        "samples": int(n_samples)
    }
    print(f"      R2: {r2:.3f}, RMSE: {rmse:.2f} pts, Iterations: {reg.regressor_.n_iter_}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ROASTLAB COFFEE CO. - MACHINE LEARNING PIPELINE TRAINING")
    print("=" * 60)
    train_linear_regression()
    train_naive_bayes()
    train_knn()
    train_svm()
    train_decision_tree()
    train_ann()
    
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, default=convert_types)
    print("=" * 60)
    print("All 6 models and datasets successfully generated and stored!")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)
