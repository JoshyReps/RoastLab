"""
Algorithm 4: Support Vector Machine (Linear SVM) with TF-IDF Vectorization
Business Task: Real-time Customer Review Feedback Sentiment Classifier
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

REVIEWS_DATA = [
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

def train_and_save(data_dir: str, models_dir: str) -> dict:
    df_reviews = pd.DataFrame(REVIEWS_DATA, columns=["review_text", "sentiment"])
    df_reviews.to_csv(os.path.join(data_dir, "customer_reviews.csv"), index=False)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)),
        ("svm", SVC(kernel="linear", C=1.0, random_state=42))
    ])
    pipeline.fit(df_reviews["review_text"], df_reviews["sentiment"])
    
    preds = pipeline.predict(df_reviews["review_text"])
    acc = accuracy_score(df_reviews["sentiment"], preds)
    
    joblib.dump(pipeline, os.path.join(models_dir, "svm_sentiment.pkl"))
    return {
        "algorithm": "Support Vector Classifier (Linear Kernel Boundary)",
        "accuracy": round(float(acc), 4),
        "support_vectors": int(pipeline.named_steps["svm"].support_vectors_.shape[0]),
        "samples": len(df_reviews)
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save(os.path.join(base, "data"), os.path.join(base, "models"))
    print("SVM trained successfully:", metrics)
