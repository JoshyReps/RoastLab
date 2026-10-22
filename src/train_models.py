"""
RoastLab Coffee Co. - Master Training Orchestrator
Executes each modular algorithm pipeline and stores consolidated model_metrics.json.
"""

import os
import json
import numpy as np

from linear_regression import train_and_save as train_lr
from naive_bayes import train_and_save as train_nb
from knn import train_and_save as train_knn
from svm import train_and_save as train_svm
from decision_tree import train_and_save as train_dt
from ann import train_and_save as train_ann

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def convert_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_all():
    print("=" * 60)
    print("ROASTLAB COFFEE CO. - MODULAR PIPELINE TRAINING")
    print("=" * 60)
    
    metrics_report = {}
    
    print("[1/6] Running Linear Regression...")
    metrics_report["linear_regression"] = train_lr(DATA_DIR, MODELS_DIR)
    
    print("[2/6] Running Naïve Bayes...")
    metrics_report["naive_bayes"] = train_nb(DATA_DIR, MODELS_DIR)
    
    print("[3/6] Running K-Nearest Neighbors...")
    metrics_report["knn"] = train_knn(DATA_DIR, MODELS_DIR)
    
    print("[4/6] Running Support Vector Machine...")
    metrics_report["svm"] = train_svm(DATA_DIR, MODELS_DIR)
    
    print("[5/6] Running Decision Tree...")
    metrics_report["decision_tree"] = train_dt(DATA_DIR, MODELS_DIR)
    
    print("[6/6] Running Artificial Neural Network (ANN)...")
    metrics_report["ann"] = train_ann(DATA_DIR, MODELS_DIR)
    
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, default=convert_types)
        
    print("=" * 60)
    print("All 6 models and datasets successfully retrained and verified!")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
