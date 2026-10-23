# ☕ RoastLab Coffee Co. — Machine Learning Operations Suite

**Course:** ELEC026.23: Machine Learning  
**Final Project:** Design and Development of a Company Website Showcasing Core Machine Learning Algorithms  
**Group Members:** Reponoya, Olpenado, Palencia, Rivera  
**Technology Stack:** Python 3.12, Scikit-Learn, Streamlit (Multipage), Altair, Pandas, NumPy, Joblib  

---

## 🌟 Overview

**RoastLab Coffee Co.** is a modern artisanal cafe chain that integrates practical machine learning tools into daily store operations to streamline inventory preparation, personalize customer beverage choices, and monitor customer feedback in real time.

This repository features a **modular, multi-page architecture** where each machine learning algorithm and application page has its own dedicated Python file.

---

## 📁 Repository Layout & File Structure

```
RoastLab/
├── .gitignore                          # Ignores venv/ and caches
├── README.md                           # Documentation & quickstart guide
├── requirements.txt                    # Project package dependencies
├── app.py                              # Main Landing Page: Company Overview & Architecture
│
├── pages/                              # Dedicated Python file for each app page
│   ├── 1_📈_Linear_Regression.py       # Cold Brew Demand (Temperature -> Liters)
│   ├── 2_🎲_Naive_Bayes.py             # Hot vs. Iced Order Predictor
│   ├── 3_🎯_KNN_Taste_Recommender.py   # 2D Taste Space Drink Matcher
│   ├── 4_⚖️_SVM_Sentiment_Analysis.py   # Review Sentiment Classifier
│   ├── 5_🌳_Decision_Tree_Bean_Quiz.py # 3-Question Single-Origin Quiz
│   ├── 6_🧠_ANN_Roaster_QC.py          # Roaster Telemetry to SCA Cupping Score
│   └── 7_📊_Data_and_Performance_Hub.py# Datasets Explorer & Metrics Hub
│
├── src/                                # Dedicated Python module for each algorithm
│   ├── common.py                       # Shared UI styles, cached loaders, and sidebar
│   ├── linear_regression.py            # Linear Regression training & data generator
│   ├── naive_bayes.py                  # Naïve Bayes pipeline with Laplace smoothing
│   ├── knn.py                          # KNN model & 2D Euclidean taste mapper
│   ├── svm.py                          # Linear SVM with TF-IDF vectorization
│   ├── decision_tree.py                # ID3 Decision Tree (Entropy / Information Gain)
│   ├── ann.py                          # Multi-Layer Perceptron (MLP) Neural Network
│   ├── train_models.py                 # Master training orchestrator
│   └── generate_docx_plan.py           # Word DOCX proposal generator
│
├── data/                               # Domain CSV datasets & Master Excel Workbook
│   ├── RoastLab_All_Datasets.xlsx      # Master workbook with all 7 datasets
│   ├── cold_brew_demand.csv
│   ├── customer_reviews.csv
│   ├── customer_taste_profiles.csv
│   ├── drinks_menu.csv
│   ├── drink_preference.csv
│   ├── coffee_beans_quiz.csv
│   └── roast_quality_telemetry.csv
│
├── models/                             # Serialized .pkl binaries & evaluation metrics
│   ├── linear_regression.pkl
│   ├── naive_bayes.pkl
│   ├── knn.pkl
│   ├── svm_sentiment.pkl
│   ├── decision_tree.pkl
│   ├── ann_roast_qc.pkl
│   └── model_metrics.json
│
└── docs/                               # Project specs, rubrics, and completed proposal
    ├── PROJECT_SPEC.md
    ├── Final_Machine-Learning-Plan_Completed.docx
    ├── Final_Machine-Learning-Plan.docx
    ├── ELEC026.23 - FINAL PROJECT SPECS & RUBRICS.docx
    └── Reponoya_MLDeployment.pdf
```

---

## 🏗️ Algorithm Breakdown

| # | Page / Module | Business Task | Input Features | Output Target |
|---|---|---|---|---|
| 1 | **Linear Regression** | Cold Brew Batch Demand | Forecast Peak Temp (°C) | Steep Liters ($Y = mx + b$) |
| 2 | **Naïve Bayes** | Hot vs. Iced Order Predictor | Time, Order Type, Weather | Drink Class (Laplace $\alpha=1.0$) |
| 3 | **KNN ($k=3$)** | Flavor Profile Drink Matcher | Desired Sweetness & Caffeine (1-10) | Top 3 Menu Items (Euclidean Distance) |
| 4 | **Linear SVM** | Review Sentiment Tagging | Review Text | Positive vs. Negative Alert |
| 5 | **Decision Tree** | Single-Origin Bean Selector Quiz | Body, Acidity, Flavor Notes | Recommended Bean Origin (ID3) |
| 6 | **Neural Network** | Roaster Quality Control & QC | Charge/Drop Temp, RoR, Crack Time | SCA Cupping Score (80-95 pts) |

---

## 🚀 Running the App

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Launch the Multi-Page Application
```powershell
streamlit run app.py
```
Streamlit automatically detects all files in `pages/` and renders a clean, organized navigation bar in the sidebar!

### 3. Retrain Models Individually or All at Once
- Run any single algorithm script directly, e.g.:
  ```powershell
  python src/ann.py
  ```
- Or run the master pipeline:
  ```powershell
  python src/train_models.py
  ```
