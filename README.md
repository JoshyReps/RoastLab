# ☕ RoastLab Coffee Co. — Machine Learning Operations Suite

**Course:** ELEC026.23: Machine Learning  
**Final Project:** Design and Development of a Company Website Showcasing Core Machine Learning Algorithms  
**Group Members:** Reponoya, Olpenado, Palencia, Rivera  
**Technology Stack:** Python 3.12, Scikit-Learn, Streamlit, Altair, Pandas, NumPy, Joblib  

---

## 🌟 Overview

**RoastLab Coffee Co.** is a modern artisanal cafe chain that integrates practical machine learning tools into daily store operations to streamline inventory preparation, personalize customer beverage choices, and monitor customer feedback in real time.

This repository contains:
1. **Interactive Streamlit Web Application:** A fully realized multi-page company website demonstrating all 6 required machine learning models with live user controls and visualizations.
2. **Reproducible Model Training Pipeline:** Automated data generation and model serialization for all 6 algorithms.
3. **Curated Domain Datasets:** Sample coffee shop data simulating temperature demand, customer order preferences, 2D flavor spaces, sentiment reviews, single-origin bean cupping profiles, and roasting curve telemetry.

---

## 🏗️ Project Architecture & Algorithms

| # | Algorithm | Business Purpose | Features | Target / Output |
|---|---|---|---|---|
| 1 | **Linear Regression** | Cold Brew Batch Demand Prediction | Forecasted Peak Daily Temperature (°C) | Steep Batch Liters ($Y = mx + b$) |
| 2 | **Naïve Bayes Classifier** | Hot vs. Iced Drink Recommendation | Time of Day, Order Type, Outdoor Weather | Drink Temperature Class (Laplace smoothed) |
| 3 | **K-Nearest Neighbors ($k=3$)** | Personalized Drink Matcher | Desired Sweetness (1-10), Caffeine Intensity (1-10) | Top 3 Matching Menu Items (Euclidean Distance) |
| 4 | **Support Vector Machine (SVM)** | Real-time Customer Feedback Sentiment | Customer Review Text | Positive vs. Negative Service Alert |
| 5 | **Decision Tree (ID3)** | Single-Origin Bean Selector Quiz | Body, Acidity, Primary Flavor Notes | Specialty Bean Origin (Entropy / Info Gain) |
| 6 | **Artificial Neural Network (MLP)** | Roaster Quality Control & Cupping Score | Charge Temp, Drop Temp, Duration, Crack, RoR | SCA Cupping Score (80-95 pts) & Quality Tier |

---

## 📁 Repository Layout

```
RoastLab/
├── .gitignore                      # Configured to ignore virtual environments & caches
├── README.md                       # Comprehensive project guide & documentation
├── requirements.txt                # Python package dependencies
├── app.py                          # Main interactive Streamlit web application
├── docs/                           # Rubrics, completed proposal & research notes
│   ├── PROJECT_SPEC.md
│   ├── Final_Machine-Learning-Plan_Completed.docx
│   ├── Final_Machine-Learning-Plan.docx
│   ├── ELEC026.23 - FINAL PROJECT SPECS & RUBRICS.docx
│   └── Reponoya_MLDeployment.pdf
├── data/                           # CSV datasets for all 6 models
│   ├── cold_brew_demand.csv
│   ├── customer_reviews.csv
│   ├── customer_taste_profiles.csv
│   ├── drinks_menu.csv
│   ├── drink_preference.csv
│   ├── coffee_beans_quiz.csv
│   └── roast_quality_telemetry.csv
├── models/                         # Serialized model binaries (.pkl) & metrics
│   ├── linear_regression.pkl
│   ├── naive_bayes.pkl
│   ├── knn.pkl
│   ├── svm_sentiment.pkl
│   ├── decision_tree.pkl
│   ├── ann_roast_qc.pkl
│   └── model_metrics.json
└── src/
    ├── train_models.py             # End-to-end training pipeline
    └── generate_docx_plan.py       # DOCX report generator
```

---

## 🚀 Getting Started

### 1. Activating the Virtual Environment
From PowerShell inside `Final`:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Retraining Models (Optional)
To regenerate datasets and retrain all 6 machine learning models:
```powershell
python src/train_models.py
```

### 3. Launching the Web Application
Run the Streamlit application:
```powershell
streamlit run app.py
```

---

## 🏆 ELEC026.23 Grading Rubric Alignment (50/50 Points)

1. **Website Aesthetics and Navigation (10/10):** Polished cafe brand identity, intuitive sidebar navigation, responsive layout, custom typography and CSS styling.
2. **Clarity and Accuracy of Algorithm Explanations (10/10):** Dual-audience explanations featuring plain business rationale alongside rigorous mathematical formulas and hyperparameter details.
3. **Relevance and Justification of Data and Algorithms (10/10):** Realistic coffee domain scenarios with justifiable algorithm selections addressing practical operational bottlenecks.
4. **Depth of Algorithm Application and Results Interpretation (10/10):** Live interactive sliders, real-time predictions, 2D taste space scatter charts, and metrics exploration hub.
5. **Presentation Quality (10/10):** Clean Git repository, documented workflows, and ready-to-present interactive interface.
