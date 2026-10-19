# ELEC026.23: Machine Learning — Final Project Specification

**Course:** ELEC026.23: Machine Learning  
**Project Title:** Design and Development of a Company Website Showcasing Core Machine Learning Algorithms  
**Company:** RoastLab Coffee Co.  
**Industry:** Specialty Food & Beverage / Smart Retail  
**Group Members:** Reponoya, Olpenado, Palencia, Rivera  

---

## 1. Company Profile

- **Company Description:** RoastLab Coffee Co. is a modern artisanal cafe chain integrating practical machine learning into daily store operations to streamline inventory preparation, personalize customer beverage choices, and monitor customer feedback in real time.
- **Vision:** To deliver an unmatched artisanal coffee experience by seamlessly pairing hand-crafted brewing with intelligent, data-driven cafe management.
- **Mission:** To empower store managers and baristas with transparent machine learning tools that reduce food waste, simplify drink selection for guests, and ensure consistent service quality across all branches.
- **Operational Goals:**
  1. Accurately forecast daily batch production to minimize ingredient waste and prevent stockouts.
  2. Deliver personalized drink and coffee bean recommendations to increase customer satisfaction and boost sales.
  3. Automate review sentiment analysis to identify and resolve customer service issues in real time.
  4. Ensure precise roasting quality control across batches using deep sensory profile prediction.

---

## 2. Core Machine Learning Algorithms Mapping

| # | Algorithm | Business Function / Problem | Inputs | Target / Output |
|---|---|---|---|---|
| 1 | **Linear Regression** | Cold Brew Batch Demand Prediction | Forecasted Peak Daily Temperature (°C) | Recommended Batch Steep Volume (Liters) |
| 2 | **Naïve Bayes Classifier** | Drink Temperature Preference Prediction | Time of Day, Order Type, Weather Condition | Drink Type (Hot vs. Iced) with Laplace-smoothed probabilities |
| 3 | **K-Nearest Neighbors (KNN)** | Personalized Drink Menu Recommendation | Sweetness Preference (1-10), Caffeine Strength (1-10) | Top-K Matching Coffee Drinks with Euclidean distance metrics |
| 4 | **Support Vector Machine (SVM)** | Real-time Customer Feedback Sentiment Analysis | Customer Review Text | Sentiment Classification (Positive vs. Negative) with confidence |
| 5 | **Decision Tree (ID3)** | Single-Origin Bean Selector Quiz | Body Preference, Acidity Preference, Primary Flavor Profile | Recommended Single-Origin Bean (Ethiopian, Colombian, Sumatra, etc.) |
| 6 | **Artificial Neural Network (ANN / MLP)** | Roaster Quality Control & Sensory Cupping Score | Charge Temp, Drop Temp, Total Roast Time, First Crack Time, Average Rate of Rise (RoR) | Predicted SCA Cupping Score (80-95 scale) & Quality Tier |

---

## 3. Rubric Breakdown & Target Scoring (50/50 Points)

1. **Website Aesthetics & Navigation (10 pts):** Professional, visually appealing cafe-themed UI with clean sidebar navigation, responsive layout, and interactive widgets.
2. **Clarity & Accuracy of Explanations (10 pts):** Accurate, plain-language business explanations alongside technical metrics (equations, loss curves, confusion matrices, information gain).
3. **Relevance & Justification of Data & Algorithms (10 pts):** Clear domain rationale linking coffee retail challenges to each model's mathematical strengths.
4. **Depth of Application & Results Interpretation (10 pts):** Real-time interactive predictions, confidence intervals, sample dataset inspections, and actionable business recommendations.
5. **Presentation Quality (10 pts):** Clean code structure, reproducible training pipelines, and seamless demo readiness.
