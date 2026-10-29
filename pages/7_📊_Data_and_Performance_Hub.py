import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_dataset, load_all_metrics

st.set_page_config(
    page_title="7. Data Hub | RoastLab",
    page_icon="📊",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>📊 RoastLab Data & Evaluation Hub</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Model Benchmarks, Sample Datasets & Architecture Verification</p>", unsafe_allow_html=True)

metrics = load_all_metrics()

st.markdown("### 🏆 Algorithm Metrics Summary")
summary_rows = [
    {"#": 1, "Algorithm": "Linear Regression", "Task": "Cold Brew Volume Prediction", "Key Metric": f"R² = {metrics.get('linear_regression', {}).get('r2_score', 0.909)}", "Error": f"RMSE = {metrics.get('linear_regression', {}).get('rmse', 1.78)}L", "Samples": 180},
    {"#": 2, "Algorithm": "Naïve Bayes", "Task": "Whole Bean Origin Matching", "Key Metric": f"Accuracy = {metrics.get('naive_bayes', {}).get('accuracy', 0.604):.1%}", "Error": "Laplace α=1.0", "Samples": 500},
    {"#": 3, "Algorithm": "K-Nearest Neighbors", "Task": "5D Sensory Drink Recommender", "Key Metric": "Dynamic k (1-7)", "Error": "Normalized Euclidean", "Samples": 420},
    {"#": 4, "Algorithm": "Support Vector Machine", "Task": "Feedback Sentiment Analysis", "Key Metric": f"Accuracy = {metrics.get('svm', {}).get('accuracy', 0.907):.1%}", "Error": "Linear Margin", "Samples": metrics.get('svm', {}).get('samples', 3024)},
    {"#": 5, "Algorithm": "Decision Tree (ID3)", "Task": "Cafe Branch Site Selection", "Key Metric": f"Depth = {metrics.get('decision_tree', {}).get('tree_depth', 4)}", "Error": "Entropy Split", "Samples": metrics.get('decision_tree', {}).get('samples', 26)},
    {"#": 6, "Algorithm": "Artificial Neural Network", "Task": "Roaster QC & Cupping Score", "Key Metric": f"R² = {metrics.get('ann', {}).get('r2_score', 0.958)}", "Error": f"RMSE = {metrics.get('ann', {}).get('rmse', 0.63)} pts", "Samples": 350}
]
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("---")

col_excel, col_spacer = st.columns([2, 1])
with col_excel:
    st.markdown("### 📗 Master Spreadsheet (All Datasets in One Excel Workbook)")
    excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "RoastLab_All_Datasets.xlsx")
    if os.path.exists(excel_path):
        with open(excel_path, "rb") as f:
            st.download_button(
                label="📥 Download Master Excel Workbook (RoastLab_All_Datasets.xlsx)",
                data=f.read(),
                file_name="RoastLab_All_Datasets.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    st.caption("Contains all domain datasets arranged in separate labelled tabs.")

st.markdown("---")
st.markdown("### 📁 Explore Individual Domain Datasets")
dataset_choice = st.selectbox(
    "Select dataset to preview:",
    [
        "Cold Brew Daily Demand (Linear Regression)",
        "Whole Bean Origin Matching (Naïve Bayes)",
        "Drinks Menu Catalog (KNN)",
        "Customer Taste Profiles (KNN)",
        "Customer Review Feedback (SVM)",
        "Store Location Expansion Surveys (Decision Tree)",
        "Roast Drum Telemetry (ANN)"
    ]
)

mapping = {
    "Cold Brew Daily Demand (Linear Regression)": "cold_brew_demand",
    "Whole Bean Origin Matching (Naïve Bayes)": "drink_preference",
    "Drinks Menu Catalog (KNN)": "drinks_menu",
    "Customer Taste Profiles (KNN)": "customer_taste_profiles",
    "Customer Review Feedback (SVM)": "customer_reviews",
    "Store Location Expansion Surveys (Decision Tree)": "store_location_expansion",
    "Roast Drum Telemetry (ANN)": "roast_quality_telemetry"
}

df_selected = load_dataset(mapping[dataset_choice])
if df_selected is not None:
    st.write(f"Total rows: **{len(df_selected)}** | Columns: **{list(df_selected.columns)}**")
    st.dataframe(df_selected.head(25), use_container_width=True)
    csv_bytes = df_selected.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {dataset_choice} CSV",
        data=csv_bytes,
        file_name=f"{mapping[dataset_choice]}.csv",
        mime="text/csv"
    )
