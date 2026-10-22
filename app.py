import streamlit as st
import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.common import apply_custom_theme, render_sidebar_info

st.set_page_config(
    page_title="RoastLab Coffee Co. | Home",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h1 class='brand-title'>RoastLab Coffee Co.</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Artisanal Coffee Meets Machine Learning Operations</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 🎯 Smart Inventory")
    st.write("Linear Regression forecasts daily cold brew batch production to virtually eliminate food waste and stockouts.")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("#### 🤝 Hyper-Personalization")
    st.write("Naïve Bayes, KNN, and Decision Trees personalize every order from espresso strength to single-origin beans.")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("#### 🔬 Quality & Sentiment")
    st.write("SVM routes live customer reviews, while an Artificial Neural Network guarantees SCA cupping score consistency.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

tab_mission, tab_arch = st.tabs(["🏛️ Mission & Vision", "🏗️ Machine Learning Architecture"])

with tab_mission:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 🌟 Vision")
        st.info("To deliver an unmatched artisanal coffee experience by seamlessly pairing hand-crafted brewing with intelligent, data-driven cafe management.")
        st.markdown("### 🎯 Operational Goals")
        st.markdown("""
        1. **Batch Accuracy:** Forecast cold brew batch sizes 24 hours in advance to minimize spoilage on cool days and stockouts during heatwaves.
        2. **Order Conversion:** Shorten customer queue times with automated contextual recommendations (Hot vs. Iced).
        3. **Discovery:** Help novice guests order premium single-origin pour-overs without intimidation.
        4. **Quality Assurance:** Predict batch cupping scores directly from roast drum telemetry sensors.
        """)
    with col_m2:
        st.markdown("### 🚀 Mission")
        st.success("To empower store managers and baristas with transparent machine learning tools that reduce food waste, simplify drink selection for guests, and ensure consistent service quality across all branches.")
        st.markdown("### 🏷️ Brand Identity")
        st.markdown("""
        - **Industry:** Specialty Food & Beverage / Smart Retail
        - **Target Customers:** Urban commuters, specialty coffee enthusiasts, remote workers.
        - **Value Proposition:** Handcrafted excellence backed by zero-waste inventory and data-tuned roasting.
        """)
        
with tab_arch:
    st.markdown("### 🛠️ Machine Learning Operational Stack")
    st.markdown("""
    Use the **sidebar navigation** to explore each dedicated algorithm page:
    
    | Page | Algorithm | Practical Business Application | Key Metric |
    |---|---|---|---|
    | **1. Linear Regression** | Cold Brew Batch Demand Prediction | Peak Forecasted Temp (°C) $\\rightarrow$ Steep Liters | $R^2 = 0.909$, $\\text{RMSE} = 1.78\\,\\text{L}$ |
    | **2. Naïve Bayes** | Hot vs. Iced Drink Recommendation | Time, Order Type, Weather $\\rightarrow$ Drink Type | $\\text{Accuracy} = 81.6\\%$ (Laplace $\\alpha=1.0$) |
    | **3. KNN ($k=3$)** | Flavor Profile Drink Recommender | Desired Sweetness & Caffeine $\\rightarrow$ Top 3 Menu Items | Euclidean 2D Taste Space |
    | **4. Linear SVM** | Real-time Customer Sentiment Tagging | Review Text $\\rightarrow$ Positive vs. Negative Alert | $\\text{Accuracy} = 100\\%$, 24 Support Vectors |
    | **5. Decision Tree** | Single-Origin Bean Selector Quiz | Body, Acidity, Flavor Notes $\\rightarrow$ Bean Origin | Entropy / Information Gain (Depth 3) |
    | **6. Neural Network** | Roaster Telemetry Quality Control | Charge/Drop Temp, RoR, Crack Time $\\rightarrow$ Cupping Score | $R^2 = 0.987$, $\\text{RMSE} = 0.38\\,\\text{pts}$ |
    | **7. Data Hub** | Dataset Explorer & Benchmarks | View & Download CSVs, Master Excel Workbook | 7 Domain Datasets |
    """)
    

st.markdown("---")
st.info("👈 **Get Started:** Select any algorithm or the Data Hub from the left sidebar to interact with the models!")
