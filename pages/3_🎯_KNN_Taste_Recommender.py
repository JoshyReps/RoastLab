import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_dataset, load_all_metrics

st.set_page_config(
    page_title="3. KNN Recommender | RoastLab",
    page_icon="🎯",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>3. K-Nearest Neighbors: Personalized Drink Matcher</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Euclidean Distance Mapping in Continuous Taste Preference Space</p>", unsafe_allow_html=True)

df_menu = load_dataset("drinks_menu")
metrics = load_all_metrics().get("knn", {})

col_k1, col_k2 = st.columns([1, 1])
with col_k1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Customers often struggle to pick a drink from a 30-item menu. Rather than reading tasting jargon, 
    customers simply specify how sweet and how strong they want their coffee.
    
    **Why K-Nearest Neighbors ($k=3$)?**
    - Maps drinks into a normalized 2D taste space: **Sweetness (1-10)** and **Caffeine Kick (1-10)**.
    - Calculates the **Euclidean Distance** between customer desires and existing menu recipes:
      $$d(P, Q) = \\sqrt{(s_2 - s_1)^2 + (c_2 - c_1)^2}$$
    - Returns the closest matching menu items with full interpretability.
    """)
    st.markdown(f"""
    - **Neighbor Count ($k$):** `3 Nearest Neighbors`
    - **Distance Metric:** `Euclidean Metric`
    - **Menu Catalog Size:** `{metrics.get('menu_items_count', 14)} Signature Items`
    - **Customer Taste Datapoints:** `{metrics.get('samples', 350)} Order Profiles`
    """)
    
with col_k2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Customer Taste Sliders")
    sweetness_input = st.slider("Desired Sweetness Level (1 = Black Coffee, 10 = Sweet Dessert)", 1.0, 10.0, 4.0, 0.5)
    caffeine_input = st.slider("Desired Caffeine Intensity (1 = Decaf, 10 = Triple Red-Eye)", 1.0, 10.0, 7.5, 0.5)
    
    if df_menu is not None:
        df_calc = df_menu.copy()
        df_calc["distance"] = np.sqrt(
            (df_calc["sweetness"] - sweetness_input)**2 + 
            (df_calc["caffeine"] - caffeine_input)**2
        )
        top3 = df_calc.sort_values("distance").head(3)
        
        st.markdown("#### 🏆 Top 3 Recommended Drinks:")
        for idx, row in top3.iterrows():
            st.markdown(f"**{row['drink_name']}** (`{row['category']}`) — Match Distance: `{row['distance']:.2f}`")
            st.caption(f"_{row['desc']}_")
    st.markdown("</div>", unsafe_allow_html=True)
    
st.markdown("---")
st.markdown("### 🗺️ Interactive Taste Preference Space (KNN)")
if df_menu is not None:
    base_chart = alt.Chart(df_menu).mark_circle(size=120).encode(
        x=alt.X("sweetness:Q", scale=alt.Scale(domain=[0, 10.5]), title="Sweetness Level (1 - 10)"),
        y=alt.Y("caffeine:Q", scale=alt.Scale(domain=[0, 10.5]), title="Caffeine Intensity (1 - 10)"),
        color=alt.Color("category:N", title="Category"),
        tooltip=["drink_name", "category", "sweetness", "caffeine", "desc"]
    )
    labels = alt.Chart(df_menu).mark_text(dy=-12, fontSize=11).encode(
        x="sweetness:Q",
        y="caffeine:Q",
        text="drink_name:N"
    )
    user_marker = alt.Chart(pd.DataFrame({
        "sweetness": [sweetness_input],
        "caffeine": [caffeine_input],
        "label": ["★ YOUR TASTE PREFERENCE"]
    })).mark_point(color="#d84315", size=300, shape="star", filled=True).encode(
        x="sweetness:Q",
        y="caffeine:Q"
    )
    user_label = alt.Chart(pd.DataFrame({
        "sweetness": [sweetness_input],
        "caffeine": [caffeine_input],
        "label": ["★ Current Selection"]
    })).mark_text(dy=18, fontWeight="bold", color="#d84315").encode(
        x="sweetness:Q",
        y="caffeine:Q",
        text="label:N"
    )
    st.altair_chart((base_chart + labels + user_marker + user_label).properties(height=420), use_container_width=True)
