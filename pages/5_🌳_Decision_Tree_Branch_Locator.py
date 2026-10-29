import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_dataset, load_all_metrics

st.set_page_config(
    page_title="5. Decision Tree | RoastLab",
    page_icon="🌳",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>5. Decision Tree (ID3): Coffee Shop Location Feasibility (Yes / No)</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Binary Environmental Site Selection Evaluator Using Entropy & Information Gain</p>", unsafe_allow_html=True)

model = load_model("decision_tree")
metrics = load_all_metrics().get("decision_tree", {})
df_sites = load_dataset("coffee_shop_location_decision")

col_desc, col_inputs = st.columns([1, 1])

with col_desc:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Before leasing a new commercial retail space, RoastLab evaluates 4 primary environmental conditions 
    to decide whether to open a new coffee shop branch:
    - **Foot Traffic Density:** High, Moderate, or Low pedestrian volume
    - **Rent Cost:** Affordable, Moderate, or Expensive commercial lease
    - **Local Coffee Competition:** Low or High competitor density
    - **Site Accessibility:** Good or Poor transit and pedestrian access
    
    The decision tree simplifies this complex real estate choice into a direct binary outcome:
    - 🟢 **YES:** Location is favorable and recommended for opening.
    - 🔴 **NO:** Location is unfavorable; do not open due to excessive financial risk.
    """)
    
    st.markdown(r"""
    **Mathematical Formulation (Binary Entropy & Information Gain):**  
    $$\text{Entropy}(S) = -p_{\text{yes}} \log_2(p_{\text{yes}}) - p_{\text{no}} \log_2(p_{\text{no}})$$
    $$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
    """)
    
    st.markdown(f"""
    - **Target Outputs:** `2 Classes: Yes / No`
    - **Splitting Criterion:** `Entropy / Information Gain (ID3)`
    - **Tree Depth:** `{metrics.get('tree_depth', 5)} decision levels`
    - **Training Observations:** `{metrics.get('samples', 17)} Evaluated Site Surveys`
    - **Model Classification Accuracy:** `{metrics.get('accuracy', 1.0):.1%}`
    """)

with col_inputs:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🏢 Environmental Site Conditions")
    st.caption("Specify the environmental conditions of the proposed location:")
    
    c1, c2 = st.columns(2)
    with c1:
        foot_traffic = st.selectbox(
            "1. Foot Traffic Density",
            ["High", "Moderate", "Low"],
            index=0,
            help="Daily pedestrian volume passing the storefront"
        )
        rent_cost = st.selectbox(
            "2. Commercial Rent Cost",
            ["Affordable", "Moderate", "Expensive"],
            index=0,
            help="Monthly lease overhead relative to local market average"
        )
    with c2:
        competition = st.selectbox(
            "3. Local Coffee Competition",
            ["Low", "High"],
            index=0,
            help="Density of existing cafes and coffee shops within 500m"
        )
        accessibility = st.selectbox(
            "4. Site Accessibility",
            ["Good", "Poor"],
            index=0,
            help="Proximity to transit stations, parking, and sidewalk flow"
        )
        
    if model:
        input_data = pd.DataFrame([{
            "foot_traffic": foot_traffic,
            "rent_cost": rent_cost,
            "competition": competition,
            "accessibility": accessibility
        }])
        
        prediction = model.predict(input_data)[0]
        
        if prediction == "Yes":
            st.markdown("""
            <div style="background-color: #e8f5e9; border: 2px solid #2e7d32; border-radius: 10px; padding: 1.2rem; margin-top: 1rem;">
                <h3 style="color: #1b5e20; margin: 0; font-weight: 800;">🟢 DECISION: YES (Open Coffee Shop Here)</h3>
                <p style="color: #2e7d32; font-size: 0.95rem; margin: 0.5rem 0 0 0; line-height: 1.5;">
                    <strong>Favorable Site:</strong> This location satisfies all feasibility thresholds. 
                    The combination of foot traffic and acceptable overhead creates high expected profitability.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #ffebee; border: 2px solid #c62828; border-radius: 10px; padding: 1.2rem; margin-top: 1rem;">
                <h3 style="color: #b71c1c; margin: 0; font-weight: 800;">🔴 DECISION: NO (Do Not Open Here)</h3>
                <p style="color: #c62828; font-size: 0.95rem; margin: 0.5rem 0 0 0; line-height: 1.5;">
                    <strong>Unfavorable Site:</strong> Environmental conditions fail critical expansion criteria. 
                    Low customer volume, expensive rent, or heavy competition present an unacceptable financial risk.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Model file not found. Please train models first.")
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

col_table, col_rules = st.columns([1.2, 1.0])

with col_table:
    st.markdown("### 📊 Environmental Site Survey Dataset")
    st.caption("Historical surveyed locations used to train the ID3 decision tree:")
    if df_sites is not None:
        st.dataframe(df_sites, hide_index=True, use_container_width=True)
    else:
        st.info("Dataset file not found.")

with col_rules:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🌳 ID3 Decision Rules Summary")
    st.write("""
    The decision tree derives the following simple, transparent rules:
    
    1. **Accessibility Gate:**
       - If **Accessibility = Poor** ➔ **NO** *(Customer drop-off is too severe)*
       
    2. **Foot Traffic Threshold:**
       - If **Foot Traffic = Low** ➔ **NO** *(Insufficient customer volume)*
       
    3. **High Foot Traffic Scenarios:**
       - If **Foot Traffic = High** and **Rent ≠ Expensive** ➔ **YES**
       - If **Foot Traffic = High**, **Rent = Expensive**, but **Competition = Low** ➔ **YES**
       - If **Foot Traffic = High**, **Rent = Expensive**, and **Competition = High** ➔ **NO**
       
    4. **Moderate Foot Traffic Scenarios:**
       - If **Foot Traffic = Moderate**, **Competition = Low**, and **Rent ≠ Expensive** ➔ **YES**
       - If **Foot Traffic = Moderate** and **Competition = High** ➔ **NO**
    """)
    st.markdown("</div>", unsafe_allow_html=True)
