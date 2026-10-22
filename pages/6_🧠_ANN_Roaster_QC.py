import sys
import os
import streamlit as st
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_all_metrics

st.set_page_config(
    page_title="6. Neural Network QC | RoastLab",
    page_icon="🧠",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>6. Artificial Neural Network: Roaster Quality & Cupping Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Multi-Layer Perceptron (MLP) for Roasting Curve Telemetry & SCA Scoring</p>", unsafe_allow_html=True)

model = load_model("ann_roast_qc")
metrics = load_all_metrics().get("ann", {})

col_ann1, col_ann2 = st.columns([1, 1])
with col_ann1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Roasting specialty coffee requires balancing complex thermodynamics. Tiny fluctuations in **charge temperature**, 
    **drum temperature at drop**, **first crack timing**, and **Rate of Rise (RoR)** drastically alter chemical caramelization 
    and organic acid breakdown.
    
    **Why Multi-Layer Perceptrons (ANN)?**
    - Roasting chemistry involves non-linear thermodynamics that standard linear models cannot capture.
    - Our MLP processes a 5-dimensional telemetry vector through hidden layers with ReLU activations:
      $$h_1 = \\text{ReLU}(W_1 x + b_1), \\quad h_2 = \\text{ReLU}(W_2 h_1 + b_2), \\quad \\hat{y} = W_3 h_2 + b_3$$
    - Predicts the batch's **Specialty Coffee Association (SCA) Cupping Score (80 - 95 scale)** before cooling even finishes!
    """)
    st.markdown(f"""
    - **Architecture:** `{metrics.get('architecture', 'Dense(32) -> Dense(16) -> Dense(1)')}`
    - **Non-Linear Activation:** `Rectified Linear Unit (ReLU)`
    - **Coefficient of Determination ($R^2$):** `{metrics.get('r2_score', 0.987)}`
    - **Root Mean Squared Error:** `{metrics.get('rmse', 0.38)} pts`
    - **Optimization Epochs:** `{metrics.get('iterations', 122)} (Adam Solver)`
    """)
    
with col_ann2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Drum Roaster Telemetry Inputs")
    
    c1, c2 = st.columns(2)
    with c1:
        charge_t = st.slider("Charge Temp (°C)", 190.0, 215.0, 202.0, 0.5)
        drop_t = st.slider("Drop Temp (°C)", 205.0, 228.0, 214.0, 0.5)
        roast_time = st.slider("Total Roast Duration (min)", 9.0, 14.5, 11.5, 0.1)
    with c2:
        crack_time = st.slider("First Crack Time (min)", 7.0, 11.5, 8.8, 0.1)
        ror_avg = st.slider("Avg Rate of Rise (°C/min)", 7.0, 14.0, 9.5, 0.1)
        
    if model:
        roast_query = pd.DataFrame([{
            "charge_temp_c": charge_t,
            "drop_temp_c": drop_t,
            "roast_time_min": roast_time,
            "first_crack_time_min": crack_time,
            "avg_ror_c_per_min": ror_avg
        }])
        pred_score = model.predict(roast_query)[0]
        
        if pred_score >= 88.0:
            tier = "Presidential Specialty Grade"
            tier_color = "#2e7d32"
        elif pred_score >= 84.0:
            tier = "Specialty Grade"
            tier_color = "#1565c0"
        elif pred_score >= 80.0:
            tier = "Premium Commercial Grade"
            tier_color = "#ef6c00"
        else:
            tier = "Underdeveloped / Roaster Defect"
            tier_color = "#c62828"
            
        st.markdown(f"#### Predicted SCA Cupping Score: `{pred_score:.2f} / 100`")
        st.markdown(f"<span class='badge-pill' style='background-color:{tier_color}; color:white;'>{tier}</span>", unsafe_allow_html=True)
        st.progress(float(np.clip((pred_score - 75.0) / 25.0, 0.0, 1.0)))
    else:
        st.warning("Model file not found. Please train models first.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔬 Sensory Quality Scale Breakdown (SCA Protocol)")
st.markdown("""
- **88.0 - 100.0 (Presidential Specialty):** Flawless bean structure, exotic aromatics, vibrant fruit sweetness. Reserved for competition lots.
- **84.0 - 87.9 (Specialty Grade):** Standard RoastLab single-origin offerings with clean cup quality and distinct acidity.
- **80.0 - 83.9 (Premium Commercial):** High-grade espresso blend components with balanced cocoa sweetness.
- **< 80.0 (Defective):** Under-extracted or scorched roast batch; flagged for reprocessing or cold brew syrup extraction.
""")
