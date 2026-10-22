import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_dataset, load_all_metrics

st.set_page_config(
    page_title="1. Linear Regression | RoastLab",
    page_icon="📈",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>1. Linear Regression: Cold Brew Batch Demand Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Optimizing 24-Hour Steeping Batches from Weather Forecasts</p>", unsafe_allow_html=True)

model = load_model("linear_regression")
df_cb = load_dataset("cold_brew_demand")
metrics = load_all_metrics().get("linear_regression", {})

col_desc, col_model = st.columns([1, 1])
with col_desc:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Cold brew coffee requires **18 to 24 hours of cold water extraction**. Because of this long preparation cycle, 
    baristas cannot simply make more on demand when temperatures spike. 
    - **Underestimating demand** results in sold-out drinks during peak hot hours (lost revenue).
    - **Overestimating demand** leads to oxidized coffee that must be poured down the drain (spoilage).
    
    Using **Ordinary Least Squares (OLS) Linear Regression**, RoastLab models the relationship between forecasted peak daytime 
    temperature ($X$) and the total liters of cold brew ordered ($Y$).
    """)
    
    st.markdown(f"""
    **Mathematical Formulation:**  
    $$Y = \\beta_1 X + \\beta_0 + \\epsilon$$
    - **Fitted Model Equation:** `{metrics.get('formula', 'Y = 0.960 * X - 4.5')}`
    - **Coefficient of Determination ($R^2$):** `{metrics.get('r2_score', 0.909)}`
    - **Root Mean Squared Error (RMSE):** `{metrics.get('rmse', 1.78)} Liters`
    - **Training Observations:** `{metrics.get('samples', 180)} Operating Days`
    """)
    
with col_model:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Live Batch Estimator")
    temp_input = st.slider(
        "Tomorrow's Forecasted Peak Temperature (°C)", 
        min_value=16.0, 
        max_value=40.0, 
        value=29.5, 
        step=0.5
    )
    
    if model:
        pred = model.predict([[temp_input]])[0]
        st.metric(
            label="Recommended Steep Volume", 
            value=f"{pred:.1f} Liters",
            delta=f"{pred - 23.0:.1f} L vs average"
        )
        st.info(f"💡 **Barista Action:** Steep **{pred:.1f} Liters** today at 3:00 PM for tomorrow's shift. Requires ~{pred*0.08:.2f} kg of coarse ground beans.")
    else:
        st.warning("Model file not found. Please train models first.")
    st.markdown("</div>", unsafe_allow_html=True)
    
st.markdown("---")
st.markdown("### 📊 Historical Temperature vs. Batch Volume Correlation")

if df_cb is not None and model is not None:
    x_vals = np.linspace(df_cb["forecast_peak_temp_c"].min(), df_cb["forecast_peak_temp_c"].max(), 50)
    y_vals = model.predict(x_vals.reshape(-1, 1))
    df_line = pd.DataFrame({"forecast_peak_temp_c": x_vals, "cold_brew_liters_demanded": y_vals})
    
    scatter = alt.Chart(df_cb).mark_circle(size=60, color="#795548", opacity=0.7).encode(
        x=alt.X("forecast_peak_temp_c:Q", title="Forecast Peak Temperature (°C)"),
        y=alt.Y("cold_brew_liters_demanded:Q", title="Cold Brew Demanded (Liters)"),
        tooltip=["forecast_peak_temp_c", "cold_brew_liters_demanded"]
    )
    line = alt.Chart(df_line).mark_line(color="#d84315", size=3).encode(
        x="forecast_peak_temp_c:Q",
        y="cold_brew_liters_demanded:Q"
    )
    user_point = alt.Chart(pd.DataFrame({"x": [temp_input], "y": [pred]})).mark_point(
        color="#2e7d32", size=200, shape="cross"
    ).encode(x="x:Q", y="y:Q")
    
    chart = (scatter + line + user_point).properties(height=360).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("🟢 Green cross indicates your current interactive slider prediction on the fitted regression line.")
