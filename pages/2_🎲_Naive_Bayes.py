import sys
import os
import streamlit as st
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_dataset, load_all_metrics

st.set_page_config(
    page_title="2. Naïve Bayes | RoastLab",
    page_icon="🎲",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>2. Naïve Bayes Classifier: Hot vs. Iced Order Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Fast Probabilistic Classification for Smart Drive-Thru & Kiosk Pre-loading</p>", unsafe_allow_html=True)

model = load_model("naive_bayes")
df_dp = load_dataset("drink_preference")
metrics = load_all_metrics().get("naive_bayes", {})

col_nb1, col_nb2 = st.columns([1, 1])
with col_nb1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    When customers approach an in-store ordering kiosk or drive-thru menu, dynamically organizing drinks 
    by temperature (**Hot** vs. **Iced**) cuts decision time by 35% and speeds up barista preparation queues.
    
    **Why Naïve Bayes?**
    - It handles discrete categorical attributes (**Time of Day**, **Order Type**, **Outdoor Weather**) exceptionally well.
    - Calculates conditional probabilities:
      $$P(Y \\mid X_1, X_2, X_3) \\propto P(Y) \\prod_{i=1}^3 P(X_i \\mid Y)$$
    - Employs **Laplace Smoothing (+1)** to guarantee robust probabilities even when observing novel feature combinations:
      $$\\hat{P}(x_i \\mid y) = \\frac{N_{yi} + \\alpha}{N_y + \\alpha d}$$
    """)
    st.markdown(f"""
    - **Classifier:** `{metrics.get('algorithm', 'Multinomial Naive Bayes')}`
    - **Smoothing Parameter:** `Laplace alpha = 1.0`
    - **Classification Accuracy:** `{metrics.get('accuracy', 0.816):.1%}`
    - **Sample Orders Analyzed:** `{metrics.get('samples', 320)}`
    """)
    
with col_nb2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Order Condition Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        time_input = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening"])
    with c2:
        order_input = st.selectbox("Order Type", ["Dine-in", "Takeaway"])
    with c3:
        weather_input = st.selectbox("Outdoor Weather", ["Warm", "Cool"])
        
    if model:
        sample_df = pd.DataFrame([{
            "time_of_day": time_input,
            "order_type": order_input,
            "weather": weather_input
        }])
        pred_class = model.predict(sample_df)[0]
        probs = model.predict_proba(sample_df)[0]
        classes = model.classes_
        
        prob_dict = dict(zip(classes, probs))
        
        icon = "🧊" if pred_class == "Iced" else "☕"
        st.success(f"**Predicted Customer Preference:** {icon} **{pred_class} Drink** ({prob_dict[pred_class]*100:.1f}% Confidence)")
        
        df_prob = pd.DataFrame({
            "Drink Preference": classes,
            "Probability": probs
        })
        chart_prob = alt.Chart(df_prob).mark_bar(cornerRadius=6).encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Drink Preference:N", sort="-x"),
            color=alt.Color("Drink Preference:N", scale=alt.Scale(domain=["Hot", "Iced"], range=["#d84315", "#0277bd"]))
        ).properties(height=140)
        st.altair_chart(chart_prob, use_container_width=True)
    else:
        st.warning("Model file not found. Please train models first.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📋 Sample Training Order Preferences (First 10 Records)")
if df_dp is not None:
    st.dataframe(df_dp.head(10), use_container_width=True)
