import sys
import os
import streamlit as st
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model

st.set_page_config(
    page_title="RoastLab POS • Smart Category Dispatch",
    page_icon="☕",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

# Top Navigation / Terminal Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ebd9c8; padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
    <div>
        <h2 class='brand-title' style='margin:0;'>☕ RoastLab POS & Kiosk Dispatch</h2>
        <span style='color: #795548; font-size: 0.95rem; font-weight: 500;'>Terminal #04 • Downtown Flagship • Automated Menu Pre-load Engine</span>
    </div>
    <div style="text-align: right;">
        <span class="badge-pill badge-active">🟢 POS ONLINE</span>
        <span class="badge-pill badge-specialty">⚡ AI ENGINE ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model("naive_bayes")

# Live Environment Inputs (Simulating automated store sensors & channel feeds)
st.markdown("##### 📡 Store Context & Environment Sensors")
col_env1, col_env2, col_env3 = st.columns(3)

with col_env1:
    shift_selection = st.selectbox(
        "🕒 Current Store Shift",
        ["Morning (06:00 - 11:30)", "Afternoon (11:30 - 17:00)", "Evening (17:00 - Close)"],
        index=0
    )
    time_val = "Morning" if "Morning" in shift_selection else ("Afternoon" if "Afternoon" in shift_selection else "Evening")

with col_env2:
    channel_selection = st.selectbox(
        "🚗 Order Channel / Mode",
        ["Takeaway / Drive-Thru Lane", "Dine-In Lounge"],
        index=0
    )
    order_val = "Takeaway" if "Takeaway" in channel_selection else "Dine-in"

with col_env3:
    weather_selection = st.selectbox(
        "🌤️ Ambient Weather Telemetry",
        ["Warm & Sunny (29°C)", "Cool & Overcast (18°C)"],
        index=0
    )
    weather_val = "Warm" if "Warm" in weather_selection else "Cool"

st.markdown("<br>", unsafe_allow_html=True)

# Run Inference
if model:
    input_data = pd.DataFrame([{
        "time_of_day": time_val,
        "order_type": order_val,
        "weather": weather_val
    }])
    
    pred_category = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probs))
    conf_score = prob_dict[pred_category] * 100

    col_main, col_telemetry = st.columns([1.6, 1.0])

    # Left Column: Customer-Facing Kiosk Display & Barista Station
    with col_main:
        if pred_category == "Iced":
            banner_bg = "#e1f5fe"
            banner_border = "#0288d1"
            banner_title = "🧊 KIOSK DISPLAY: ICED REFRESHERS & COLD BREWS"
            barista_directive = "⚡ Barista Queue Directive: Pre-chill glassware, verify cold drip draft pressure, ready fresh ice bin."
            catalog_items = [
                {"name": "Signature Cold Brew", "price": "$5.50", "tag": "20h Steep • Single Origin", "desc": "Smooth, bold cocoa undertones, served over crystal clear ice."},
                {"name": "Nitro Draft Coffee", "price": "$6.25", "tag": "Draft Tap • Dairy Free", "desc": "Velvety nitrogen-infused micro-bubbles with natural creamy sweetness."},
                {"name": "Iced Brown Sugar Oat Latte", "price": "$6.75", "tag": "Hand-shaken • Oat Milk", "desc": "Double espresso shaken with spiced organic brown sugar and creamy oat."}
            ]
        else:
            banner_bg = "#fbe9e7"
            banner_border = "#d84315"
            banner_title = "☕ KIOSK DISPLAY: ARTISAN HOT ESPRESSO & STEAMED LATTES"
            barista_directive = "⚡ Barista Queue Directive: Pre-warm ceramic demitasse cups, purge grouphead, engage steam wand."
            catalog_items = [
                {"name": "Artisanal Flat White", "price": "$5.25", "tag": "Double Ristretto • Whole Milk", "desc": "Rich espresso layered beneath a glossy 5mm blanket of textured microfoam."},
                {"name": "Single-Origin Cortado", "price": "$4.75", "tag": "1:1 Ratio • Balanced", "desc": "Equal parts rich espresso and warm steamed milk to cut natural acidity."},
                {"name": "Mocha Truffle Latte", "price": "$6.50", "tag": "Valrhona Cacao • Steamed", "desc": "House-melted dark chocolate ganache swirled with fresh espresso."}
            ]

        st.markdown(f"""
        <div style="background-color: {banner_bg}; border-left: 6px solid {banner_border}; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.2rem;">
            <h4 style="margin: 0 0 0.4rem 0; color: #212121; font-weight: 700;">{banner_title}</h4>
            <span style="font-size: 0.92rem; color: #424242; font-weight: 500;">{barista_directive}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🛒 Top Featured Items on Customer Screen")
        for item in catalog_items:
            c_item, c_btn = st.columns([3.5, 1.2])
            with c_item:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #ebd9c8; border-radius:8px; padding:0.9rem; margin-bottom:0.7rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.05rem; font-weight:700; color:#3e2723;">{item['name']}</span>
                        <span style="font-size:1.05rem; font-weight:800; color:#d84315;">{item['price']}</span>
                    </div>
                    <span class="badge-pill badge-specialty" style="margin-top:0.3rem;">{item['tag']}</span>
                    <p style="font-size:0.86rem; color:#616161; margin:0.4rem 0 0 0;">{item['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                if st.button("➕ Quick Add", key=f"btn_{item['name']}"):
                    st.toast(f"✅ Added {item['name']} to POS Order Ticket!")

    # Right Column: Decision Telemetry & Operational KPIs
    with col_telemetry:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ AI Dispatch Telemetry")
        st.metric(
            label="Predicted Customer Affinity",
            value=f"{pred_category} Beverage",
            delta=f"{conf_score:.1f}% Confidence"
        )
        st.caption("Categorical distribution across current store operating state:")
        
        df_prob = pd.DataFrame({
            "Category": classes,
            "Probability": probs
        })
        chart = alt.Chart(df_prob).mark_bar(cornerRadius=5).encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Category:N", sort="-x"),
            color=alt.Color("Category:N", scale=alt.Scale(domain=["Hot", "Iced"], range=["#d84315", "#0288d1"]), legend=None)
        ).properties(height=120)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Store Efficiency Metrics")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric("Queue Time", "-38 sec", delta="vs static menu")
        with kpi2:
            st.metric("Conversion", "89.4%", delta="+12.1% uplift")
        st.caption("Active model: `MultinomialNB-v1.4` • Zero cold-start latency")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Prediction engine offline. Please ensure trained models are loaded.")
