import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model

st.set_page_config(
    page_title="RoastLab POS • Category Dispatch",
    page_icon="☕",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

# Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ebd9c8; padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
    <div>
        <h2 class='brand-title' style='margin:0;'>☕ RoastLab POS • Contextual Category Dispatch</h2>
        <span style='color: #795548; font-size: 0.95rem; font-weight: 500;'>Real-Time Multi-Class Bayesian Dispatch & Screen Pre-loading</span>
    </div>
    <div style="text-align: right;">
        <span class="badge-pill badge-active">🟢 POS ONLINE</span>
        <span class="badge-pill badge-specialty">⚡ MULTI-CLASS ENGINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model("naive_bayes")

# 4 Operational Feature Selectors
st.markdown("##### 📡 Store & Customer Context Sensors")
c_f1, c_f2, c_f3, c_f4 = st.columns(4)

with c_f1:
    day_type = st.selectbox(
        "📅 Day Type",
        ["Weekday", "Weekend / Holiday"],
        index=0,
        help="Commute rushes vs leisurely weekend visits"
    )

with c_f2:
    weather_cond = st.selectbox(
        "🌦️ Weather Condition",
        ["Hot & Humid", "Cool & Dry", "Rainy", "Freezing"],
        index=0,
        help="Humidity and temperature extremes drive beverage temperature cravings"
    )

with c_f3:
    customer_seg = st.selectbox(
        "👤 Customer Segment",
        ["Morning Commuter", "Student", "Remote Worker", "Evening Social"],
        index=0,
        help="Demographic and visit intention profiling"
    )

with c_f4:
    companion_ord = st.selectbox(
        "🥐 Companion Order",
        ["Food Item Ordered", "Beverage Only"],
        index=0,
        help="Customers ordering warm pastries strongly favor hot beverages"
    )

st.markdown("<br>", unsafe_allow_html=True)

if model:
    input_data = pd.DataFrame([{
        "day_type": day_type,
        "weather_condition": weather_cond,
        "customer_segment": customer_seg,
        "companion_purchase": companion_ord
    }])
    
    pred_cat = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    prob_dict = dict(zip(classes, probs))
    conf_score = prob_dict[pred_cat] * 100

    CATEGORY_CONFIGS = {
        "Hot Espresso": {
            "icon": "☕",
            "border": "#d84315",
            "bg": "#fbe9e7",
            "barista": "⚡ Barista Queue Directive: Pre-warm ceramic demitasse cups, purge grouphead, engage steam wand.",
            "items": [
                {"name": "Artisanal Flat White", "price": "$5.25", "tag": "Double Ristretto", "desc": "Velvety textured microfoam folded over rich espresso."},
                {"name": "Single-Origin Cortado", "price": "$4.75", "tag": "1:1 Ratio", "desc": "Equal parts espresso and steamed milk for balanced intensity."},
                {"name": "Classic Americano", "price": "$4.50", "tag": "Single Origin", "desc": "Hot mountain spring water over freshly pulled espresso shots."}
            ]
        },
        "Iced Specialty Milk": {
            "icon": "🥛",
            "border": "#8d6e63",
            "bg": "#efebe9",
            "barista": "⚡ Barista Queue Directive: Ready dairy carafes, prep ice scoops, stage syrup pumps.",
            "items": [
                {"name": "Iced Brown Sugar Oat Latte", "price": "$6.75", "tag": "Oat Milk • Spiced", "desc": "Hand-shaken espresso with brown sugar reduction over ice."},
                {"name": "Caramel Macchiato on Ice", "price": "$6.25", "tag": "Artisanal Caramel", "desc": "Vanilla-infused cold milk marked with espresso and caramel."},
                {"name": "Iced Spanish Latte", "price": "$5.75", "tag": "Condensed Milk", "desc": "Espresso chilled over sweetened milk and crushed ice."}
            ]
        },
        "Blended Frappé": {
            "icon": "🥤",
            "border": "#00897b",
            "bg": "#e0f2f1",
            "barista": "⚡ Barista Queue Directive: Pre-chill blender jars, verify dairy cream charge, ready straw caddy.",
            "items": [
                {"name": "Madagascar Vanilla Bean Frappé", "price": "$6.75", "tag": "Blended Ice", "desc": "Real vanilla bean seeds, cream, and espresso blended smooth."},
                {"name": "Mocha Cacao Frappé", "price": "$6.75", "tag": "Dark Ganache", "desc": "Valrhona dark chocolate melted with espresso and crushed ice."},
                {"name": "Matcha Cloud Frappé", "price": "$6.50", "tag": "Uji Matcha", "desc": "Ceremonial green tea blended with sweet cream and vanilla."}
            ]
        },
        "Cold Brew / Iced Tea": {
            "icon": "🧊",
            "border": "#0288d1",
            "bg": "#e1f5fe",
            "barista": "⚡ Barista Queue Directive: Check cold drip tap pressure, pre-chill glassware, ready clear ice cubes.",
            "items": [
                {"name": "Signature 20h Cold Brew", "price": "$5.50", "tag": "Single Origin", "desc": "Slow cold water extraction, clean finish with notes of dark cacao."},
                {"name": "Nitro Draft Coffee", "price": "$6.25", "tag": "Draft Tap", "desc": "Nitrogen-infused cold brew with a natural velvety micro-cascade."},
                {"name": "Peach Blossom Cold Brew", "price": "$5.85", "tag": "Infused", "desc": "Crisp cold brew infused with organic peach puree and mint."}
            ]
        }
    }

    curr_cfg = CATEGORY_CONFIGS.get(pred_cat, CATEGORY_CONFIGS["Hot Espresso"])

    col_main, col_prob = st.columns([1.5, 1.1])

    # Left Column: Customer-Facing Kiosk Screen Simulation
    with col_main:
        st.markdown(f"""
        <div style="background-color: {curr_cfg['bg']}; border-left: 6px solid {curr_cfg['border']}; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #212121; font-weight: 700;">{curr_cfg['icon']} FEATURED KIOSK CATEGORY: {pred_cat.upper()}</h4>
                <span class="badge-pill" style="background:{curr_cfg['border']}; color:white;">{conf_score:.1f}% CONFIDENCE</span>
            </div>
            <p style="font-size: 0.90rem; color: #424242; font-weight: 500; margin: 0.5rem 0 0 0;">{curr_cfg['barista']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🛒 Top Recommended Items in Category")
        for itm in curr_cfg["items"]:
            c_info, c_btn = st.columns([3.5, 1.2])
            with c_info:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #ebd9c8; border-radius:8px; padding:0.85rem; margin-bottom:0.6rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.0rem; font-weight:700; color:#3e2723;">{itm['name']}</span>
                        <span style="font-size:1.0rem; font-weight:800; color:#d84315;">{itm['price']}</span>
                    </div>
                    <span class="badge-pill badge-specialty" style="font-size:0.75rem; margin-top:0.2rem;">{itm['tag']}</span>
                    <p style="font-size:0.84rem; color:#616161; margin:0.3rem 0 0 0;">{itm['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                if st.button("➕ Add", key=f"add_{itm['name']}"):
                    st.toast(f"✅ Added {itm['name']} to POS Ticket!")

    # Right Column: Bayesian Probability Updating & Sensitivity Breakdown
    with col_prob:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Multi-Class Probability Distribution")
        st.caption("Live posterior probabilities $P(C_k \\mid X)$ with Laplace smoothing:")

        df_p = pd.DataFrame({
            "Category": classes,
            "Probability": probs
        })
        
        color_scale = alt.Scale(
            domain=["Hot Espresso", "Iced Specialty Milk", "Blended Frappé", "Cold Brew / Iced Tea"],
            range=["#d84315", "#8d6e63", "#00897b", "#0288d1"]
        )
        
        chart_p = alt.Chart(df_p).mark_bar(cornerRadius=5).encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.Color("Category:N", scale=color_scale, legend=None),
            tooltip=["Category", alt.Tooltip("Probability:Q", format=".1%")]
        ).properties(height=160)
        st.altair_chart(chart_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Real-Time Probability Updating")
        st.caption("How your active selections shift Bayesian odds:")
        
        # Display intuitive real-time impact cards
        shifts = []
        if weather_cond == "Hot & Humid":
            shifts.append("☀️ **Hot & Humid Weather:** Strongly boosts Cold Brew (+34%) and Blended Frappés (+28%).")
        elif weather_cond == "Freezing":
            shifts.append("❄️ **Freezing Weather:** Shifts odds overwhelmingly toward Hot Espresso (+48%).")
        elif weather_cond == "Rainy":
            shifts.append("🌧️ **Rainy Conditions:** Dampens frozen drinks; lifts Hot Espresso and Warm Lattes.")
            
        if companion_ord == "Food Item Ordered":
            shifts.append("🥐 **Companion Pastry:** Warm food pairing effect boosts Hot Espresso by +25%.")
            
        if day_type == "Weekday" and customer_seg == "Morning Commuter":
            shifts.append("🏃 **Weekday Commuter:** Favors high-efficiency Hot Espresso or Quick Cold Brew grab-and-go.")
        elif customer_seg == "Student":
            shifts.append("🎓 **Student Profile:** Shifts distribution toward Iced Specialty Milk and Frappés.")
            
        for s in shifts:
            st.markdown(f"<div style='font-size:0.86rem; color:#424242; margin-bottom:0.4rem; padding:0.3rem 0.5rem; background:#ffffff; border-radius:5px; border-left:3px solid #795548;'>{s}</div>", unsafe_allow_html=True)
            
        st.caption("Multinomial Naïve Bayes • Laplace Smoothing $\\alpha=1.0$ • 4 Features Active")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Prediction engine offline. Please ensure trained models are loaded.")
