import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_dataset

st.set_page_config(
    page_title="RoastLab FlavorMatch™ • Drink Concierge",
    page_icon="☕",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

# Top Header / Consumer Concierge Branding
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ebd9c8; padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
    <div>
        <h2 class='brand-title' style='margin:0;'>☕ RoastLab FlavorMatch™ Concierge</h2>
        <span style='color: #795548; font-size: 0.95rem; font-weight: 500;'>Guest-Facing Beverage Personalization & Taste Profile Discovery</span>
    </div>
    <div style="text-align: right;">
        <span class="badge-pill badge-active">🟢 CONCIERGE ONLINE</span>
        <span class="badge-pill badge-specialty">🎯 14 SIGNATURE RECIPES</span>
    </div>
</div>
""", unsafe_allow_html=True)

df_menu = load_dataset("drinks_menu")

# Taste Calibration Controls
st.markdown("##### 🎛️ Calibrate Your Taste Profile")
col_s1, col_s2 = st.columns(2)

with col_s1:
    sweetness_input = st.slider(
        "Sweetness Balance (1.0 = Pure Black / Unsweetened  ➔  10.0 = Confectionery Sweet)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5
    )

with col_s2:
    caffeine_input = st.slider(
        "Caffeine Intensity (1.0 = Gentle Decaf  ➔  10.0 = Triple Ristretto Kick)",
        min_value=1.0,
        max_value=10.0,
        value=7.5,
        step=0.5
    )

st.markdown("<br>", unsafe_allow_html=True)

if df_menu is not None:
    # Compute Euclidean Proximity
    df_calc = df_menu.copy()
    df_calc["distance"] = np.sqrt(
        (df_calc["sweetness"] - sweetness_input)**2 + 
        (df_calc["caffeine"] - caffeine_input)**2
    )
    df_ranked = df_calc.sort_values("distance").reset_index(drop=True)
    
    top_pick = df_ranked.iloc[0]
    second_pick = df_ranked.iloc[1]
    third_pick = df_ranked.iloc[2]
    
    col_cards, col_radar = st.columns([1.4, 1.1])
    
    # Left Column: Curated Recommendation Cards
    with col_cards:
        st.markdown("##### 🏆 Recommended Beverages for Your Profile")
        
        # Best Match Primary Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fdfbf7 0%, #f7efe6 100%); border: 2px solid #bcaaa4; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span class="badge-pill" style="background:#2e7d32; color:white; font-size:0.8rem;">⭐ BEST FLAVOR MATCH</span>
                <span style="font-weight: 800; font-size: 1.2rem; color: #d84315;">$5.75</span>
            </div>
            <h3 style="margin: 0.2rem 0; color: #3e2723; font-weight: 800;">{top_pick['drink_name']}</h3>
            <span class="badge-pill badge-specialty">{top_pick['category']}</span>
            <span style="color: #616161; font-size: 0.88rem; font-weight: 500;">Match Proximity: {top_pick['distance']:.2f}</span>
            <p style="margin: 0.7rem 0 0.4rem 0; font-size: 0.94rem; color: #424242;">{top_pick['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.5, 2.5])
        with c_act1:
            if st.button("🛒 Order Best Match", key="btn_order_best"):
                st.toast(f"🎉 Order placed for {top_pick['drink_name']}!")
        with c_act2:
            st.caption("Customizations: Oat/Almond Milk available • Single Origin Espresso")
            
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #ebd9c8;'>", unsafe_allow_html=True)
        st.markdown("###### Alternative Pairings")
        
        col_alt1, col_alt2 = st.columns(2)
        with col_alt1:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #ebd9c8; border-radius:8px; padding:0.9rem;">
                <span style="font-size:0.78rem; font-weight:700; color:#795548;">RUNNER UP #2</span>
                <div style="font-size:1.0rem; font-weight:700; color:#3e2723; margin:0.2rem 0;">{second_pick['drink_name']}</div>
                <span class="badge-pill badge-specialty" style="font-size:0.75rem;">{second_pick['category']}</span>
                <p style="font-size:0.82rem; color:#616161; margin:0.4rem 0 0 0;">{second_pick['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select #2", key="btn_alt2"):
                st.toast(f"Selected {second_pick['drink_name']}!")
                
        with col_alt2:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #ebd9c8; border-radius:8px; padding:0.9rem;">
                <span style="font-size:0.78rem; font-weight:700; color:#795548;">RUNNER UP #3</span>
                <div style="font-size:1.0rem; font-weight:700; color:#3e2723; margin:0.2rem 0;">{third_pick['drink_name']}</div>
                <span class="badge-pill badge-specialty" style="font-size:0.75rem;">{third_pick['category']}</span>
                <p style="font-size:0.82rem; color:#616161; margin:0.4rem 0 0 0;">{third_pick['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select #3", key="btn_alt3"):
                st.toast(f"Selected {third_pick['drink_name']}!")

    # Right Column: Interactive 2D Taste Radar Space
    with col_radar:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ Menu Taste Map")
        st.caption("Live spatial proximity to catalog recipes:")
        
        base_chart = alt.Chart(df_menu).mark_circle(size=140).encode(
            x=alt.X("sweetness:Q", scale=alt.Scale(domain=[0, 10.5]), title="Sweetness Level (1 - 10)"),
            y=alt.Y("caffeine:Q", scale=alt.Scale(domain=[0, 10.5]), title="Caffeine Intensity (1 - 10)"),
            color=alt.Color("category:N", title="Beverage Category", legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=["drink_name", "category", "sweetness", "caffeine"]
        )
        labels = alt.Chart(df_menu).mark_text(dy=-12, fontSize=10, color="#5d4037").encode(
            x="sweetness:Q",
            y="caffeine:Q",
            text="drink_name:N"
        )
        user_marker = alt.Chart(pd.DataFrame({
            "sweetness": [sweetness_input],
            "caffeine": [caffeine_input],
            "label": ["★ Current Selection"]
        })).mark_point(color="#d84315", size=320, shape="star", filled=True).encode(
            x="sweetness:Q",
            y="caffeine:Q"
        )
        user_label = alt.Chart(pd.DataFrame({
            "sweetness": [sweetness_input],
            "caffeine": [caffeine_input],
            "label": ["★ Your Taste"]
        })).mark_text(dy=18, fontWeight="bold", color="#d84315", fontSize=11).encode(
            x="sweetness:Q",
            y="caffeine:Q",
            text="label:N"
        )
        
        chart = (base_chart + labels + user_marker + user_label).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Retail Performance KPIs
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Guest Satisfaction", "4.92 / 5.0", delta="+0.38 vs static menu")
    with k2:
        st.metric("Menu Discovery", "94.6%", delta="Repeat order diversity")
    with k3:
        st.metric("Order Ticket Avg", "$6.10", delta="+$0.85 per transaction")
    with k4:
        st.metric("Engine Match Latency", "12 ms", delta="k=3 Spatial Search")

else:
    st.warning("Catalog database unavailable. Please check system data connectivity.")
