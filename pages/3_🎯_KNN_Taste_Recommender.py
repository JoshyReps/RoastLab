import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_dataset, load_all_metrics

st.set_page_config(
    page_title="3. KNN Recommender | RoastLab",
    page_icon="🎯",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>3. K-Nearest Neighbors: 5D Sensory Drink Recommender</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Multi-Dimensional Min-Max Normalization & Normalized Euclidean Distance Neighbor Search</p>", unsafe_allow_html=True)

df_menu = load_dataset("drinks_menu")
metrics = load_all_metrics().get("knn", {})

FEATURE_COLS = ["sweetness", "caffeine", "bitterness", "creaminess", "acidity"]
FEATURE_LABELS = {
    "sweetness": "Sweetness",
    "caffeine": "Caffeine Strength",
    "bitterness": "Bitterness / Roast",
    "creaminess": "Creaminess / Body",
    "acidity": "Acidity / Brightness"
}

col_desc, col_inputs = st.columns([1, 1])
with col_desc:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Artisanal cafe menus feature dozens of complex espresso, cold brew, and specialty beverages. Guests often find it difficult 
    to translate their cravings into beverage names (e.g. flat white vs. cortado vs. breve), but can effortlessly rate their desired sensory preferences: 
    sweetness, caffeine intensity, roast bitterness, dairy creaminess, and citrus acidity.
    
    **Why K-Nearest Neighbors (KNN with Min-Max Normalization)?**
    - Non-parametric, instance-based learning algorithm that does not assume linear boundaries or rigid decision trees.
    - Matches a guest's multi-attribute palate against RoastLab's catalog of handcrafted signature drinks by locating the $k$ closest neighbors in 5-dimensional taste space.
    """)
    
    st.markdown(r"""
    **Mathematical Formulation:**  
    To ensure attributes with large numerical scales do not distort distance calculations, all features are normalized to $[0, 1]$:
    $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} = \frac{x - 1.0}{9.0}$$
    
    Proximity between the guest preference vector $\mathbf{P}$ and menu recipe $\mathbf{Q}$ is computed via **Normalized Euclidean Distance**:
    $$d(\mathbf{P}, \mathbf{Q}) = \sqrt{\sum_{i=1}^{5} (p_{i,\text{norm}} - q_{i,\text{norm}})^2}$$
    """)
    
    st.markdown(f"""
    - **Distance Metric:** `Normalized Euclidean Distance in 5D Space`
    - **Feature Scaling:** `Min-Max Normalization to [0.0, 1.0]`
    - **Sensory Dimensions:** `5 Continuous Variables (Scale 1.0 - 10.0)`
    - **Drink Menu Catalog:** `{metrics.get('menu_items_count', 14)} Handcrafted Signature Recipes`
    - **Historical Taste Observations:** `{metrics.get('samples', 420)} Calibrated Cupping Profiles`
    - **Neighborhood Size:** `Dynamic k (1 <= k <= 7)`
    """)

with col_inputs:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Sensory Taste Profile Calibration")
    st.caption("Adjust the sliders below to calibrate your taste preferences (1.0 to 10.0):")
    
    c1, c2 = st.columns(2)
    with c1:
        sweetness_in = st.slider("🍬 Sweetness Level", 1.0, 10.0, 3.5, 0.5, help="1.0 Unsweetened / Black -> 10.0 Confectionery Sweet")
        caffeine_in = st.slider("⚡ Caffeine Intensity", 1.0, 10.0, 7.5, 0.5, help="1.0 Gentle Decaf -> 10.0 High-Performance Triple Shot")
        bitterness_in = st.slider("🔥 Bitterness / Roast Depth", 1.0, 10.0, 5.0, 0.5, help="1.0 Mild & Delicate -> 10.0 Intense French Dark Roast")
    with c2:
        creaminess_in = st.slider("🥛 Creaminess / Dairy Body", 1.0, 10.0, 7.0, 0.5, help="1.0 Thin / Black / Tea-like -> 10.0 Rich Textured Microfoam")
        acidity_in = st.slider("🍋 Acidity / Citrus Brightness", 1.0, 10.0, 4.0, 0.5, help="1.0 Mellow Low Acid -> 10.0 Vivid Crisp Citrus Notes")
        k_val = st.slider("🎯 Nearest Matches (k)", 1, 7, 3, 1, help="Adjust k to expand or contract nearest-neighbor cluster")
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🏆 Top Nearest Neighbor Matches & 2D Projection Radar")

if df_menu is not None:
    # 1. Min-Max Normalization: x_norm = (x - 1.0) / (10.0 - 1.0)
    user_raw = np.array([sweetness_in, caffeine_in, bitterness_in, creaminess_in, acidity_in])
    user_norm = (user_raw - 1.0) / 9.0

    df_calc = df_menu.copy()
    menu_matrix_raw = df_calc[FEATURE_COLS].values
    menu_matrix_norm = (menu_matrix_raw - 1.0) / 9.0

    # 2. Multi-Dimensional Normalized Euclidean Distance
    diffs = menu_matrix_norm - user_norm
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    df_calc["norm_distance"] = dists
    
    # Match percentage score
    df_calc["match_score"] = np.clip(100.0 - (dists * 35.0), 45.0, 99.5)
    df_ranked = df_calc.sort_values("norm_distance").reset_index(drop=True)

    col_results, col_viz = st.columns([1.2, 1.3])

    # Left Column: Top-K Recommended Drinks
    with col_results:
        st.markdown(f"##### 🏆 Top {k_val} Recommended Drinks (Ranked by Proximity)")
        
        for i in range(min(k_val, len(df_ranked))):
            row = df_ranked.iloc[i]
            is_best = (i == 0)
            
            card_border = "#2e7d32" if is_best else "#d7ccc8"
            card_bg = "#fbfaf8" if is_best else "#ffffff"
            rank_badge = "⭐ #1 BEST MATCH" if is_best else f"#{i+1} MATCH"
            badge_color = "#2e7d32" if is_best else "#795548"
            
            st.markdown(f"""
            <div style="background:{card_bg}; border: 1.5px solid {card_border}; border-radius: 10px; padding: 0.95rem; margin-bottom: 0.8rem; box-shadow: 0 2px 5px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge-pill" style="background:{badge_color}; color:white; font-size:0.75rem;">{rank_badge}</span>
                    <span style="font-weight:700; color:#d84315; font-size:1.05rem;">{row.get('price', '$5.50')}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin:0.3rem 0;">
                    <span style="font-size:1.1rem; font-weight:800; color:#3e2723;">{row['drink_name']}</span>
                    <span class="badge-pill badge-specialty" style="font-size:0.75rem;">{row['category']}</span>
                </div>
                <div style="display:flex; gap:12px; font-size:0.82rem; color:#616161; margin-bottom:0.4rem;">
                    <span>Alignment: <strong style="color:#2e7d32;">{row['match_score']:.1f}%</strong></span>
                    <span>Normalized Distance: <strong>{row['norm_distance']:.3f}</strong></span>
                </div>
                <p style="font-size:0.85rem; color:#424242; margin:0.2rem 0 0.5rem 0;">{row['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            c_ord, c_sp = st.columns([1.5, 3])
            with c_ord:
                if st.button(f"🛒 Order {row['drink_name'].split()[0]}", key=f"btn_k_{i}"):
                    st.toast(f"✅ Added {row['drink_name']} to Order!")

    # Right Column: Interactive 2D Projection with Axis Toggles
    with col_viz:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ 2D Flavor Radar Projection")
        st.caption("Toggle any two axes to explore the multi-dimensional space:")
        
        c_ax1, c_ax2 = st.columns(2)
        with c_ax1:
            x_axis = st.selectbox(
                "Horizontal (X-Axis)",
                FEATURE_COLS,
                index=0,
                format_func=lambda col: FEATURE_LABELS[col]
            )
        with c_ax2:
            y_axis = st.selectbox(
                "Vertical (Y-Axis)",
                FEATURE_COLS,
                index=3,
                format_func=lambda col: FEATURE_LABELS[col]
            )
            
        # Highlight Top-K items in dataframe
        top_k_names = set(df_ranked.head(k_val)["drink_name"])
        df_plot = df_menu.copy()
        df_plot["is_in_top_k"] = df_plot["drink_name"].apply(lambda n: "In Top-k" if n in top_k_names else "Other")
        
        # Base Scatter of Catalog
        scatter = alt.Chart(df_plot).mark_circle().encode(
            x=alt.X(f"{x_axis}:Q", scale=alt.Scale(domain=[0.5, 10.5]), title=FEATURE_LABELS[x_axis]),
            y=alt.Y(f"{y_axis}:Q", scale=alt.Scale(domain=[0.5, 10.5]), title=FEATURE_LABELS[y_axis]),
            size=alt.condition(alt.datum.is_in_top_k == "In Top-k", alt.value(220), alt.value(90)),
            color=alt.Color("category:N", title="Beverage Category", legend=alt.Legend(orient="bottom", columns=2)),
            stroke=alt.condition(alt.datum.is_in_top_k == "In Top-k", alt.value("#2e7d32"), alt.value("transparent")),
            strokeWidth=alt.value(2.5),
            tooltip=["drink_name", "category", x_axis, y_axis, "desc"]
        )
        
        # Labels for Top-K
        top_k_subset = df_plot[df_plot["is_in_top_k"] == "In Top-k"]
        labels = alt.Chart(top_k_subset).mark_text(dy=-13, fontSize=10, fontWeight="bold", color="#3e2723").encode(
            x=f"{x_axis}:Q",
            y=f"{y_axis}:Q",
            text="drink_name:N"
        )
        
        # User Coordinates Marker
        user_coords = {
            "sweetness": sweetness_in,
            "caffeine": caffeine_in,
            "bitterness": bitterness_in,
            "creaminess": creaminess_in,
            "acidity": acidity_in
        }
        df_user = pd.DataFrame([{
            x_axis: user_coords[x_axis],
            y_axis: user_coords[y_axis],
            "label": "★ Your Taste Selection"
        }])
        
        user_star = alt.Chart(df_user).mark_point(
            color="#d84315",
            size=360,
            shape="star",
            filled=True
        ).encode(
            x=f"{x_axis}:Q",
            y=f"{y_axis}:Q"
        )
        
        user_txt = alt.Chart(df_user).mark_text(
            dy=18,
            fontWeight="bold",
            color="#d84315",
            fontSize=11
        ).encode(
            x=f"{x_axis}:Q",
            y=f"{y_axis}:Q",
            text="label:N"
        )
        
        chart = (scatter + labels + user_star + user_txt).properties(height=380)
        st.altair_chart(chart, use_container_width=True)
        
        st.caption(f"Green-bordered circles represent your current **Top {k_val}** nearest neighbors in 5D space.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Catalog database unavailable. Please check system data connectivity.")
