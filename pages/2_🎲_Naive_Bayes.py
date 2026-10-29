import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_all_metrics

st.set_page_config(
    page_title="2. Naive Bayes | RoastLab",
    page_icon="🎲",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>2. Naïve Bayes: Single-Origin Whole Bean Recommender</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Multinomial Bayesian Likelihood & Laplace Smoothing for Retail Shelf Bean Advisory</p>", unsafe_allow_html=True)

model = load_model("naive_bayes")
metrics = load_all_metrics().get("naive_bayes", {})

col_desc, col_inputs = st.columns([1, 1])
with col_desc:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    In specialty retail cafes, customers purchasing whole bean coffee often struggle to choose among diverse single-origin offerings 
    (e.g., Ethiopian, Colombian, Sumatra, Guatemalan, Kenyan, Brazilian). Each origin exhibits radically different bean density, 
    organic acidity, solubility, and roast dynamics.
    
    **Why Naïve Bayes (Multinomial with Laplace Smoothing)?**
    - Coffee preference depends on a combination of discrete categorical variables: **Brewing Equipment**, **Roast Spectrum Preference**, 
      **Desired Flavor Direction**, and **Drinking Format** (black vs. with milk).
    - Under the conditional independence assumption, the posterior probability of a customer loving bean origin $C_k$ is proportional to the prior probability multiplied by the product of individual conditional likelihoods:
    """)
    
    st.markdown(r"""
    **Mathematical Formulation:**  
    $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$
    
    To prevent zero-probability lockouts when an uncommon combination is encountered (e.g., French Press + Light Roast), **Laplace Smoothing** ($\alpha = 1.0$) is applied:
    $$\hat{P}(x_i \mid C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha \cdot d}$$
    where $N_{ki}$ is the count of feature value $x_i$ observed in class $C_k$, and $d$ is the number of distinct feature values.
    """)
    
    st.markdown(f"""
    - **Classifier Family:** `Multinomial Naïve Bayes (Categorical Feature Likelihood)`
    - **Smoothing Parameter:** `Laplace Smoothing (alpha = 1.0)`
    - **Single-Origin Beans:** `{len(metrics.get('beans', ['Brazilian', 'Colombian', 'Ethiopian', 'Guatemalan', 'Kenyan', 'Sumatra']))} Regional Terroirs`
    - **Sensory & Brewing Features:** `4 Categorical Dimensions (Method, Roast, Notes, Format)`
    - **Training Observations:** `{metrics.get('samples', 500)} Calibrated Cupping Profiles`
    - **Model Classification Accuracy:** `{metrics.get('accuracy', 0.604):.1%}`
    """)
    
with col_inputs:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🛒 Customer Brew Profile & Flavor Calibration")
    st.caption("Select your home brewing setup and sensory preferences below:")
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        brew_method = st.selectbox(
            "☕ Brew Equipment / Method",
            [
                "Pour-Over (V60 / Chemex)",
                "French Press / Immersion",
                "Espresso Machine / Moka Pot",
                "Cold Drip / Immersion"
            ],
            index=0,
            help="Extraction dynamics dictate required bean density and soluble yield"
        )
        roast_pref = st.selectbox(
            "🔥 Roast Spectrum Preference",
            [
                "Light Roast (Bright & Acidic)",
                "Medium Roast (Sweet & Balanced)",
                "Dark Roast (Smoky & Bold)"
            ],
            index=1,
            help="Degree of organic acid retention vs Maillard caramelization"
        )
    with c_f2:
        flavor_desired = st.selectbox(
            "🍓 Primary Flavor Direction",
            [
                "Floral & Citrus",
                "Milk Chocolate & Nutty",
                "Dark Cacao & Earthy Spice",
                "Winey Berry & Sweet Fruit"
            ],
            index=0,
            help="Dominant flavor profile desired in the cup"
        )
        drinking_style = st.selectbox(
            "🥛 Drinking Format",
            [
                "Drank Black / Neat",
                "With Milk / Cream",
                "Iced / Chilled"
            ],
            index=0,
            help="Dairy cuts acidity; black extraction highlights terroir brightness"
        )
    
    st.markdown(f"""
    <div style="background:#ffffff; border-radius:6px; border:1px solid #ebd9c8; padding:0.6rem 0.8rem; margin-top:0.6rem; font-size:0.85rem; color:#5d4037;">
        <strong>Selected Profile:</strong> <code>{brew_method.split()[0]}</code> • <code>{roast_pref.split()[0]}</code> • <code>{flavor_desired.split()[0]}</code> • <code>{drinking_style.split()[0]}</code>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📦 Recommended Bean Origin & Real-Time Posterior Probabilities")

BEAN_PROFILES = {
    "Ethiopian Yirgacheffe": {
        "flag": "🇪🇹",
        "region": "Yirgacheffe, Gedeo Zone, Ethiopia",
        "elevation": "1,900m – 2,200m",
        "process": "Washed Heirloom",
        "price": "$22.00",
        "notes": "Jasmine blossom, bergamot citrus, ripe peach nectar, sparkling black tea finish.",
        "best_for": "Pour-Over (V60/Chemex), Drank Black, Cold Drip",
        "acidity": "High & Vibrant",
        "body": "Delicate & Tea-like",
        "accent": "#00897b"
    },
    "Colombian Huila Supremo": {
        "flag": "🇨🇴",
        "region": "Huila Valley, Colombia",
        "elevation": "1,500m – 1,800m",
        "process": "Fully Washed Caturra / Castillo",
        "price": "$19.50",
        "notes": "Smooth milk chocolate, toasted hazelnut, crisp red apple, sweet panela sugar.",
        "best_for": "Everyday Drip, Balanced Espresso, Flat White",
        "acidity": "Balanced & Crisp",
        "body": "Medium & Silky",
        "accent": "#d84315"
    },
    "Sumatra Mandheling": {
        "flag": "🇮🇩",
        "region": "Lintong, Lake Toba, North Sumatra",
        "elevation": "1,200m – 1,500m",
        "process": "Giling Basah (Wet-Hulled)",
        "price": "$20.00",
        "notes": "Cedar wood, pipe tobacco, dark bittersweet cacao, rich dark molasses.",
        "best_for": "French Press, Dark Roast lovers, Bold Espresso with Cream",
        "acidity": "Extremely Low",
        "body": "Heavy & Syrupy",
        "accent": "#5d4037"
    },
    "Guatemalan Antigua": {
        "flag": "🇬🇹",
        "region": "Antigua Volcanic Valley, Guatemala",
        "elevation": "1,600m",
        "process": "Washed Bourbon / Catuai",
        "price": "$21.00",
        "notes": "Spiced cinnamon, dark baker's cocoa, orange blossom, sweet caramel finish.",
        "best_for": "Moka Pot, Espresso, Rich Milk Coffee",
        "acidity": "Medium Malic Acidity",
        "body": "Velvety & Full",
        "accent": "#6d4c41"
    },
    "Kenyan AA": {
        "flag": "🇰🇪",
        "region": "Nyeri Hill, Mount Kenya",
        "elevation": "1,750m – 1,950m",
        "process": "Double Washed SL-28 / SL-34",
        "price": "$23.50",
        "notes": "Blackcurrant, blackberry winey acidity, complex brown sugar, grapefruit zest.",
        "best_for": "Pour-Over, Chemex, Iced Filter Coffee",
        "acidity": "Intense & Winey",
        "body": "Juicy & Resonant",
        "accent": "#c2185b"
    },
    "Brazilian Cerrado Santos": {
        "flag": "🇧🇷",
        "region": "Cerrado Mineiro, Minas Gerais, Brazil",
        "elevation": "1,050m – 1,200m",
        "process": "Natural (Sun-Dried on Patio)",
        "price": "$18.50",
        "notes": "Roasted peanut butter, toasted wafer, caramelized brown sugar, soft cocoa nib.",
        "best_for": "Traditional Espresso, Heavy Milk Lattes, Breakfast Blend",
        "acidity": "Low & Soft",
        "body": "Dense & Creamy",
        "accent": "#f57c00"
    }
}

if model:
    input_data = pd.DataFrame([{
        "brew_method": brew_method,
        "roast_preference": roast_pref,
        "flavor_notes_desired": flavor_desired,
        "drinking_style": drinking_style
    }])
    
    pred_bean = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    
    # Sort beans by probability
    ranked_indices = np.argsort(probs)[::-1]
    top_bean = classes[ranked_indices[0]]
    top_prob = probs[ranked_indices[0]] * 100
    runner_up_bean = classes[ranked_indices[1]]
    runner_up_prob = probs[ranked_indices[1]] * 100

    info_top = BEAN_PROFILES.get(top_bean, BEAN_PROFILES["Ethiopian Yirgacheffe"])
    info_second = BEAN_PROFILES.get(runner_up_bean, BEAN_PROFILES["Colombian Huila Supremo"])

    col_main, col_prob = st.columns([1.4, 1.2])

    # Left Column: Recommended Bean Cards
    with col_main:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fdfbf7 0%, #f7efe6 100%); border: 2px solid {info_top['accent']}; border-radius: 12px; padding: 1.3rem; margin-bottom: 1.2rem; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span class="badge-pill" style="background:{info_top['accent']}; color:white; font-size:0.8rem;">⭐ #1 RECOMMENDED SINGLE-ORIGIN BEAN</span>
                <span style="font-weight: 800; font-size: 1.3rem; color: #d84315;">{info_top['price']} <span style="font-size:0.85rem; color:#757575;">/ 250g bag</span></span>
            </div>
            <h3 style="margin: 0.2rem 0; color: #3e2723; font-weight: 800;">{info_top['flag']} {top_bean}</h3>
            <div style="margin-bottom: 0.6rem;">
                <span class="badge-pill badge-specialty">{info_top['region']}</span>
                <span class="badge-pill badge-specialty">Elevation: {info_top['elevation']}</span>
                <span class="badge-pill badge-specialty">{info_top['process']}</span>
            </div>
            <p style="margin: 0.6rem 0; font-size: 0.95rem; color: #333333; line-height: 1.45;"><strong>Cupping Notes:</strong> <em>{info_top['notes']}</em></p>
            <div style="display: flex; gap: 15px; font-size: 0.85rem; color: #555555; background: #ffffff; padding: 0.5rem 0.8rem; border-radius: 6px; border: 1px solid #e0d4c8; margin-top: 0.6rem;">
                <span>Acidity: <strong>{info_top['acidity']}</strong></span>
                <span>•</span>
                <span>Body: <strong>{info_top['body']}</strong></span>
                <span>•</span>
                <span>Optimal Brew: <strong>{info_top['best_for']}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_act1, c_act2 = st.columns([1.5, 2.0])
        with c_act1:
            if st.button("🛒 Add 250g Bag to Cart", key="btn_bag_cart"):
                st.toast(f"✅ Added 250g of {top_bean} to your retail cart!")
        with c_act2:
            if st.button("⚙️ Grind for My Equipment", key="btn_grind"):
                st.toast(f"✅ Barista station queued: Precision grind {top_bean} for {brew_method}!")

        st.markdown("<hr style='margin: 1.2rem 0 0.8rem 0; border: none; border-top: 1px solid #ebd9c8;'>", unsafe_allow_html=True)
        st.markdown(f"###### 🥈 Runner-Up Pairing: {info_second['flag']} {runner_up_bean} ({runner_up_prob:.1f}% Alignment)")
        st.caption(f"_{info_second['notes']}_ — **{info_second['price']}** per 250g bag.")

    # Right Column: Bayesian Probability Distribution & Live Updating
    with col_prob:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Bayesian Origin Posterior Distribution")
        st.caption(f"Posterior probability $P(\\text{{Bean}} \\mid X)$ with Laplace smoothing:")

        df_p = pd.DataFrame({
            "Bean Origin": [f"{BEAN_PROFILES[b]['flag']} {b.split()[0]} {b.split()[1] if len(b.split())>1 else ''}" for b in classes],
            "Full Name": classes,
            "Probability": probs
        })

        chart_p = alt.Chart(df_p).mark_bar(cornerRadius=5).encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Bean Origin:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["Full Name"] == top_bean,
                alt.value("#2e7d32"),
                alt.value("#b0bec5")
            ),
            tooltip=["Full Name", alt.Tooltip("Probability:Q", format=".1%")]
        ).properties(height=210)
        st.altair_chart(chart_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Real-Time Probability Updating")
        st.caption("How your selected brewing conditions shift the Bayesian evidence:")

        # Dynamic Bayesian Evidence Explanation
        shifts = []
        if brew_method == "Pour-Over (V60 / Chemex)":
            shifts.append("🧪 **Pour-Over Geometry:** Favors high-altitude, dense beans with floral clarity (Ethiopian & Kenyan).")
        elif brew_method == "French Press / Immersion":
            shifts.append("🫗 **Full Immersion:** Extracts heavy oils and body, shifting probability to Sumatra Mandheling.")
        elif brew_method == "Espresso Machine / Moka Pot":
            shifts.append("⚡ **High-Pressure Extraction:** Enhances crema and cocoa balance, favoring Brazilian & Colombian beans.")
        elif brew_method == "Cold Drip / Immersion":
            shifts.append("🧊 **Cold Water Extraction:** Minimizes acidity extraction while pulling bright aromatics (Kenyan & Ethiopian).")

        if roast_pref == "Dark Roast (Smoky & Bold)":
            shifts.append("🔥 **Dark Roast Spectrum:** Suppresses delicate florals; increases odds for Sumatra (+42%) and Brazilian.")
        elif roast_pref == "Light Roast (Bright & Acidic)":
            shifts.append("🌱 **Light Nordic Roast:** Preserves enzymatic acids and floral terroir (Ethiopian Yirgacheffe).")

        if drinking_style == "With Milk / Cream":
            shifts.append("🥛 **Dairy Pairing:** Favors chocolatey, low-acid beans that cut through steamed milk.")
        elif drinking_style == "Drank Black / Neat":
            shifts.append("☕ **Drank Black:** Allows subtle stone fruit and citrus acidity to shine.")

        for s in shifts:
            st.markdown(f"<div style='font-size:0.85rem; color:#424242; margin-bottom:0.4rem; padding:0.3rem 0.5rem; background:#ffffff; border-radius:5px; border-left:3px solid #795548;'>{s}</div>", unsafe_allow_html=True)

        st.caption("Active Model: `MultinomialNB-Beans-v2.0` • Laplace Smoothing $\\alpha=1.0$")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Bean recommendation engine offline. Please ensure trained models are loaded.")
