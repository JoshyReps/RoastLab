import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_all_metrics

st.set_page_config(
    page_title="5. Decision Tree | RoastLab",
    page_icon="🌳",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>5. Decision Tree (ID3): Single-Origin Bean Matcher</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Information Gain & Entropy-Driven Bean Origin Recommendation</p>", unsafe_allow_html=True)

model = load_model("decision_tree")
metrics = load_all_metrics().get("decision_tree", {})

col_dt1, col_dt2 = st.columns([1, 1])
with col_dt1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Single-origin coffees have distinctive regional terroir (e.g. floral Ethiopian heirloom vs. earthy Sumatra). 
    However, everyday guests often feel intimidated by cupping vocabulary.
    
    **Why Decision Tree (Entropy / ID3)?**
    - Uses **Information Gain** to identify which taste attribute splits guest preference most decisively:
      $$\\text{Entropy}(S) = -\\sum_{i=1}^c p_i \\log_2(p_i)$$
      $$\\text{Gain}(S, A) = \\text{Entropy}(S) - \\sum_{v \\in \\text{Values}(A)} \\frac{|S_v|}{|S|} \\text{Entropy}(S_v)$$
    - Follows an interpretable hierarchical flowchart that baristas can explain naturally in 3 questions.
    """)
    st.markdown(f"""
    - **Splitting Criterion:** `Entropy (Information Gain)`
    - **Tree Depth:** `{metrics.get('tree_depth', 3)} levels`
    - **Terminal Leaves:** `{metrics.get('leaves_count', 6)} bean categories`
    - **Classification Accuracy:** `{metrics.get('accuracy', 1.0):.1%}`
    """)
    
with col_dt2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### ☕ 3-Question Bean Selector Quiz")
    q_body = st.selectbox("1. What body texture do you prefer in your cup?", ["Light/Tea-like", "Medium/Smooth", "Heavy/Syrupy"])
    q_acidity = st.selectbox("2. How vibrant should the acidity/brightness be?", ["High/Bright", "Balanced", "Low/Mild"])
    q_flavor = st.selectbox("3. What primary aroma and tasting note excites you?", ["Fruity/Floral", "Chocolatey/Nutty", "Earthy/Spicy"])
    
    if model:
        df_query = pd.DataFrame([{
            "body": q_body,
            "acidity": q_acidity,
            "flavor_profile": q_flavor
        }])
        match_bean = model.predict(df_query)[0]
        
        bean_profiles = {
            "Ethiopian Yirgacheffe": {"region": "Yirgacheffe, Ethiopia", "elevation": "1,900 - 2,200m", "notes": "Jasmine, bergamot, lemon zest, sweet peach"},
            "Kenyan AA": {"region": "Nyeri, Kenya", "elevation": "1,750 - 1,950m", "notes": "Blackcurrant, grape, winey acidity, brown sugar"},
            "Colombian Supremo": {"region": "Huila, Colombia", "elevation": "1,500 - 1,800m", "notes": "Milk chocolate, toasted hazelnut, red apple"},
            "Guatemalan Antigua": {"region": "Antigua Valley, Guatemala", "elevation": "1,600m", "notes": "Dark cocoa, cinnamon spice, orange blossom"},
            "Sumatra Mandheling": {"region": "Lintong, North Sumatra", "elevation": "1,200 - 1,500m", "notes": "Cedar, dark cocoa, pipe tobacco, molasses"},
            "Brazilian Santos": {"region": "Minas Gerais, Brazil", "elevation": "1,000 - 1,200m", "notes": "Roasted peanut, caramel, heavy creamy mouthfeel"}
        }
        
        st.success(f"🌿 **Recommended Bean Origin:** **{match_bean}**")
        info = bean_profiles.get(match_bean, {"region": "Specialty Origin", "elevation": "1,600m", "notes": "Artisanal roast notes"})
        st.markdown(f"""
        - **Growing Region:** {info['region']}
        - **Farm Elevation:** {info['elevation']}
        - **Cupping Notes:** _{info['notes']}_
        """)
    else:
        st.warning("Model file not found. Please train models first.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🌿 Featured Single-Origin Bean Catalog")
st.markdown("""
- **Ethiopian Yirgacheffe:** Floral, peach, Meyer lemon. Best prepared as a delicate Chemex pour-over.
- **Colombian Supremo:** Balanced sweetness, milk chocolate, red apple. Great as an everyday V60 brew.
- **Sumatra Mandheling:** Full-bodied, cedar wood, rich molasses. Perfect for French Press lovers.
""")
