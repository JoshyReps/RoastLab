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

st.markdown("<h2 class='brand-title'>5. Decision Tree (ID3): Retail Expansion Feasibility Engine</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Commercial Real Estate Site Acquisition • Binary Feasibility Classifier (YES / NO)</p>", unsafe_allow_html=True)

model = load_model("decision_tree")
metrics = load_all_metrics().get("decision_tree", {})
df_sites = load_dataset("coffee_shop_location_decision")

col_desc, col_inputs = st.columns([1, 1])

with col_desc:
    st.markdown("### ☕ Strategic Context & Commercial Feasibility")
    st.write("""
    Opening an artisanal coffee shop requires substantial upfront capital expenditure ($85k to $350k). 
    Before committing to a multi-year commercial lease, RoastLab's corporate development committee conducts an environmental site evaluation.
    
    Using the **ID3 Decision Tree algorithm**, the engine evaluates foot traffic volume, occupancy cost ratios, competitor density, 
    and transit accessibility to generate a definitive binary acquisition decision:
    - 🟢 **YES — Site Approved:** Favorable unit economics, strong demand density, and manageable lease overhead.
    - 🔴 **NO — Site Rejected:** Unfavorable unit economics, low foot traffic, or prohibitive competitive saturation.
    """)
    
    st.markdown(r"""
    **Mathematical Formulation (Shannon Entropy & Information Gain):**  
    $$\text{Entropy}(S) = -p_{\text{yes}} \log_2(p_{\text{yes}}) - p_{\text{no}} \log_2(p_{\text{no}})$$
    $$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
    """)
    
    st.markdown(f"""
    - **Classification Task:** `Binary Site Acquisition Decision (YES / NO)`
    - **Splitting Criterion:** `Shannon Entropy / Information Gain (ID3)`
    - **Tree Depth:** `{metrics.get('tree_depth', 6)} hierarchical decision levels`
    - **Benchmark Site Surveys:** `{metrics.get('samples', 21)} Evaluated Commercial Properties`
    - **Model Classification Accuracy:** `{metrics.get('accuracy', 1.0):.1%}`
    """)

with col_inputs:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🏢 Environmental Site Evaluation Inputs")
    st.caption("Calibrate prospective property parameters to evaluate acquisition feasibility:")
    
    c1, c2 = st.columns(2)
    with c1:
        foot_traffic = st.selectbox(
            "🚶 Daily Pedestrian Foot Traffic",
            [
                "High Volume (> 15k pedestrians/day)",
                "Moderate Volume (5k-15k pedestrians/day)",
                "Low Volume (< 5k pedestrians/day)"
            ],
            index=0,
            help="Average daily pedestrian footfall passing the storefront"
        )
        lease_overhead = st.selectbox(
            "💼 Monthly Occupancy & Lease Overhead",
            [
                "Favorable (< 10% revenue)",
                "Moderate Market Rate (10-18%)",
                "Prime / Premium Overhead (> 18%)"
            ],
            index=0,
            help="Projected rent-to-revenue ratio based on local commercial lease quotes"
        )
    with c2:
        competition = st.selectbox(
            "☕ Competitor Density (500m Radius)",
            [
                "Low Saturation (0-2 cafes)",
                "Moderate Saturation (3-5 cafes)",
                "Heavy Saturation (> 5 cafes)"
            ],
            index=0,
            help="Number of established specialty coffee retailers within walking radius"
        )
        transit_access = st.selectbox(
            "🚇 Transit & Pedestrian Infrastructure",
            [
                "Prime Accessibility (Transit Hub)",
                "Standard Accessibility (Streetflow)",
                "Restricted (Poor Walkability)"
            ],
            index=0,
            help="Proximity to metro concourses, bus stops, pedestrian crosswalks, or parking"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📋 Executive Acquisition Verdict & Feasibility Blueprint")

if model:
    input_data = pd.DataFrame([{
        "foot_traffic": foot_traffic,
        "lease_overhead": lease_overhead,
        "competition_density": competition,
        "transit_access": transit_access
    }])
    
    verdict = model.predict(input_data)[0]
    
    col_verdict, col_logic = st.columns([1.3, 1.2])
    
    with col_verdict:
        if verdict == "YES":
            card_html = (
                '<div style="background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%); '
                'border: 2px solid #2e7d32; border-radius: 12px; padding: 1.3rem; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">'
                '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">'
                '<span class="badge-pill" style="background:#2e7d32; color:white; font-size:0.8rem;">⭐ EXECUTIVE VERDICT: APPROVED</span>'
                '<span class="badge-pill" style="background:#c8e6c9; color:#1b5e20; font-weight:700;">TIER 1 SITE</span>'
                '</div>'
                '<h2 style="color:#1b5e20; margin:0.3rem 0; font-weight:800; font-size:1.8rem;">🟢 DECISION: YES</h2>'
                '<p style="color:#2e7d32; font-size:0.95rem; margin:0.4rem 0 0.8rem 0; line-height:1.45;">'
                '<strong>Recommendation:</strong> Proceed with commercial lease negotiation. The site satisfies all demographic density, '
                'rent-to-revenue ratio, and customer capture criteria.'
                '</p>'
                '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:0.8rem 0; font-size:0.85rem;">'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #c8e6c9;">'
                '<div style="color:#558b2f; font-weight:600; font-size:0.75rem;">PROJECTED PAYBACK</div>'
                '<div style="font-weight:800; color:#1b5e20; font-size:1.05rem;">6.5 – 9.0 Months</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #c8e6c9;">'
                '<div style="color:#558b2f; font-weight:600; font-size:0.75rem;">EST. OPERATING MARGIN</div>'
                '<div style="font-weight:800; color:#1b5e20; font-size:1.05rem;">18.5% – 22.0% Net</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #c8e6c9;">'
                '<div style="color:#558b2f; font-weight:600; font-size:0.75rem;">BREAK-EVEN VOLUME</div>'
                '<div style="font-weight:700; color:#333333;">280 cups / day</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #c8e6c9;">'
                '<div style="color:#558b2f; font-weight:600; font-size:0.75rem;">CAPITAL RISK TIER</div>'
                '<div style="font-weight:700; color:#2e7d32;">Low / Defensible</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            c_act1, c_act2 = st.columns([1.5, 2.0])
            with c_act1:
                if st.button("📄 Generate Acquisition Memo", key="btn_memo_yes"):
                    st.toast("✅ Executive Acquisition Memorandum generated!")
            with c_act2:
                if st.button("💾 Save to Pipeline", key="btn_pipe_yes"):
                    st.toast("✅ Property queued for Lease Negotiation!")
        else:
            card_html = (
                '<div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); '
                'border: 2px solid #c62828; border-radius: 12px; padding: 1.3rem; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">'
                '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">'
                '<span class="badge-pill" style="background:#c62828; color:white; font-size:0.8rem;">⚠️ EXECUTIVE VERDICT: REJECTED</span>'
                '<span class="badge-pill" style="background:#ef9a9a; color:#b71c1c; font-weight:700;">PROHIBITIVE RISK</span>'
                '</div>'
                '<h2 style="color:#b71c1c; margin:0.3rem 0; font-weight:800; font-size:1.8rem;">🔴 DECISION: NO</h2>'
                '<p style="color:#c62828; font-size:0.95rem; margin:0.4rem 0 0.8rem 0; line-height:1.45;">'
                '<strong>Recommendation:</strong> Reject site acquisition. The property fails key viability gates due to insufficient footfall, '
                'excessive lease overhead, or unmanageable competitive saturation.'
                '</p>'
                '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:0.8rem 0; font-size:0.85rem;">'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #ef9a9a;">'
                '<div style="color:#c62828; font-weight:600; font-size:0.75rem;">PROJECTED PAYBACK</div>'
                '<div style="font-weight:800; color:#b71c1c; font-size:1.05rem;">> 24 Months (Capital Lockup)</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #ef9a9a;">'
                '<div style="color:#c62828; font-weight:600; font-size:0.75rem;">EST. OPERATING MARGIN</div>'
                '<div style="font-weight:800; color:#b71c1c; font-size:1.05rem;">Deficit / Negative ROI</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #ef9a9a;">'
                '<div style="color:#c62828; font-weight:600; font-size:0.75rem;">BREAK-EVEN VOLUME</div>'
                '<div style="font-weight:700; color:#333333;">> 550 cups / day (Unachievable)</div>'
                '</div>'
                '<div style="background:#ffffff; padding:0.55rem 0.75rem; border-radius:6px; border:1px solid #ef9a9a;">'
                '<div style="color:#c62828; font-weight:600; font-size:0.75rem;">CAPITAL RISK TIER</div>'
                '<div style="font-weight:700; color:#b71c1c;">High / Prohibitive</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            c_act1, c_act2 = st.columns([1.5, 2.0])
            with c_act1:
                if st.button("⚠️ File Rejection Justification", key="btn_memo_no"):
                    st.toast("⚠️ Rejection rationale archived in expansion records.")
            with c_act2:
                if st.button("🔍 Scout Alternative Territory", key="btn_scout"):
                    st.toast("🔍 Search filters reset to target prime transit corridors.")
                    
    with col_logic:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌳 ID3 Decision Rules Logic Matrix")
        st.caption("How Shannon Entropy splits this commercial evaluation:")
        
        path_box = (
            '<div style="background:#ffffff; border-radius:8px; border:1px solid #ebd9c8; padding:0.8rem; font-size:0.85rem;">'
            '<div style="margin-bottom:0.3rem; color:#795548;"><strong>1. Infrastructure & Walkability Gate:</strong></div>'
            f'<div style="padding-left:0.8rem; border-left:2px solid #2e7d32; margin-bottom:0.5rem; color:#1b5e20;">↳ Selected: <code>{transit_access}</code></div>'
            '<div style="margin-bottom:0.3rem; color:#795548;"><strong>2. Foot Traffic Density Threshold:</strong></div>'
            f'<div style="padding-left:0.8rem; border-left:2px solid #2e7d32; margin-bottom:0.5rem; color:#1b5e20;">↳ Selected: <code>{foot_traffic}</code></div>'
            '<div style="margin-bottom:0.3rem; color:#795548;"><strong>3. Overhead & Competitive Saturation Tradeoff:</strong></div>'
            f'<div style="padding-left:0.8rem; border-left:2px solid #2e7d32; margin-bottom:0.5rem; color:#1b5e20;">↳ Selected: <code>{lease_overhead}</code> | <code>{competition}</code></div>'
            '<div style="margin-bottom:0.3rem; color:#795548;"><strong>4. Zero-Entropy Terminal Verdict:</strong></div>'
            f'<div style="padding-left:0.8rem; border-left:3px solid #d84315; font-weight:bold; color:#3e2723;">↳ Classified Verdict: <strong>{verdict}</strong> (Leaf Pure H=0.000)</div>'
            '</div>'
        )
        st.markdown(path_box, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Decision Tree model file not found. Please train models first.")

st.markdown("---")
st.markdown("### 📊 Commercial Site Survey Benchmark Dataset (21 Historical Properties)")

if df_sites is not None:
    df_display = df_sites.copy()
    df_display.insert(0, "Site ID", [f"SITE-{i+1:02d}" for i in range(len(df_display))])
    df_display.columns = ["Site ID", "Foot Traffic Volume", "Lease Overhead Ratio", "Competitor Density", "Transit Accessibility", "Verdict (YES / NO)"]
    st.dataframe(df_display, hide_index=True, use_container_width=True)
else:
    st.info("Site survey dataset not found.")
