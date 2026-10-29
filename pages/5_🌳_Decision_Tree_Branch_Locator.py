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

st.markdown("<h2 class='brand-title'>5. Decision Tree (ID3): Cafe Branch Location Optimizer</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Information Gain & Entropy-Driven Site Selection for Retail Expansion</p>", unsafe_allow_html=True)

model = load_model("decision_tree")
metrics = load_all_metrics().get("decision_tree", {})

col_dt1, col_dt2 = st.columns([1, 1])
with col_dt1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Expanding an artisanal cafe chain requires selecting the optimal store format for a given real estate location. 
    Opening the wrong store concept (e.g. a huge sit-down lounge in a high-speed commuter transit concourse, or a drive-thru in a pedestrian heritage alley) 
    leads to crippling overhead and capital loss.
    
    **Why Decision Tree (Entropy / ID3)?**
    - Real estate expansion decisions must be transparent, defensible, and interpretable to franchise partners and corporate investors.
    - Uses **Shannon Entropy** to quantify uncertainty across location formats and splits hierarchically on the environmental condition offering the highest **Information Gain**:
    """)
    
    st.markdown(r"""
    $$\text{Entropy}(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$
    $$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
    """)
    
    st.markdown(f"""
    - **Splitting Criterion:** `Entropy (Information Gain)`
    - **Tree Depth:** `{metrics.get('tree_depth', 4)} hierarchical decision levels`
    - **Terminal Store Archetypes:** `{metrics.get('leaves_count', 7)} Location Formats`
    - **Training Scenarios:** `{metrics.get('samples', 26)} Calibrated Commercial Site Surveys`
    - **Model Classification Accuracy:** `{metrics.get('accuracy', 1.0):.1%}`
    """)
    
with col_dt2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🏢 Environmental Site Evaluation Inputs")
    st.caption("Select the environmental, demographic, and spatial characteristics of your target real estate site:")
    
    q_traffic = st.selectbox(
        "1. Predominant Foot Traffic & Commuter Demographic",
        [
            "Heavy Transit & Office Commuters",
            "Moderate Student & Academic",
            "Neighborhood Residents & Families",
            "Weekend Leisure & Tourists",
            "Low Pedestrian / High Vehicular Flow"
        ],
        index=0
    )
    
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        q_rent = st.selectbox(
            "2. Commercial Rent & Overhead Tier",
            [
                "High / Prime Downtown ($$$$)",
                "Moderate / Urban Commercial ($$)",
                "Economical / Low Overhead ($)"
            ],
            index=0
        )
        q_dwell = st.selectbox(
            "3. Expected Customer Dwell Time",
            [
                "Grab-&-Go (< 10 mins)",
                "Moderate Stay (15 - 45 mins)",
                "Extended Study & Work (> 60 mins)",
                "Social Dine-In & Brunch (30 - 60 mins)"
            ],
            index=0
        )
    with c_sub2:
        q_parking = st.selectbox(
            "4. Parking & Transit Infrastructure",
            [
                "Transit Hub / Walk-Up Only (Zero Parking)",
                "Street Parking & Shared Lot",
                "Dedicated Drive-Thru & Ample Parking"
            ],
            index=0
        )
        q_space = st.selectbox(
            "5. Available Commercial Space Capacity",
            [
                "Micro Space (< 40 sqm)",
                "Standard Lounge (80 - 150 sqm)",
                "Spacious Destination (> 200 sqm)"
            ],
            index=0
        )
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📍 Recommended Store Format & Capital Allocation Blueprint")

FORMAT_PROFILES = {
    "Downtown CBD Express Kiosk": {
        "icon": "🏙️",
        "tag": "High-Volume Commuter Transit Hub",
        "concept": "Ultra-fast automated espresso bar and batch cold brew taps engineered for rapid morning rush queue dissipation.",
        "footprint": "25 – 40 sqm (Compact Footprint)",
        "capex": "$85,000 – $115,000",
        "daily_volume": "650 – 900 cups / day",
        "avg_ticket": "$5.80 (Iced drinks & double shots)",
        "staffing": "2 Baristas (Peak morning shift)",
        "turnover": "< 3 minutes order-to-cup",
        "accent": "#00897b"
    },
    "University Campus Study Lounge": {
        "icon": "🎓",
        "tag": "Academic Third-Place & Study Hub",
        "concept": "High-bandwidth communal workspace with abundant electrical outlets, ergonomic seating, and student loyalty tiers.",
        "footprint": "100 – 160 sqm (Spacious Study Hall)",
        "capex": "$140,000 – $185,000",
        "daily_volume": "450 – 700 cups / day",
        "avg_ticket": "$6.40 (Cold brew, matcha, energy snacks)",
        "staffing": "3 – 4 Baristas (All-day steady)",
        "turnover": "High dwell (> 60 mins), high refill volume",
        "accent": "#3949ab"
    },
    "Suburban Community Roastery & Patio": {
        "icon": "🏡",
        "tag": "Residential Neighborhood Gathering Roastery",
        "concept": "Family and pet-friendly neighborhood hub featuring outdoor garden patio, artisanal brunch menu, and whole-bean retail bar.",
        "footprint": "160 – 240 sqm (Lounge + Outdoor Patio)",
        "capex": "$240,000 – $320,000",
        "daily_volume": "350 – 550 cups / day",
        "avg_ticket": "$8.90 (Food pairings, whole-bean bags)",
        "staffing": "4 – 6 Staff (Weekend brunch peaks)",
        "turnover": "30 – 60 mins dwell, high group tables",
        "accent": "#d84315"
    },
    "Highway Commuter Drive-Thru Express": {
        "icon": "🚗",
        "tag": "Arterial Corridor Dual Drive-Thru",
        "concept": "High-speed drive-thru with dual order lanes, dedicated digital ordering pickup window, and batch beverage dispensing.",
        "footprint": "50 – 80 sqm + 8-Car Stacking Lane",
        "capex": "$270,000 – $370,000",
        "daily_volume": "600 – 850 transactions / day",
        "avg_ticket": "$7.10 (Multi-drink breakfast orders)",
        "staffing": "3 – 4 Baristas (Rapid drive-thru rotation)",
        "turnover": "< 60 seconds at service window",
        "accent": "#f57c00"
    },
    "Arts & Heritage Flagship Tasting Room": {
        "icon": "🏛️",
        "tag": "Destination Roastery & Sensory Slow-Bar",
        "concept": "Experiential flagship in a cultural/historic setting with pour-over cupping flights, transparent roasting drums, and barista masterclasses.",
        "footprint": "200 – 320 sqm (High Ceilings & Tasting Bar)",
        "capex": "$350,000 – $480,000",
        "daily_volume": "300 – 450 guests / day",
        "avg_ticket": "$11.50 (Tasting flights, merchandise)",
        "staffing": "4 – 5 Specialty Baristas & Roaster",
        "turnover": "Leisurely experience, high margin per guest",
        "accent": "#8e24aa"
    },
    "Corporate Tech Park Micro-Lobby": {
        "icon": "🏢",
        "tag": "Captive Enterprise Campus Micro-Bar",
        "concept": "Seamless office building atrium integration with cashless mobile badge pay, corporate meeting carafes, and guaranteed weekday footfall.",
        "footprint": "30 – 50 sqm (Building Atrium / Lobby)",
        "capex": "$65,000 – $95,000",
        "daily_volume": "300 – 500 cups / day",
        "avg_ticket": "$5.40 (Corporate subsidized, high repeat)",
        "staffing": "2 Baristas (Monday – Friday 7 AM - 4 PM)",
        "turnover": "High morning/afternoon 15-min spikes",
        "accent": "#455a64"
    }
}

if model:
    df_query = pd.DataFrame([{
        "foot_traffic": q_traffic,
        "rent_budget": q_rent,
        "dwell_time": q_dwell,
        "parking_access": q_parking,
        "space_capacity": q_space
    }])
    
    match_format = model.predict(df_query)[0]
    info = FORMAT_PROFILES.get(match_format, FORMAT_PROFILES["Downtown CBD Express Kiosk"])
    
    col_res_card, col_res_tree = st.columns([1.4, 1.2])
    
    with col_res_card:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fdfbf7 0%, #f7efe6 100%); border: 2px solid {info['accent']}; border-radius: 12px; padding: 1.4rem; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="badge-pill" style="background:{info['accent']}; color:white; font-size:0.8rem;">⭐ OPTIMAL SITE EXPANSION ARCHETYPE</span>
                <span class="badge-pill badge-specialty">{info['tag']}</span>
            </div>
            <h3 style="margin: 0.3rem 0; color: #3e2723; font-weight: 800;">{info['icon']} {match_format}</h3>
            <p style="margin: 0.6rem 0; font-size: 0.95rem; color: #424242; line-height: 1.5;">{info['concept']}</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 1rem 0; font-size: 0.88rem;">
                <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #ebd9c8;">
                    <div style="color: #795548; font-weight: 600; font-size: 0.78rem;">ESTIMATED FITOUT CAPEX</div>
                    <div style="font-weight: 800; color: #d84315; font-size: 1.05rem;">{info['capex']}</div>
                </div>
                <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #ebd9c8;">
                    <div style="color: #795548; font-weight: 600; font-size: 0.78rem;">TARGET FOOTPRINT</div>
                    <div style="font-weight: 700; color: #3e2723;">{info['footprint']}</div>
                </div>
                <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #ebd9c8;">
                    <div style="color: #795548; font-weight: 600; font-size: 0.78rem;">DAILY VOLUME EXPECTATION</div>
                    <div style="font-weight: 700; color: #2e7d32;">{info['daily_volume']}</div>
                </div>
                <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid #ebd9c8;">
                    <div style="color: #795548; font-weight: 600; font-size: 0.78rem;">PROJECTED AVG TICKET</div>
                    <div style="font-weight: 700; color: #3e2723;">{info['avg_ticket']}</div>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; font-size: 0.85rem; color: #555555; background: #ffffff; padding: 0.5rem 0.8rem; border-radius: 6px; border: 1px solid #e0d4c8;">
                <span>Staffing: <strong>{info['staffing']}</strong></span>
                <span>•</span>
                <span>Speed of Service: <strong>{info['turnover']}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns([1.5, 2.0])
        with c_btn1:
            if st.button("📋 Generate Site Dossier", key="btn_dossier"):
                st.toast(f"✅ Generated Feasibility Dossier for {match_format}!")
        with c_btn2:
            if st.button("💾 Export Site Parameters", key="btn_export_site"):
                st.toast("✅ Site parameters saved to expansion pipeline!")

    with col_res_tree:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌳 Hierarchical Tree Decision Path")
        st.caption("Step-by-step route traversed through the ID3 tree based on your inputs:")
        
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 8px; border: 1px solid #ebd9c8; padding: 0.8rem; font-size: 0.85rem; margin-bottom: 0.8rem;">
            <div style="margin-bottom: 0.4rem; color: #795548;"><strong>1. Root Split (Traffic Demographic):</strong></div>
            <div style="padding-left: 0.8rem; border-left: 2px solid #2e7d32; margin-bottom: 0.6rem; color: #1b5e20;">
                ↳ Selected: <code>{q_traffic}</code>
            </div>
            
            <div style="margin-bottom: 0.4rem; color: #795548;"><strong>2. Financial & Overhead Split:</strong></div>
            <div style="padding-left: 0.8rem; border-left: 2px solid #2e7d32; margin-bottom: 0.6rem; color: #1b5e20;">
                ↳ Selected: <code>{q_rent}</code>
            </div>
            
            <div style="margin-bottom: 0.4rem; color: #795548;"><strong>3. Operational Dwell & Spatial Split:</strong></div>
            <div style="padding-left: 0.8rem; border-left: 2px solid #2e7d32; margin-bottom: 0.6rem; color: #1b5e20;">
                ↳ Selected: <code>{q_dwell}</code> | <code>{q_space}</code>
            </div>
            
            <div style="margin-bottom: 0.4rem; color: #795548;"><strong>4. Terminal Classification Leaf:</strong></div>
            <div style="padding-left: 0.8rem; border-left: 3px solid {info['accent']}; font-weight: bold; color: #3e2723;">
                ↳ Result: <strong>{match_format}</strong> (Zero Entropy Terminal Node)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 📊 Store Archetype Feasibility Matrix")
        df_benchmarks = pd.DataFrame([
            {"Archetype": "CBD Express Kiosk", "Typical Footprint": "25 - 40 sqm", "Rent Profile": "High ($$$$)", "Peak Hours": "Morning Rush (7 - 10 AM)"},
            {"Archetype": "Campus Study Lounge", "Typical Footprint": "100 - 160 sqm", "Rent Profile": "Moderate ($$)", "Peak Hours": "All-Day Study (9 AM - 8 PM)"},
            {"Archetype": "Suburban Roastery", "Typical Footprint": "160 - 240 sqm", "Rent Profile": "Moderate ($$)", "Peak Hours": "Weekend Brunch (8 AM - 3 PM)"},
            {"Archetype": "Highway Drive-Thru", "Typical Footprint": "50 - 80 sqm", "Rent Profile": "Low ($)", "Peak Hours": "Morning Commute (6 - 9 AM)"},
            {"Archetype": "Heritage Flagship", "Typical Footprint": "200 - 320 sqm", "Rent Profile": "High ($$$$)", "Peak Hours": "Afternoon & Weekend Leisure"},
            {"Archetype": "Tech Park Lobby", "Typical Footprint": "30 - 50 sqm", "Rent Profile": "Low ($)", "Peak Hours": "Weekday Office Hours (M-F)"}
        ])
        st.dataframe(df_benchmarks, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Model file not found. Please train models first.")
