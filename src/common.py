"""
RoastLab Coffee Co. - Common Utilities & Streamlit Components
Shared styles, resource loaders, and sidebar navigation components.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

def apply_custom_theme():
    st.markdown("""
    <style>
        .brand-title {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 800;
            color: #3e2723;
            margin-bottom: 0px;
        }
        .brand-subtitle {
            color: #795548;
            font-size: 1.15rem;
            margin-top: 0px;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #fdfbf7;
            border: 1px solid #ebd9c8;
            border-radius: 10px;
            padding: 1.2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .badge-specialty { background-color: #d7ccc8; color: #3e2723; }
        .badge-active { background-color: #c8e6c9; color: #1b5e20; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar_info():
    """Renders consistent project info in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ☕ RoastLab Co.")
        st.caption("ELEC026.23 Machine Learning Final Project")
        st.markdown("**Team Members:**")
        st.markdown("""
        - **Reponoya**
        - **Olpenado**
        - **Palencia**
        - **Rivera**
        """)
        st.markdown("---")
        st.caption("Specialty Coffee Retail Analytics Suite • v1.0")

@st.cache_resource
def load_model(model_name: str):
    """Loads a serialized .pkl model with caching."""
    filename = f"{model_name}.pkl" if not model_name.endswith(".pkl") else model_name
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Loads a CSV dataset from data/ directory with caching."""
    filename = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_all_metrics() -> dict:
    """Loads the model metrics JSON report."""
    path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
