"""
AirSight AI — Environmental AQI Analytics & Diagnostic Estimation Platform
Production-grade machine learning platform for atmospheric sub-index synthesis,
spatial exploratory data analysis, and model validation.
"""

import os
import io
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress sklearn unpickle version warnings gracefully
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="AirSight AI — Environmental AQI Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Reset & Typography ─────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* ── Top Header Card ───────────────────────────────────────────── */
    .dashboard-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .header-eyebrow {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 6px;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0 0 6px 0;
    }
    .header-subtitle {
        font-size: 0.90rem;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Stat Metric Cards ─────────────────────────────────────────── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .metric-eyebrow {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-number {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    .metric-number-accent {
        color: #2563EB;
    }
    .metric-caption {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* ── Content Card Container ────────────────────────────────────── */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    /* ── Scientific Disclosure Banner ──────────────────────────────── */
    .disclosure-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 0 0 22px 0;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }

    /* ── Section Dividers & Headings ───────────────────────────────── */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.015em;
        margin: 28px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
    }

    /* ── EPA Category Badges ───────────────────────────────────────── */
    .epa-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .epa-good          { background: #DCFCE7; color: #15803D; }
    .epa-mod           { background: #FEF3C7; color: #B45309; }
    .epa-usg           { background: #FFEDD5; color: #C2410C; }
    .epa-unhealthy     { background: #FEE2E2; color: #B91C1C; }
    .epa-veryunhealthy { background: #EDE9FE; color: #6D28D9; }
    .epa-hazardous     { background: #FFE4E6; color: #9F1239; }

    /* ── Left Sidebar Styling ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.2rem !important;
    }
    .sidebar-brand {
        padding-bottom: 16px;
        margin-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .sidebar-brand-subtitle {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #2563EB;
        margin-top: 4px;
    }

    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #64748B;
        margin: 14px 0 8px 0;
    }

    /* ── Sidebar Feature Highlight Card ────────────────────────────── */
    .sidebar-feature-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1.5px solid #3B82F6;
        border-radius: 9px;
        padding: 12px 14px;
        margin: 4px 0 10px 0;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08);
    }
    .feature-tag {
        font-size: 0.68rem;
        font-weight: 800;
        color: #1D4ED8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .feature-acc {
        background: #2563EB;
        color: #FFFFFF;
        font-size: 0.64rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 10px;
    }
    .feature-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-top: 4px;
    }
    .feature-desc {
        font-size: 0.72rem;
        color: #475569;
        margin-top: 3px;
        line-height: 1.35;
    }

    /* ── Highlighted ML Prediction Studio in Sidebar Radio ─────────── */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3) {
        background: #EFF6FF !important;
        border: 1.5px solid #3B82F6 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin: 6px 0 !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.12) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3) p {
        color: #1D4ED8 !important;
        font-weight: 700 !important;
    }

    /* Telemetry Info Cards in Sidebar */
    .telemetry-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
    }
    .telemetry-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 0.78rem;
        border-bottom: 1px dashed #E2E8F0;
    }
    .telemetry-item:last-child {
        border-bottom: none;
    }
    .telemetry-key {
        color: #64748B;
        font-weight: 500;
    }
    .telemetry-val {
        color: #0F172A;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Buttons ───────────────────────────────────────────────────── */
    .stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: 1px solid #0F172A !important;
        border-radius: 8px !important;
        padding: 9px 18px !important;
        letter-spacing: 0.01em !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
    }
    .stButton > button:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    /* ── Tab Bar Overrides ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #E2E8F0 !important;
        gap: 8px !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        color: #64748B !important;
        padding: 8px 16px !important;
        border-radius: 6px 6px 0 0 !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
        background: transparent !important;
    }

    /* ── Form Controls & Inputs ────────────────────────────────────── */
    .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #CBD5E1 !important;
        font-size: 0.88rem !important;
    }

    /* ── Code Badges ───────────────────────────────────────────────── */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 0.82em !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ── DataFrames ────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Standard EPA Color Mapping (US-EPA 40 CFR Part 58)
EPA_COLORS = {
    "Good": "#22C55E",                          # 0 - 50
    "Moderate": "#EAB308",                      # 51 - 100
    "Unhealthy for Sensitive Groups": "#F97316", # 101 - 150
    "Unhealthy": "#EF4444",                     # 151 - 200
    "Very Unhealthy": "#8B5CF6",                # 201 - 300
    "Hazardous": "#881337",                     # 301 - 500
}

# ==============================================================================
# 2. DYNAMIC ASSET RESOLUTION & CACHED LOADERS
# ==============================================================================
def resolve_model_file():
    """Detects the trained model bundle path safely without assuming exact filename."""
    candidates = [
        "aqi_model (3).pkl",
        "aqi_model.pkl",
        "aqi_model (1).pkl",
        "aqi_model (2).pkl",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    try:
        all_pkls = [f for f in os.listdir(".") if f.endswith(".pkl")]
        if all_pkls:
            return all_pkls[0]
    except Exception:
        pass
    return None

def resolve_dataset_file():
    """Detects the benchmark dataset path safely without assuming exact filename."""
    candidates = [
        "global_air_pollution_dataset[1].csv",
        "global air pollution dataset.csv",
        "global_air_pollution_dataset.csv",
        "global air pollution dataset.xlsx",
        "global_air_pollution_dataset.xlsx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    try:
        csvs = [f for f in os.listdir(".") if f.endswith(".csv") and "pollution" in f.lower()]
        if csvs:
            return csvs[0]
    except Exception:
        pass
    return None

@st.cache_resource(show_spinner="Loading trained machine learning model...")
def load_model_bundle():
    """Loads the pre-trained model bundle."""
    model_path = resolve_model_file()
    if not model_path:
        return None, "Model file not found. Ensure `aqi_model (3).pkl` or `aqi_model.pkl` is in root."
    try:
        bundle = joblib.load(model_path)
        return bundle, model_path
    except Exception as e:
        return None, f"Failed to unpickle `{model_path}`: {str(e)}"

@st.cache_data(show_spinner="Loading global air quality benchmark dataset...")
def load_air_quality_dataset():
    """Loads and sanitizes the empirical air quality dataset."""
    dataset_path = resolve_dataset_file()
    if not dataset_path:
        return None, "Dataset not found. Ensure `global_air_pollution_dataset[1].csv` is present."
    try:
        if dataset_path.endswith(".xlsx"):
            df = pd.read_excel(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        df["Country"] = df["Country"].fillna("Unspecified")
        df["City"] = df["City"].fillna("Unknown / Regional Station")

        num_cols = ["AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        return df, dataset_path
    except Exception as e:
        return None, f"Failed to load dataset `{dataset_path}`: {str(e)}"

# Load assets
raw_bundle, model_source = load_model_bundle()
raw_df, dataset_source = load_air_quality_dataset()

# Extract model components
if raw_bundle and isinstance(raw_bundle, dict):
    model = raw_bundle.get("model")
    expected_features = raw_bundle.get("features", ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"])
    model_name = raw_bundle.get("model_name", "Random Forest Regressor")
elif raw_bundle:
    model = raw_bundle
    expected_features = getattr(model, "feature_names_in_", ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"])
    model_name = "Random Forest Regressor"
else:
    model = None
    expected_features = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
    model_name = "Unavailable"

# ==============================================================================
# 3. SCIENTIFIC & HEALTH EVALUATION HELPERS (US-EPA 40 CFR Part 58)
# ==============================================================================
def get_epa_category_details(aqi_val: float):
    """
    Classifies a continuous AQI value into official EPA standardized categories
    (US-EPA 40 CFR Part 58 Breakpoints) with tailored public health advisories.
    """
    val = round(float(aqi_val))
    if val <= 50:
        return {
            "category": "Good",
            "range": "0 – 50",
            "color": "#16A34A",
            "badge_class": "epa-good",
            "summary": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "sensitive": "None. Individuals may safely engage in all outdoor activities.",
            "general": "Air quality is acceptable for all populations.",
            "action": "Ideal conditions for outdoor exercise, ventilation, and recreation.",
        }
    elif val <= 100:
        return {
            "category": "Moderate",
            "range": "51 – 100",
            "color": "#D97706",
            "badge_class": "epa-mod",
            "summary": "Air quality is acceptable; however, there may be moderate concern for very sensitive individuals.",
            "sensitive": "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "general": "General public is unlikely to experience adverse effects.",
            "action": "Normal outdoor activity is suitable for the majority of the population.",
        }
    elif val <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "range": "101 – 150",
            "color": "#EA580C",
            "badge_class": "epa-usg",
            "summary": "Members of sensitive groups may experience health effects. General public is not likely to be affected.",
            "sensitive": "People with lung disease (such as asthma), children, and older adults should limit prolonged exertion.",
            "general": "General public generally remains unaffected in this range.",
            "action": "Take more frequent breaks during vigorous outdoor activities.",
        }
    elif val <= 200:
        return {
            "category": "Unhealthy",
            "range": "151 – 200",
            "color": "#DC2626",
            "badge_class": "epa-unhealthy",
            "summary": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
            "sensitive": "Avoid prolonged or heavy outdoor exertion. Keep inhaler handy if asthmatic.",
            "general": "Everyone should reduce prolonged or heavy outdoor exertion.",
            "action": "Wear N95/FFP2 masks outdoors and utilize HEPA indoor filtration if available.",
        }
    elif val <= 300:
        return {
            "category": "Very Unhealthy",
            "range": "201 – 300",
            "color": "#7C3AED",
            "badge_class": "epa-veryunhealthy",
            "summary": "Health alert: The risk of health effects is significantly increased for everyone in the area.",
            "sensitive": "Avoid all physical activity outdoors. Move activities indoors.",
            "general": "Significantly reduce outdoor exertion; keep windows closed to limit infiltration.",
            "action": "Run indoor air purifiers, seal drafty openings, and avoid vigorous exercise.",
        }
    else:
        return {
            "category": "Hazardous",
            "range": "301 – 500+",
            "color": "#9F1239",
            "badge_class": "epa-hazardous",
            "summary": "Health warning of emergency conditions: Everyone is more likely to experience severe health impacts.",
            "sensitive": "Remain indoors and keep activity levels as low as possible.",
            "general": "Avoid all physical outdoor exertion. Emergency air pollution conditions.",
            "action": "Stay inside hermetically sealed spaces with medical-grade air filtration.",
        }

# Standard reusable chart layout builder
def get_chart_layout(title="", height=380):
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#0F172A"),
            x=0.01,
            y=0.98,
        ),
        template="plotly_white",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#475569"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=24, r=24, t=46 if title else 20, b=24),
        xaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            tickfont=dict(size=11, color="#64748B"),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            tickfont=dict(size=11, color="#64748B"),
        ),
    )

# ==============================================================================
# 4. SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">AirSight AI</div>
            <div class="sidebar-brand-subtitle">Environmental AQI Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    MODULE_PAGES = [
        "Modern Interface (Predict AQI & Analytics)",
        "Global Intelligence & Atmospheric Macroscope",
        "City Diagnostics & Environmental Profiles",
        "ML Prediction Studio & Sensitivity Sandbox",
        "Model Architecture, Benchmarks & Validation",
        "Scientific Methodology & Mathematical Proof",
    ]

    # Initialize nav_radio in session state
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = MODULE_PAGES[0]

    # ── Featured Quick-Access Card for ML Prediction Studio ───────────
    st.markdown("""
        <div class="sidebar-feature-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="feature-tag">Featured Engine</span>
                <span class="feature-acc">R² 0.9978</span>
            </div>
            <div class="feature-title">ML Prediction Studio</div>
            <div class="feature-desc">100-Tree Random Forest inference, parameter sandbox & ensemble diagnostics</div>
        </div>
    """, unsafe_allow_html=True)

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if st.button("Predict AQI", key="btn_quick_pred", use_container_width=True):
            st.session_state["nav_radio"] = "Modern Interface (Predict AQI & Analytics)"
            st.rerun()
    with c_b2:
        if st.button("Analytics", key="btn_quick_an", use_container_width=True):
            st.session_state["nav_radio"] = "Modern Interface (Predict AQI & Analytics)"
            st.rerun()

    st.markdown('<div class="sidebar-section-label" style="margin-top:14px;">Platform Navigation</div>', unsafe_allow_html=True)

    # Determine default index
    try:
        def_idx = MODULE_PAGES.index(st.session_state["nav_radio"])
    except ValueError:
        def_idx = 0

    selected_page = st.radio(
        "Select Platform Module",
        MODULE_PAGES,
        index=def_idx,
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown('<div class="sidebar-section-label">System Telemetry</div>', unsafe_allow_html=True)

    record_count_str = f"{len(raw_df):,}" if raw_df is not None else "0"

    st.markdown(f"""
        <div class="telemetry-box">
            <div class="telemetry-item">
                <span class="telemetry-key">Model</span>
                <span class="telemetry-val">Random Forest</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-key">Architecture</span>
                <span class="telemetry-val">100 Trees</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-key">Inputs</span>
                <span class="telemetry-val">4 Sub-Indices</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-key">Holdout Test MAE</span>
                <span class="telemetry-val" style="color:#2563EB;">0.2318</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-key">Benchmark Records</span>
                <span class="telemetry-val">{record_count_str}</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-key">Standard</span>
                <span class="telemetry-val">EPA 40 CFR 58</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:0.72rem; color:#64748B; line-height:1.5; margin-top:20px; padding-top:12px; border-top:1px solid #F1F5F9;">
            Diagnostic multi-pollutant sub-index estimation. Not an autoregressive temporal forecaster.
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 5. MODULE 0: MODERN TAILWIND INTERFACE (PREDICT AQI & ANALYTICS)
# ==============================================================================
if selected_page == "Modern Interface (Predict AQI & Analytics)":
    st.markdown("""
        <style>
            .block-container { padding: 0.2rem 0.5rem !important; max-width: 100% !important; }
            iframe { border: none !important; border-radius: 8px; width: 100% !important; }
        </style>
    """, unsafe_allow_html=True)

    html_file = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            full_html = f.read()
        import streamlit.components.v1 as components
        components.html(full_html, height=1400, scrolling=True)
    else:
        st.error("HTML template not found at templates/index.html")

# ==============================================================================
# 6. MODULE 1: GLOBAL INTELLIGENCE & MACROSCOPIC EXPLORATION
# ==============================================================================
elif selected_page == "Global Intelligence & Atmospheric Macroscope":
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-eyebrow">Macroscope View</div>
            <div class="header-title">Global Atmospheric Intelligence & Spatial Macroscope</div>
            <div class="header-subtitle">Comprehensive cross-sectional analysis of 23,463 monitoring stations across 175 sovereign nations</div>
        </div>
    """, unsafe_allow_html=True)

    if raw_df is None:
        st.error(f"Dataset could not be loaded: {dataset_source}")
        st.stop()

    # ── KPI METRICS ROW (6 Metrics) ───────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Total Stations</div>
                <div class="metric-number">{len(raw_df):,}</div>
                <div class="metric-caption">Monitored stations</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Countries</div>
                <div class="metric-number metric-number-accent">{raw_df['Country'].nunique():,}</div>
                <div class="metric-caption">Sovereign territories</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Unique Cities</div>
                <div class="metric-number">{raw_df['City'].nunique():,}</div>
                <div class="metric-caption">Urban coverage</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi4:
        median_aqi = raw_df["AQI Value"].median()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Median AQI</div>
                <div class="metric-number">{median_aqi:.0f}</div>
                <div class="metric-caption">Global station median</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi5:
        pm25_dominant_pct = ((raw_df["PM2.5 AQI Value"] >= raw_df["Ozone AQI Value"]) & 
                             (raw_df["PM2.5 AQI Value"] >= raw_df["CO AQI Value"]) & 
                             (raw_df["PM2.5 AQI Value"] >= raw_df["NO2 AQI Value"])).mean() * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">PM2.5 Dominance</div>
                <div class="metric-number metric-number-accent">{pm25_dominant_pct:.1f}%</div>
                <div class="metric-caption">Governing sub-index</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi6:
        unhealthy_pct = (raw_df["AQI Value"] > 100).mean() * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Exceedance Rate</div>
                <div class="metric-number" style="color:#DC2626;">{unhealthy_pct:.1f}%</div>
                <div class="metric-caption">Stations > 100 AQI</div>
            </div>
        """, unsafe_allow_html=True)

    # ── SECTION 1: GLOBAL CHOROPLETH & REGIONAL HIERARCHICAL TREEMAP ──────────
    st.markdown('<div class="section-title">Global Choropleth Map & Geographic Aggregations</div>', unsafe_allow_html=True)

    country_agg = raw_df.groupby("Country").agg(
        mean_aqi=("AQI Value", "mean"),
        median_aqi=("AQI Value", "median"),
        city_count=("City", "count"),
        max_aqi=("AQI Value", "max"),
    ).reset_index()

    fig_map = px.choropleth(
        country_agg,
        locations="Country",
        locationmode="country names",
        color="mean_aqi",
        hover_name="Country",
        hover_data={"mean_aqi": ":.1f", "median_aqi": ":.1f", "city_count": True, "max_aqi": True},
        color_continuous_scale=[
            (0.00, "#22C55E"),  # Good (0-50)
            (0.20, "#EAB308"),  # Moderate (50-100)
            (0.35, "#F97316"),  # USG (101-150)
            (0.55, "#EF4444"),  # Unhealthy (151-200)
            (0.75, "#8B5CF6"),  # Very Unhealthy (201-300)
            (1.00, "#881337"),  # Hazardous (300+)
        ],
        labels={"mean_aqi": "Mean AQI"},
    )
    fig_map.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=520,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(
            title=dict(text="Mean AQI", font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0F172A")),
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#64748B"),
            thickness=14,
            len=0.75,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        geo=dict(
            bgcolor="#FFFFFF",
            projection_type="natural earth",
            showcoastlines=True,
            coastlinecolor="#CBD5E1",
            coastlinewidth=0.8,
            showcountries=True,
            countrycolor="#E2E8F0",
            countrywidth=0.6,
            showland=True,
            landcolor="#F8FAFC",
            showocean=True,
            oceancolor="#F8FAFC",
            showlakes=False,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── SECTION 2: HIERARCHICAL TREEMAP & CATEGORY BREAKDOWN ──────────────────
    t_col1, t_col2 = st.columns([1.2, 0.8])
    with t_col1:
        top_25_countries = raw_df["Country"].value_counts().head(25).index
        tree_df = raw_df[raw_df["Country"].isin(top_25_countries)].groupby(["Country", "AQI Category"]).size().reset_index(name="Stations")
        fig_tree = px.treemap(
            tree_df,
            path=["Country", "AQI Category"],
            values="Stations",
            color="AQI Category",
            color_discrete_map=EPA_COLORS,
            title="Top 25 Monitored Nations: Hierarchical Station Breakdown by EPA Category",
        )
        fig_tree.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=11),
            height=390,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with t_col2:
        cat_counts = raw_df["AQI Category"].value_counts().reset_index()
        cat_counts.columns = ["AQI Category", "Count"]
        epa_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        cat_counts["Order"] = cat_counts["AQI Category"].map(lambda x: epa_order.index(x) if x in epa_order else 99)
        cat_counts = cat_counts.sort_values("Order")

        fig_donut = px.pie(
            cat_counts,
            names="AQI Category",
            values="Count",
            hole=0.65,
            color="AQI Category",
            color_discrete_map=EPA_COLORS,
            title="Global EPA Category Composition",
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig_donut.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#475569"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=390,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── SECTION 3: MULTI-AXIS PARALLEL COORDINATES (HIGH-TECH EXPLORATION) ────
    st.markdown('<div class="section-title">Multi-Dimensional Atmospheric Sub-Index Synthesis (Parallel Coordinates)</div>', unsafe_allow_html=True)
    st.caption("Drag and filter along vertical axes to inspect multi-pollutant trajectories and see how extreme particulate readings govern composite AQI.")

    par_sample = raw_df.sample(n=min(1400, len(raw_df)), random_state=42)
    fig_parcoords = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=par_sample["AQI Value"],
                colorscale=[
                    [0.0, "#22C55E"],
                    [0.2, "#EAB308"],
                    [0.35, "#F97316"],
                    [0.55, "#EF4444"],
                    [0.75, "#8B5CF6"],
                    [1.0, "#881337"],
                ],
                showscale=True,
                colorbar=dict(title="Overall AQI", thickness=14, len=0.85),
                cmin=0,
                cmax=350,
            ),
            dimensions=[
                dict(range=[0, 80], label="CO AQI", values=par_sample["CO AQI Value"]),
                dict(range=[0, 180], label="Ozone AQI", values=par_sample["Ozone AQI Value"]),
                dict(range=[0, 80], label="NO2 AQI", values=par_sample["NO2 AQI Value"]),
                dict(range=[0, 400], label="PM2.5 AQI", values=par_sample["PM2.5 AQI Value"]),
                dict(range=[0, 450], label="Overall AQI", values=par_sample["AQI Value"]),
            ]
        )
    )
    fig_parcoords.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0F172A"),
        height=380,
        margin=dict(l=60, r=60, t=50, b=24),
    )
    st.plotly_chart(fig_parcoords, use_container_width=True)

    # ── SECTION 4: VIOLIN KERNEL DENSITIES & CORRELATION HEATMAP ──────────────
    st.markdown('<div class="section-title">Atmospheric Distributions & Inter-Pollutant Correlations</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1.1, 0.9])

    with col_p1:
        pollutant_melt = raw_df.melt(
            id_vars=["AQI Category"],
            value_vars=["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"],
            var_name="Pollutant",
            value_name="Sub-Index AQI",
        )
        fig_violin = px.violin(
            pollutant_melt,
            x="Pollutant",
            y="Sub-Index AQI",
            color="Pollutant",
            box=True,
            points=False,
            color_discrete_sequence=["#2563EB", "#0284C7", "#D97706", "#DC2626"],
            title="Kernel Density Profiles of Monitored Sub-Indices (0–500 Scale)",
        )
        layout_v = get_chart_layout(height=360)
        layout_v["showlegend"] = False
        fig_violin.update_layout(layout_v)
        st.plotly_chart(fig_violin, use_container_width=True)

    with col_p2:
        corr_cols = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value", "AQI Value"]
        corr_matrix = raw_df[corr_cols].corr().round(3)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1E3A8A"]],
            title="Pearson Correlation with Composite AQI",
        )
        layout_corr = get_chart_layout(height=360)
        fig_corr.update_layout(layout_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── SECTION 5: GLOBAL LEADERBOARDS (CLEANEST VS HIGHEST AQI) ──────────────
    st.markdown('<div class="section-title">Global Atmospheric Extremes (Cleanest vs Highest AQI Cities)</div>', unsafe_allow_html=True)
    city_means = raw_df.groupby(["City", "Country"])["AQI Value"].mean().reset_index()

    lead_c1, lead_c2 = st.columns(2)
    with lead_c1:
        cleanest_12 = city_means.sort_values("AQI Value").head(12)
        cleanest_12["Station"] = cleanest_12["City"] + ", " + cleanest_12["Country"]
        fig_clean = px.bar(
            cleanest_12,
            x="AQI Value",
            y="Station",
            orientation="h",
            color="AQI Value",
            color_continuous_scale=[[0, "#16A34A"], [1, "#86EFAC"]],
            text="AQI Value",
            title="Top 12 Cleanest Global Monitoring Stations (Lowest AQI)",
        )
        layout_cl = get_chart_layout(height=380)
        layout_cl["yaxis"]["autorange"] = "reversed"
        layout_cl["coloraxis_showscale"] = False
        fig_clean.update_layout(layout_cl)
        fig_clean.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig_clean, use_container_width=True)

    with lead_c2:
        polluted_12 = city_means.sort_values("AQI Value", ascending=False).head(12)
        polluted_12["Station"] = polluted_12["City"] + ", " + polluted_12["Country"]
        fig_poll = px.bar(
            polluted_12,
            x="AQI Value",
            y="Station",
            orientation="h",
            color="AQI Value",
            color_continuous_scale=[[0, "#F87171"], [1, "#7F1D1D"]],
            text="AQI Value",
            title="Top 12 Most Challenged Global Stations (Highest AQI)",
        )
        layout_pl = get_chart_layout(height=380)
        layout_pl["yaxis"]["autorange"] = "reversed"
        layout_pl["coloraxis_showscale"] = False
        fig_poll.update_layout(layout_pl)
        fig_poll.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig_poll, use_container_width=True)


# ==============================================================================
# 6. MODULE 2: GLOBAL CITY EXPLORER & URBAN PROFILER
# ==============================================================================
elif selected_page == "City Diagnostics & Environmental Profiles":
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-eyebrow">Local Diagnostics</div>
            <div class="header-title">City Diagnostics & Environmental Profiles</div>
            <div class="header-subtitle">Inspect recorded multi-pollutant observations across 175 sovereign nations and 23,462 cities</div>
        </div>
    """, unsafe_allow_html=True)

    if raw_df is None:
        st.error("Dataset not loaded.")
        st.stop()

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        all_countries = sorted([str(c) for c in raw_df["Country"].unique() if str(c) != "Unspecified"])
        default_c_idx = all_countries.index("United States of America") if "United States of America" in all_countries else 0
        selected_country = st.selectbox("Select Sovereign Country", all_countries, index=default_c_idx)

    country_df = raw_df[raw_df["Country"] == selected_country]
    city_list = sorted(country_df["City"].dropna().unique().tolist())

    with col_sel2:
        selected_city = st.selectbox("Select City", city_list, index=0)

    city_records = country_df[country_df["City"] == selected_city]
    rec_count = len(city_records)

    st.markdown(f'<div class="section-title">{selected_city}, {selected_country} ({rec_count} recorded monitoring station{"s" if rec_count > 1 else ""})</div>', unsafe_allow_html=True)

    primary_record = city_records.iloc[0]
    aqi_cat_info = get_epa_category_details(primary_record["AQI Value"])

    score_c1, score_c2, score_c3, score_c4 = st.columns([1, 1, 1, 1.2])
    with score_c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Recorded Overall AQI</div>
                <div class="metric-number" style="color:{aqi_cat_info['color']};">{int(primary_record['AQI Value'])}</div>
                <div style="margin-top:8px;"><span class="epa-badge {aqi_cat_info['badge_class']}">{aqi_cat_info['category']}</span></div>
            </div>
        """, unsafe_allow_html=True)

    with score_c2:
        nat_avg = country_df["AQI Value"].mean()
        diff_nat = primary_record["AQI Value"] - nat_avg
        diff_sign = "+" if diff_nat > 0 else ""
        diff_color = "#DC2626" if diff_nat > 0 else "#16A34A"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">National Average</div>
                <div class="metric-number">{nat_avg:.1f}</div>
                <div class="metric-caption" style="color:{diff_color}; font-weight:600;">
                    {diff_sign}{diff_nat:.1f} vs National Mean
                </div>
            </div>
        """, unsafe_allow_html=True)

    with score_c3:
        glob_avg = raw_df["AQI Value"].mean()
        diff_glob = primary_record["AQI Value"] - glob_avg
        diff_sign_g = "+" if diff_glob > 0 else ""
        diff_color_g = "#DC2626" if diff_glob > 0 else "#16A34A"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-eyebrow">Global Benchmark</div>
                <div class="metric-number">{glob_avg:.1f}</div>
                <div class="metric-caption" style="color:{diff_color_g}; font-weight:600;">
                    {diff_sign_g}{diff_glob:.1f} vs Global Mean
                </div>
            </div>
        """, unsafe_allow_html=True)

    with score_c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Diagnostic Action</div>
                <div style="font-size:0.80rem; color:#64748B; margin: 4px 0 10px 0;">Transfer station readings into the Machine Learning Estimator.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Load into ML Estimator", use_container_width=True):
            st.session_state["co_val"] = float(primary_record["CO AQI Value"])
            st.session_state["ozone_val"] = float(primary_record["Ozone AQI Value"])
            st.session_state["no2_val"] = float(primary_record["NO2 AQI Value"])
            st.session_state["pm25_val"] = float(primary_record["PM2.5 AQI Value"])
            st.session_state["loaded_city_name"] = f"{selected_city}, {selected_country}"
            st.success(f"Transferred readings for {selected_city}. Switch to 'ML Prediction Studio' in sidebar.")

    # ── SECTION: RADAR SPIDER PROFILE & SUB-INDEX BREAKDOWN ───────────────────
    st.markdown('<div class="section-title">Atmospheric Radar Profile & Sub-Index Comparison</div>', unsafe_allow_html=True)
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        # 4-Axis Polar Radar Spider Chart
        categories = ["CO", "Ozone", "NO2", "PM2.5"]
        city_vals = [
            float(primary_record["CO AQI Value"]),
            float(primary_record["Ozone AQI Value"]),
            float(primary_record["NO2 AQI Value"]),
            float(primary_record["PM2.5 AQI Value"]),
        ]
        nat_vals = [
            float(country_df["CO AQI Value"].mean()),
            float(country_df["Ozone AQI Value"].mean()),
            float(country_df["NO2 AQI Value"].mean()),
            float(country_df["PM2.5 AQI Value"].mean()),
        ]
        glob_vals = [
            float(raw_df["CO AQI Value"].mean()),
            float(raw_df["Ozone AQI Value"].mean()),
            float(raw_df["NO2 AQI Value"].mean()),
            float(raw_df["PM2.5 AQI Value"].mean()),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=city_vals + [city_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=selected_city,
            line=dict(color='#2563EB', width=2),
            fillcolor='rgba(37, 99, 235, 0.25)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=nat_vals + [nat_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'{selected_country} Avg',
            line=dict(color='#D97706', width=1.5, dash='dash'),
            fillcolor='rgba(217, 119, 6, 0.12)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=glob_vals + [glob_vals[0]],
            theta=categories + [categories[0]],
            name='Global Benchmark',
            line=dict(color='#94A3B8', width=1.5, dash='dot'),
        ))
        max_r = max(max(city_vals) * 1.15, 60)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_r], gridcolor="#E2E8F0", tickfont=dict(size=10, color="#64748B")),
                angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, sans-serif")),
                bgcolor="#FFFFFF",
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            height=370,
            title=dict(text=f"<b>Sub-Index Fingerprint: {selected_city} vs Benchmarks</b>", font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#0F172A")),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        city_sub = pd.DataFrame({
            "Pollutant": ["CO", "Ozone (O3)", "NO2", "PM2.5"],
            "Sub-Index AQI": [
                primary_record["CO AQI Value"],
                primary_record["Ozone AQI Value"],
                primary_record["NO2 AQI Value"],
                primary_record["PM2.5 AQI Value"],
            ],
            "Category": [
                primary_record["CO AQI Category"],
                primary_record["Ozone AQI Category"],
                primary_record["NO2 AQI Category"],
                primary_record["PM2.5 AQI Category"],
            ]
        })
        fig_sub = px.bar(
            city_sub,
            x="Pollutant",
            y="Sub-Index AQI",
            color="Category",
            color_discrete_map=EPA_COLORS,
            text="Sub-Index AQI",
            title=f"Recorded Pollutant Sub-Indices: {selected_city}",
        )
        layout_sub = get_chart_layout(height=370)
        fig_sub.update_layout(layout_sub)
        fig_sub.update_traces(textposition="outside")
        st.plotly_chart(fig_sub, use_container_width=True)

    # ── SECTION: NATIONAL BUBBLE MATRIX (PM2.5 vs Ozone across country) ──────
    st.markdown(f'<div class="section-title">Urban Pollutant Matrix ({selected_country})</div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([1.2, 0.8])
    with b_col1:
        fig_bubble = px.scatter(
            country_df,
            x="PM2.5 AQI Value",
            y="Ozone AQI Value",
            size="AQI Value",
            color="AQI Category",
            color_discrete_map=EPA_COLORS,
            hover_name="City",
            hover_data={"CO AQI Value": True, "NO2 AQI Value": True, "AQI Value": True},
            title=f"PM2.5 vs Ozone Sub-Indices across {len(country_df):,} Stations in {selected_country}",
            size_max=24,
        )
        layout_bub = get_chart_layout(height=380)
        fig_bubble.update_layout(layout_bub)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with b_col2:
        cleanest = country_df.sort_values("AQI Value").head(5)[["City", "AQI Value", "AQI Category"]]
        highest = country_df.sort_values("AQI Value", ascending=False).head(5)[["City", "AQI Value", "AQI Category"]]

        tab_clean, tab_polluted = st.tabs(["Cleanest Cities", "Highest AQI Cities"])
        with tab_clean:
            st.dataframe(cleanest.reset_index(drop=True), use_container_width=True)
        with tab_polluted:
            st.dataframe(highest.reset_index(drop=True), use_container_width=True)


# ==============================================================================
# 7. MODULE 3: ML PREDICTION STUDIO & SENSITIVITY SANDBOX
# ==============================================================================
elif selected_page == "ML Prediction Studio & Sensitivity Sandbox":
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-eyebrow">Interactive ML Studio</div>
            <div class="header-title">ML Prediction Studio & Sensitivity Sandbox</div>
            <div class="header-subtitle">Real-time inference, clinical health advisories, 100-tree ensemble consensus, and multi-dimensional atmospheric diagnostics</div>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(f"Model Loading Failure: {model_source}")
        st.info("To resolve: Verify that `aqi_model (3).pkl` or `aqi_model.pkl` is in root.")
        st.stop()

    if "loaded_city_name" in st.session_state:
        st.info(f"Active station telemetry loaded from: **{st.session_state['loaded_city_name']}**")

    # ── SECTION 1: MODEL ACCURACY & TELEMETRY BAR ─────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:0;">Model Performance & Accuracy Telemetry</div>', unsafe_allow_html=True)

    acc1, acc2, acc3, acc4, acc5 = st.columns(5)
    with acc1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Holdout Test MAE</div>
                <div class="metric-number metric-number-accent">0.2318</div>
                <div class="metric-caption">Mean Absolute Error (test n = 4,693)</div>
            </div>
        """, unsafe_allow_html=True)
    with acc2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Variance Explained</div>
                <div class="metric-number metric-number-accent">99.78%</div>
                <div class="metric-caption">R² Score on holdout test set</div>
            </div>
        """, unsafe_allow_html=True)
    with acc3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Precision Tier (&plusmn;5)</div>
                <div class="metric-number">99.28%</div>
                <div class="metric-caption">Within &plusmn;5 AQI points of ground truth</div>
            </div>
        """, unsafe_allow_html=True)
    with acc4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Holdout RMSE</div>
                <div class="metric-number">2.6621</div>
                <div class="metric-caption">Root Mean Squared Error (penalizes outliers)</div>
            </div>
        """, unsafe_allow_html=True)
    with acc5:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Ensemble Size</div>
                <div class="metric-number">100 Trees</div>
                <div class="metric-caption">Random Forest (random_state=42)</div>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("Comparative Benchmark Evaluation (Random Forest vs Competing Models)", expanded=False):
        comp_df = pd.DataFrame([
            {"Model Architecture": "Random Forest Regressor (Selected Champion)", "Test MAE (AQI)": "0.2318", "Test RMSE": "2.6621", "Test R²": "0.9978", "Status": "Selected Champion Model"},
            {"Model Architecture": "Rule-Based EPA Maximum Baseline", "Test MAE (AQI)": "0.3516", "Test RMSE": "4.9316", "Test R²": "0.9924", "Status": "Theoretical Upper Bound"},
            {"Model Architecture": "Gradient Boosting Regressor", "Test MAE (AQI)": "1.5591", "Test RMSE": "3.4210", "Test R²": "0.9958", "Status": "Sequential Boosting Baseline"},
            {"Model Architecture": "Linear Regression", "Test MAE (AQI)": "4.8038", "Test RMSE": "6.8924", "Test R²": "0.9852", "Status": "Underfits Piecewise Surface"},
            {"Model Architecture": "Mean Reference Baseline", "Test MAE (AQI)": "36.9632", "Test RMSE": "47.1205", "Test R²": "0.0000", "Status": "Naive Uninformative Baseline"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("All models trained on 18,770 stations (80% split) and evaluated strictly on 4,693 holdout test stations (20% split). Random Forest achieved superior performance across all error metrics.")

    st.markdown("""
        <div class="disclosure-banner">
            <strong>Sub-Index Diagnostic Synthesis:</strong> This engine inputs four standardized pollutant sub-indices
            and estimates overall continuous AQI using the trained Random Forest model.
            It does <em>not</em> forecast chronological futures or take raw concentrations (ppm/&mu;g/m&sup3;) as inputs.
        </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: DIAGNOSTIC SCENARIO PRESETS ────────────────────────────────
    st.markdown('<div class="section-title">Diagnostic Scenario Presets</div>', unsafe_allow_html=True)
    st.caption("Load verified atmospheric profiles directly into the input parameters:")

    pr_cols = st.columns(5)
    with pr_cols[0]:
        if st.button("Pristine Alpine Air", use_container_width=True):
            st.session_state["co_val"] = 1.0
            st.session_state["ozone_val"] = 20.0
            st.session_state["no2_val"] = 2.0
            st.session_state["pm25_val"] = 15.0
            st.session_state["loaded_city_name"] = "Pristine Air Preset"
            st.session_state["est_executed"] = True
            st.rerun()
    with pr_cols[1]:
        if st.button("Urban Residential", use_container_width=True):
            st.session_state["co_val"] = 3.0
            st.session_state["ozone_val"] = 42.0
            st.session_state["no2_val"] = 14.0
            st.session_state["pm25_val"] = 65.0
            st.session_state["loaded_city_name"] = "Urban Moderate Preset"
            st.session_state["est_executed"] = True
            st.rerun()
    with pr_cols[2]:
        if st.button("Photochemical Smog", use_container_width=True):
            st.session_state["co_val"] = 4.0
            st.session_state["ozone_val"] = 135.0
            st.session_state["no2_val"] = 24.0
            st.session_state["pm25_val"] = 38.0
            st.session_state["loaded_city_name"] = "Photochemical Smog Preset"
            st.session_state["est_executed"] = True
            st.rerun()
    with pr_cols[3]:
        if st.button("Traffic / NO2 Corridor", use_container_width=True):
            st.session_state["co_val"] = 18.0
            st.session_state["ozone_val"] = 30.0
            st.session_state["no2_val"] = 95.0
            st.session_state["pm25_val"] = 55.0
            st.session_state["loaded_city_name"] = "Traffic Corridor Preset"
            st.session_state["est_executed"] = True
            st.rerun()
    with pr_cols[4]:
        if st.button("Wildfire Smoke Inversion", use_container_width=True):
            st.session_state["co_val"] = 32.0
            st.session_state["ozone_val"] = 110.0
            st.session_state["no2_val"] = 52.0
            st.session_state["pm25_val"] = 340.0
            st.session_state["loaded_city_name"] = "Wildfire Inversion Preset"
            st.session_state["est_executed"] = True
            st.rerun()

    # ── SECTION 3: PARAMETERS TO INPUT ────────────────────────────────────────
    st.markdown('<div class="section-title">Model Input Parameters (Pollutant Sub-Indices)</div>', unsafe_allow_html=True)
    st.caption("Standardized non-negative numerical sub-indices (0–500 scale). Values mapped in strict training feature sequence: 1. CO, 2. Ozone, 3. NO2, 4. PM2.5.")

    in_col1, in_col2 = st.columns(2)
    with in_col1:
        co_init = float(st.session_state.get("co_val", 2.0))
        co_input = st.number_input(
            "1. Carbon Monoxide (CO AQI Value)",
            min_value=0.0,
            max_value=500.0,
            value=co_init,
            step=1.0,
            help="Sub-index for Carbon Monoxide (CO). Incomplete combustion & traffic exhaust. Clean air baseline: 0 to 50.",
            key="input_co",
        )
        co_epa = get_epa_category_details(co_input)
        st.markdown(f'<div style="font-size:0.75rem; color:#64748B; margin-top:-6px; margin-bottom:12px;">Live Sub-Index Status: <span class="epa-badge {co_epa["badge_class"]}">{co_epa["category"]} ({co_epa["range"]})</span></div>', unsafe_allow_html=True)

        ozone_init = float(st.session_state.get("ozone_val", 35.0))
        ozone_input = st.number_input(
            "2. Ground-Level Ozone (Ozone AQI Value)",
            min_value=0.0,
            max_value=500.0,
            value=ozone_init,
            step=1.0,
            help="Sub-index for Ground-Level Ozone (O3). Secondary photochemical oxidant formed by sunlight & VOC/NOx precursors.",
            key="input_ozone",
        )
        ozone_epa = get_epa_category_details(ozone_input)
        st.markdown(f'<div style="font-size:0.75rem; color:#64748B; margin-top:-6px; margin-bottom:12px;">Live Sub-Index Status: <span class="epa-badge {ozone_epa["badge_class"]}">{ozone_epa["category"]} ({ozone_epa["range"]})</span></div>', unsafe_allow_html=True)

    with in_col2:
        no2_init = float(st.session_state.get("no2_val", 8.0))
        no2_input = st.number_input(
            "3. Nitrogen Dioxide (NO2 AQI Value)",
            min_value=0.0,
            max_value=500.0,
            value=no2_init,
            step=1.0,
            help="Sub-index for Nitrogen Dioxide (NO2). Emitted from motor vehicles and thermal power generation.",
            key="input_no2",
        )
        no2_epa = get_epa_category_details(no2_input)
        st.markdown(f'<div style="font-size:0.75rem; color:#64748B; margin-top:-6px; margin-bottom:12px;">Live Sub-Index Status: <span class="epa-badge {no2_epa["badge_class"]}">{no2_epa["category"]} ({no2_epa["range"]})</span></div>', unsafe_allow_html=True)

        pm25_init = float(st.session_state.get("pm25_val", 55.0))
        pm25_input = st.number_input(
            "4. Fine Particulate Matter (PM2.5 AQI Value)",
            min_value=0.0,
            max_value=500.0,
            value=pm25_init,
            step=1.0,
            help="Sub-index for Fine Particulate Matter (PM2.5). Inhalable particles ≤ 2.5 µm. Dominant predictor in 98% of dataset.",
            key="input_pm25",
        )
        pm25_epa = get_epa_category_details(pm25_input)
        st.markdown(f'<div style="font-size:0.75rem; color:#64748B; margin-top:-6px; margin-bottom:12px;">Live Sub-Index Status: <span class="epa-badge {pm25_epa["badge_class"]}">{pm25_epa["category"]} ({pm25_epa["range"]})</span></div>', unsafe_allow_html=True)

    # Governing Sub-Index Preview
    param_dict = {
        "Carbon Monoxide (CO)": co_input,
        "Ground-Level Ozone (Ozone)": ozone_input,
        "Nitrogen Dioxide (NO2)": no2_input,
        "Fine Particulate Matter (PM2.5)": pm25_input,
    }
    gov_name = max(param_dict, key=param_dict.get)
    gov_score = param_dict[gov_name]
    gov_details = get_epa_category_details(gov_score)

    st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px; margin: 4px 0 16px 0; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.75rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">Governing Determinant:</span>
                <span style="font-size:0.86rem; color:#0F172A; font-weight:700; margin-left:6px;">{gov_name}</span>
                <span style="font-size:0.82rem; color:#2563EB; font-weight:700; margin-left:4px;">({gov_score:.0f} AQI)</span>
            </div>
            <div>
                <span class="epa-badge {gov_details['badge_class']}">{gov_details['category']}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        "CO AQI Value": co_input,
        "Ozone AQI Value": ozone_input,
        "NO2 AQI Value": no2_input,
        "PM2.5 AQI Value": pm25_input,
    }])[expected_features]

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    run_est = st.button("Generate ML Estimation & Visual Diagnostics", type="primary", use_container_width=True)

    # ── SECTION 4: POST-PREDICTION OUTPUT & DIAGNOSTIC GRAPHS ─────────────────
    if run_est or "est_executed" in st.session_state:
        st.session_state["est_executed"] = True

        start_t = time.perf_counter()
        raw_pred = model.predict(input_df)[0]
        inference_latency_ms = (time.perf_counter() - start_t) * 1000

        pred_aqi = max(0.0, round(float(raw_pred), 1))
        epa_details = get_epa_category_details(pred_aqi)

        # ── 100-Tree Individual Predictions Extraction ──
        estimators = getattr(model, "estimators_", [])
        if len(estimators) > 0:
            tree_preds = np.array([float(t.predict(input_df.values)[0]) for t in estimators])
            tree_mean = float(np.mean(tree_preds))
            tree_std = float(np.std(tree_preds))
            tree_ci_low = float(np.percentile(tree_preds, 2.5))
            tree_ci_high = float(np.percentile(tree_preds, 97.5))
        else:
            tree_preds = np.array([pred_aqi])
            tree_mean = pred_aqi
            tree_std = 0.0
            tree_ci_low = pred_aqi
            tree_ci_high = pred_aqi

        # ── PRIMARY ESTIMATION & CLINICAL HEALTH CARD ──
        st.markdown('<div class="section-title">Estimation Result & Clinical Health Guidance</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_aqi,
                title={"text": f"Estimated AQI: <b>{epa_details['category']}</b>", "font": {"size": 16, "family": "Plus Jakarta Sans, sans-serif", "color": "#0F172A"}},
                gauge={
                    "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": epa_details["color"], "thickness": 0.3},
                    "bgcolor": "#F8FAFC",
                    "borderwidth": 1,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(34, 197, 94, 0.2)"},
                        {"range": [50, 100], "color": "rgba(234, 179, 8, 0.2)"},
                        {"range": [100, 150], "color": "rgba(249, 115, 22, 0.2)"},
                        {"range": [150, 200], "color": "rgba(239, 68, 68, 0.2)"},
                        {"range": [200, 300], "color": "rgba(139, 92, 246, 0.2)"},
                        {"range": [300, 500], "color": "rgba(136, 19, 55, 0.2)"},
                    ],
                }
            ))
            fig_gauge.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                font={"color": "#0F172A", "family": "Plus Jakarta Sans, sans-serif"},
                margin=dict(l=24, r=24, t=30, b=24),
                height=300,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
                <div style="font-size:0.78rem; color:#64748B; text-align:center; margin-top:-10px;">
                    Inference Latency: <code>{inference_latency_ms:.2f} ms</code> &bull; Tree Consensus (&sigma;): <code>{tree_std:.2f}</code> &bull; Standard: <code>EPA 40 CFR 58</code>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div class="content-card" style="border-left: 4px solid {epa_details['color']};">
                    <span class="epa-badge {epa_details['badge_class']}">{epa_details['category']} ({epa_details['range']})</span>
                    <h3 style="margin-top:10px; margin-bottom:6px; color:#0F172A; font-size:1.4rem;">Estimated Overall AQI: {pred_aqi}</h3>
                    <p style="color:#475569; font-size:0.88rem; line-height:1.5;">{epa_details['summary']}</p>
                    <hr style="border-color: #F1F5F9; margin: 12px 0;">
                    <div style="margin-bottom:8px;">
                        <strong style="color:#0F172A; font-size:0.84rem;">Sensitive Groups Advisory:</strong><br>
                        <span style="color:#64748B; font-size:0.84rem;">{epa_details['sensitive']}</span>
                    </div>
                    <div style="margin-bottom:8px;">
                        <strong style="color:#0F172A; font-size:0.84rem;">General Public Guidance:</strong><br>
                        <span style="color:#64748B; font-size:0.84rem;">{epa_details['general']}</span>
                    </div>
                    <div>
                        <strong style="color:#0F172A; font-size:0.84rem;">Recommended Protective Actions:</strong><br>
                        <span style="color:#64748B; font-size:0.84rem;">{epa_details['action']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ── GRAPH SUITE: ENSEMBLE CONSENSUS & POLAR RADAR PROFILE ─────────────
        st.markdown('<div class="section-title">Model Diagnostics & Atmospheric Profile Suite</div>', unsafe_allow_html=True)
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            # Graph 1: 100-Tree Random Forest Consensus Distribution
            unique_preds = len(np.unique(tree_preds))
            fig_hist = px.histogram(
                x=tree_preds,
                nbins=max(5, min(25, unique_preds if unique_preds > 1 else 10)),
                labels={"x": "Tree Prediction (AQI)", "count": "Decision Trees"},
                color_discrete_sequence=["#2563EB"],
            )
            fig_hist.add_vline(
                x=pred_aqi,
                line_dash="dash",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"Ensemble Mean ({pred_aqi})",
                annotation_position="top right",
                annotation_font=dict(size=10, color="#DC2626"),
            )
            layout_hist = get_chart_layout(
                title=f"100-Tree Ensemble Voting Consensus (σ = {tree_std:.2f}, 95% CI: [{tree_ci_low:.1f}, {tree_ci_high:.1f}])",
                height=340
            )
            layout_hist["xaxis"]["title"] = "Individual Tree Estimated AQI"
            layout_hist["yaxis"]["title"] = "Tree Count (out of 100)"
            fig_hist.update_layout(layout_hist)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"Histogram of all 100 individual decision trees. Narrow dispersion (&sigma; = {tree_std:.2f}) quantifies high model certainty.")

        with g_col2:
            # Graph 2: 4-Axis Pollutant Sub-Index Radar Profile
            rad_cats = ["CO", "Ozone", "NO2", "PM2.5", "CO"]
            rad_vals = [co_input, ozone_input, no2_input, pm25_input, co_input]
            max_rad = max(100.0, max(rad_vals) * 1.15)

            fig_radar = go.Figure()
            # Reference: Good Limit (50)
            fig_radar.add_trace(go.Scatterpolar(
                r=[50, 50, 50, 50, 50],
                theta=rad_cats,
                mode="lines",
                name="EPA Good (50)",
                line=dict(color="#22C55E", dash="dash", width=1.5),
            ))
            # Reference: Moderate Limit (100)
            fig_radar.add_trace(go.Scatterpolar(
                r=[100, 100, 100, 100, 100],
                theta=rad_cats,
                mode="lines",
                name="EPA Moderate (100)",
                line=dict(color="#EAB308", dash="dash", width=1.5),
            ))
            # Current Station Profile
            fig_radar.add_trace(go.Scatterpolar(
                r=rad_vals,
                theta=rad_cats,
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.15)",
                line=dict(color="#2563EB", width=2.5),
                name="Current Inputs",
            ))
            layout_rad = get_chart_layout(title="4-Axis Atmospheric Sub-Index Radar Profile", height=340)
            layout_rad["polar"] = dict(
                radialaxis=dict(visible=True, range=[0, max_rad], tickfont=dict(size=10, color="#64748B"), gridcolor="#F1F5F9"),
                angularaxis=dict(tickfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, sans-serif")),
                bgcolor="#FFFFFF",
            )
            fig_radar.update_layout(layout_rad)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Visualizes the shape of pollution across the 4 axes relative to the EPA Good (50) and Moderate (100) safety rings.")

        # ── GRAPH SUITE: DETERMINANTS WATERFALL & PARTIAL DEPENDENCE ──────────
        g_col3, g_col4 = st.columns(2)

        with g_col3:
            # Graph 3: Sub-Index Determinant Breakdown & Model Comparison
            theory_max = max(co_input, ozone_input, no2_input, pm25_input)
            diff_max = abs(pred_aqi - theory_max)

            bars_labels = ["CO Sub-Index", "Ozone Sub-Index", "NO2 Sub-Index", "PM2.5 Sub-Index", "EPA Max Rule", "RF Prediction"]
            bars_vals = [co_input, ozone_input, no2_input, pm25_input, theory_max, pred_aqi]
            bars_colors = [
                "#94A3B8",
                "#0284C7" if ozone_input == theory_max else "#94A3B8",
                "#D97706" if no2_input == theory_max else "#94A3B8",
                "#DC2626" if pm25_input == theory_max else "#94A3B8",
                "#64748B",
                "#2563EB",
            ]

            fig_breakdown = go.Figure(go.Bar(
                x=bars_vals,
                y=bars_labels,
                orientation="h",
                marker=dict(color=bars_colors, line=dict(color="#0F172A", width=0.5)),
                text=[f"{v:.1f}" for v in bars_vals],
                textposition="auto",
            ))
            layout_bd = get_chart_layout(
                title=f"Sub-Index Determinants vs Model Prediction (Δ to Max = {diff_max:.2f})",
                height=340
            )
            layout_bd["xaxis"]["title"] = "AQI Value Points"
            fig_breakdown.update_layout(layout_bd)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            st.caption(f"Governing pollutant is {gov_name} ({theory_max:.0f}). The Random Forest prediction matches the EPA max rule within {diff_max:.2f} AQI points.")

        with g_col4:
            # Graph 4: Partial Dependence Sensitivity Sweep Curves
            sweep_vals = np.linspace(0, 300, 35)
            base_inputs = {
                "CO AQI Value": co_input,
                "Ozone AQI Value": ozone_input,
                "NO2 AQI Value": no2_input,
                "PM2.5 AQI Value": pm25_input,
            }
            traces = []
            colors_dict = {
                "PM2.5 AQI Value": "#DC2626",
                "Ozone AQI Value": "#0284C7",
                "NO2 AQI Value": "#D97706",
                "CO AQI Value": "#64748B",
            }
            for feat, color in colors_dict.items():
                sim_df = pd.DataFrame([base_inputs] * len(sweep_vals))
                sim_df[feat] = sweep_vals
                sim_preds = model.predict(sim_df[expected_features])
                traces.append(go.Scatter(
                    x=sweep_vals,
                    y=sim_preds,
                    mode="lines",
                    name=feat.replace(" AQI Value", ""),
                    line=dict(color=color, width=2.5),
                ))

            fig_sens = go.Figure(data=traces)
            fig_sens.add_vline(
                x=theory_max,
                line_dash="dot",
                line_color="#0F172A",
                annotation_text=f"Governing Level ({theory_max:.0f})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#0F172A"),
            )
            layout_sens = get_chart_layout(title="Partial Response of AQI to Individual Pollutant Variations", height=340)
            layout_sens["xaxis"]["title"] = "Simulated Pollutant Sub-Index Level"
            layout_sens["yaxis"]["title"] = "Estimated Overall AQI"
            fig_sens.update_layout(layout_sens)
            st.plotly_chart(fig_sens, use_container_width=True)
            st.caption("Partial dependence sweeps showing model response as each pollutant is varied from 0 to 300, holding others fixed.")

        # ── GRAPH SUITE: GLOBAL EMPIRICAL BENCHMARK PERCENTILE ────────────────
        if raw_df is not None and "AQI Value" in raw_df.columns:
            st.markdown('<div class="section-title">Global Context: Empirical Station Placement</div>', unsafe_allow_html=True)
            global_pct = (raw_df["AQI Value"] <= pred_aqi).mean() * 100

            hist_vals, bin_edges = np.histogram(raw_df["AQI Value"].clip(upper=400), bins=60)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            fig_global = go.Figure()
            fig_global.add_trace(go.Scatter(
                x=bin_centers,
                y=hist_vals,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(203, 213, 225, 0.4)",
                line=dict(color="#64748B", width=2),
                name="23,463 Global Stations",
            ))
            fig_global.add_vline(
                x=pred_aqi,
                line_dash="solid",
                line_color="#2563EB",
                line_width=3,
                annotation_text=f"Your Estimate: {pred_aqi} ({global_pct:.1f}th %ile)",
                annotation_position="top right",
                annotation_font=dict(size=11, color="#2563EB", family="Plus Jakarta Sans"),
            )
            layout_glob = get_chart_layout(
                title=f"Position Across 23,463 Global Monitoring Stations (Cleaner than {global_pct:.1f}% of Monitored Stations)",
                height=280
            )
            layout_glob["xaxis"]["title"] = "Empirical Station AQI Value (0–400 Scale)"
            layout_glob["yaxis"]["title"] = "Monitoring Station Frequency"
            fig_global.update_layout(layout_glob)
            st.plotly_chart(fig_global, use_container_width=True)
            st.caption(f"Contextualizes this estimate against {len(raw_df):,} verified global monitoring stations across 175 countries.")


# ==============================================================================
# 8. MODULE 4: MODEL ENGINEERING, BENCHMARKS & BATCH TESTING
# ==============================================================================
elif selected_page == "Model Architecture, Benchmarks & Validation":
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-eyebrow">Engineering Workbench</div>
            <div class="header-title">Model Architecture, Benchmarks & Validation Suite</div>
            <div class="header-subtitle">Comparative model evaluation, 3D decision landscape, and batch CSV inference engine</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="disclosure-banner">
            <strong>Evaluation Architecture:</strong>
            The selected model is a <code>RandomForestRegressor</code> (100 decision trees, random_state=42).
            Below, we review the 4 competing architectures tested during model selection, live empirical validation on all 23,463 records,
            and an integrated custom dataset batch inference suite.
        </div>
    """, unsafe_allow_html=True)

    # ── PRIMARY BENCHMARK CARDS ───────────────────────────────────────────────
    cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
    with cv_c1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Champion Test MAE</div>
                <div class="metric-number metric-number-accent">0.2318</div>
                <div class="metric-caption">Mean Absolute Error (AQI pts)</div>
            </div>
        """, unsafe_allow_html=True)

    with cv_c2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Test Set RMSE</div>
                <div class="metric-number">2.6621</div>
                <div class="metric-caption">Root Mean Squared Error (AQI pts)</div>
            </div>
        """, unsafe_allow_html=True)

    with cv_c3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Holdout R²</div>
                <div class="metric-number metric-number-accent">0.9978</div>
                <div class="metric-caption">Variance explained</div>
            </div>
        """, unsafe_allow_html=True)

    with cv_c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-eyebrow">Within ±5 AQI Pts</div>
                <div class="metric-number metric-number-accent">99.28%</div>
                <div class="metric-caption">High-precision prediction bound</div>
            </div>
        """, unsafe_allow_html=True)

    # ── SECTION: MODEL COMPARISON LEADERBOARD & GINI FEATURE IMPORTANCES ───────
    st.markdown('<div class="section-title">Model Architecture Comparison & Feature Importances</div>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns([1.1, 0.9])

    with m_col1:
        model_benchmarks_df = pd.DataFrame({
            "Architecture": ["Mean Baseline", "Linear Regression", "Gradient Boosting", "Random Forest Regressor"],
            "Mean Absolute Error": [36.9632, 4.8038, 1.5591, 0.2318],
            "Type": ["Baseline", "Parametric", "Boosting Ensemble", "Bagging Ensemble (Selected)"]
        })
        fig_model_comp = px.bar(
            model_benchmarks_df,
            x="Architecture",
            y="Mean Absolute Error",
            color="Mean Absolute Error",
            color_continuous_scale=[[0, "#2563EB"], [0.2, "#0284C7"], [1, "#DC2626"]],
            text="Mean Absolute Error",
            title="Comparison of Tested Architectures: MAE on Held-Out Split (Lower is Better)",
        )
        layout_m = get_chart_layout(height=360)
        layout_m["coloraxis_showscale"] = False
        fig_model_comp.update_layout(layout_m)
        fig_model_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_model_comp, use_container_width=True)

    with m_col2:
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "Feature": expected_features,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text=[f"{v*100:.2f}%" for v in imp_df["Importance"]],
                color="Importance",
                color_continuous_scale=[[0, "#EFF6FF"], [1, "#1E3A8A"]],
                title="Gini Feature Importances (Fitted Random Forest)",
            )
            layout_imp = get_chart_layout(height=360)
            layout_imp["coloraxis_showscale"] = False
            fig_imp.update_layout(layout_imp)
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── SECTION: 3D ATMOSPHERIC DECISION SURFACE ──────────────────────────────
    st.markdown('<div class="section-title">3D Atmospheric Decision Surface (PM2.5 vs Ozone vs Overall AQI)</div>', unsafe_allow_html=True)
    st.caption("Rotate and zoom the 3D surface to view the upper-envelope piecewise boundary modeled by the Random Forest ensemble.")

    sample_3d = raw_df.sample(n=min(1200, len(raw_df)), random_state=42)
    fig_3d = px.scatter_3d(
        sample_3d,
        x="PM2.5 AQI Value",
        y="Ozone AQI Value",
        z="AQI Value",
        color="AQI Value",
        color_continuous_scale=[[0.0, "#22C55E"], [0.25, "#EAB308"], [0.5, "#EF4444"], [1.0, "#881337"]],
        opacity=0.75,
        title="3D Atmospheric Decision Surface",
    )
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#F1F5F9", title="PM2.5 AQI"),
            yaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#F1F5F9", title="Ozone AQI"),
            zaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#F1F5F9", title="Overall AQI"),
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── SECTION: PARITY & RESIDUAL GROUND TRUTH ANALYSIS ──────────────────────
    st.markdown('<div class="section-title">Empirical Parity & Residual Ground Truth Analysis</div>', unsafe_allow_html=True)

    if raw_df is not None and model is not None:
        @st.cache_data
        def run_empirical_validation():
            X = raw_df[expected_features]
            y_true = raw_df["AQI Value"].values
            y_pred = model.predict(X)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            within_5 = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100
            return {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "within_5": within_5,
                "y_true": y_true,
                "y_pred": y_pred,
                "residuals": y_pred - y_true,
            }

        live_metrics = run_empirical_validation()

        r_col1, r_col2 = st.columns(2)
        sample_indices = np.random.RandomState(42).choice(len(raw_df), size=min(2000, len(raw_df)), replace=False)
        sub_sample = pd.DataFrame({
            "Actual AQI": live_metrics["y_true"][sample_indices],
            "Predicted AQI": live_metrics["y_pred"][sample_indices],
            "Residual": live_metrics["residuals"][sample_indices],
        })

        with r_col1:
            fig_parity = px.scatter(
                sub_sample,
                x="Actual AQI",
                y="Predicted AQI",
                color="Residual",
                color_continuous_scale="RdBu_r",
                title="Parity Plot: Actual vs Model Predicted AQI (N=2,000 Sample)",
                opacity=0.6,
            )
            fig_parity.add_trace(go.Scatter(
                x=[0, 500],
                y=[0, 500],
                mode="lines",
                line=dict(color="#0F172A", dash="dash", width=1.5),
                name="1:1 Parity Line",
            ))
            layout_parity = get_chart_layout(height=360)
            fig_parity.update_layout(layout_parity)
            st.plotly_chart(fig_parity, use_container_width=True)

        with r_col2:
            fig_res = px.histogram(
                sub_sample,
                x="Residual",
                nbins=40,
                title="Residual Error Distribution (Predicted − Actual)",
                color_discrete_sequence=["#2563EB"],
            )
            layout_res = get_chart_layout(height=360)
            fig_res.update_layout(layout_res)
            st.plotly_chart(fig_res, use_container_width=True)

    # ── SECTION: CUSTOM DATASET BATCH INFERENCE SUITE ─────────────────────────
    st.markdown('<div class="section-title">Custom Dataset Batch Upload & Verification Engine</div>', unsafe_allow_html=True)

    b_up1, b_up2 = st.columns([1, 1.2])
    with b_up1:
        st.caption("Download the schema template or upload your custom CSV file for automated batch estimation.")
        sample_template = pd.DataFrame([
            {"Station_ID": "ST-01", "City": "Sample City A", "CO AQI Value": 2, "Ozone AQI Value": 34, "NO2 AQI Value": 6, "PM2.5 AQI Value": 52, "AQI Value": 52},
            {"Station_ID": "ST-02", "City": "Sample City B", "CO AQI Value": 8, "Ozone AQI Value": 92, "NO2 AQI Value": 28, "PM2.5 AQI Value": 164, "AQI Value": 164},
            {"Station_ID": "ST-03", "City": "Sample City C", "CO AQI Value": 1, "Ozone AQI Value": 18, "NO2 AQI Value": 3, "PM2.5 AQI Value": 16, "AQI Value": 18},
            {"Station_ID": "ST-04", "City": "Sample City D", "CO AQI Value": 18, "Ozone AQI Value": 130, "NO2 AQI Value": 48, "PM2.5 AQI Value": 245, "AQI Value": 245},
        ])
        tmpl_buffer = io.StringIO()
        sample_template.to_csv(tmpl_buffer, index=False)
        st.download_button(
            label="Download Benchmark CSV Template",
            data=tmpl_buffer.getvalue(),
            file_name="airsight_sample_template.csv",
            mime="text/csv",
        )

    with b_up2:
        uploaded_file = st.file_uploader(
            "Upload custom CSV file for inference",
            type=["csv"],
            help=f"Must contain features: {', '.join(expected_features)}",
        )

    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded `{uploaded_file.name}` ({len(up_df):,} rows, {len(up_df.columns)} columns)")

            col_lookup = {c.strip().lower(): c for c in up_df.columns}
            missing_cols = []
            matched_cols = {}

            for req_f in expected_features:
                if req_f.strip().lower() in col_lookup:
                    matched_cols[req_f] = col_lookup[req_f.strip().lower()]
                else:
                    missing_cols.append(req_f)

            has_ground_truth = False
            actual_aqi_col = None
            for candidate_aqi in ["aqi value", "aqi", "actual aqi", "recorded aqi"]:
                if candidate_aqi in col_lookup:
                    has_ground_truth = True
                    actual_aqi_col = col_lookup[candidate_aqi]
                    break

            if missing_cols:
                st.error(f"Missing required features: `{missing_cols}`")
            else:
                st.success("Schema verified. All 4 pollutant features present.")
                if has_ground_truth:
                    st.info(f"Ground Truth detected in column `{actual_aqi_col}`. Prediction performance will be calculated.")

                if st.button("Run Batch AQI Estimation", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        X_batch = up_df[[matched_cols[f] for f in expected_features]].copy()
                        X_batch.columns = expected_features

                        for f in expected_features:
                            X_batch[f] = pd.to_numeric(X_batch[f], errors="coerce").fillna(0)

                        batch_preds = model.predict(X_batch)
                        batch_preds = np.clip(np.round(batch_preds, 1), 0, 500)

                        output_df = up_df.copy()
                        output_df["Estimated_AQI"] = batch_preds
                        output_df["EPA_Category"] = [get_epa_category_details(p)["category"] for p in batch_preds]

                        st.success(f"Batch prediction complete for {len(output_df):,} records!")

                        if has_ground_truth:
                            y_actual = pd.to_numeric(output_df[actual_aqi_col], errors="coerce")
                            valid_mask = ~y_actual.isna()

                            if valid_mask.sum() > 0:
                                y_act_clean = y_actual[valid_mask].values
                                y_pred_clean = batch_preds[valid_mask]

                                b_mae = np.mean(np.abs(y_act_clean - y_pred_clean))
                                b_rmse = np.sqrt(np.mean((y_act_clean - y_pred_clean) ** 2))
                                b_ss_res = np.sum((y_act_clean - y_pred_clean) ** 2)
                                b_ss_tot = np.sum((y_act_clean - np.mean(y_act_clean)) ** 2)
                                b_r2 = 1 - (b_ss_res / b_ss_tot) if b_ss_tot > 0 else 1.0

                                eval_c1, eval_c2, eval_c3 = st.columns(3)
                                with eval_c1:
                                    st.metric("Batch Test MAE", f"{b_mae:.3f} pts")
                                with eval_c2:
                                    st.metric("Batch Test RMSE", f"{b_rmse:.3f} pts")
                                with eval_c3:
                                    st.metric("Batch Test R²", f"{b_r2:.4f}")

                        st.dataframe(output_df, use_container_width=True)
                        out_csv_buffer = io.StringIO()
                        output_df.to_csv(out_csv_buffer, index=False)
                        st.download_button(
                            label="Download Processed Predictions CSV",
                            data=out_csv_buffer.getvalue(),
                            file_name=f"airsight_predictions_{uploaded_file.name}",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")


# ==============================================================================
# 9. MODULE 5: SCIENTIFIC METHODOLOGY & MATHEMATICAL PROOF
# ==============================================================================
elif selected_page == "Scientific Methodology & Mathematical Proof":
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-eyebrow">Scientific Integrity</div>
            <div class="header-title">Scientific Methodology & Mathematical Proof</div>
            <div class="header-subtitle">Empirical investigation into dataset structure, the maximum-sub-index rule, and temporal limits</div>
        </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: THE MAXIMUM SUB-INDEX RULE DISCOVERY ───────────────────────
    st.markdown('<div class="section-title">The Maximum Sub-Index Finding (99.4% Adherence)</div>', unsafe_allow_html=True)
    m_row1, m_row2 = st.columns([1.1, 0.9])

    with m_row1:
        st.markdown("""
            <div class="content-card">
                <div class="metric-eyebrow">Core Analytical Finding</div>
                <div class="metric-number metric-number-accent">99.4% Match</div>
                <p style="font-size:0.88rem; color:#334155; line-height:1.6; margin-top:10px;">
                    In <strong>23,322 out of 23,463 records</strong> (99.4%), the recorded overall AQI Value exactly equals the
                    <strong>maximum of the four pollutant sub-indices</strong>:
                </p>
                <div style="background:#F1F5F9; border-radius:6px; padding:10px 14px; font-family:'JetBrains Mono',monospace; font-size:0.88rem; color:#0F172A; margin:12px 0;">
                    Overall AQI = max(CO AQI, Ozone AQI, NO2 AQI, PM2.5 AQI)
                </div>
                <p style="font-size:0.84rem; color:#64748B; line-height:1.6;">
                    Only <strong>141 records</strong> deviate from this maximum rule. This transparent finding explains why
                    Random Forest achieves a near-perfect R² of 0.9978: the model has effectively learned the piecewise-maximum function governing official composite AQI calculation.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with m_row2:
        rule_counts = pd.DataFrame({
            "Compliance": ["Adheres to Max Rule (23,322)", "Deviates from Max Rule (141)"],
            "Count": [23322, 141]
        })
        fig_rule_pie = px.pie(
            rule_counts,
            names="Compliance",
            values="Count",
            hole=0.62,
            color="Compliance",
            color_discrete_map={
                "Adheres to Max Rule (23,322)": "#2563EB",
                "Deviates from Max Rule (141)": "#DC2626",
            },
            title="Max-Sub-Index Rule Adherence across 23,463 Records",
        )
        fig_rule_pie.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=11),
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_rule_pie, use_container_width=True)

    # ── SECTION 2: COMPARISON TABLE: MAX RULE VS MACHINE LEARNING ─────────────
    st.markdown('<div class="section-title">Comparative Performance: Heuristic Max Rule vs Random Forest</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame([
        {"Method": "Simple Heuristic Max Rule", "Test MAE": "0.3516 pts", "Test RMSE": "4.9316 pts", "Test R²": "0.9924", "Interpretability": "100% Deterministic (Zero Parameters)"},
        {"Method": "Trained Random Forest Regressor", "Test MAE": "0.2318 pts", "Test RMSE": "2.6621 pts", "Test R²": "0.9978", "Interpretability": "100 Trees Ensemble (Learned Boundary)"},
    ])
    st.dataframe(comp_df, use_container_width=True)

    # ── SECTION 3: TEMPORAL BOUNDARIES & FORECASTING TRANSPARENCY ─────────────
    st.markdown('<div class="section-title">Diagnostic Estimation vs Chronological Forecasting</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)

    with t1:
        st.markdown("""
            <div class="content-card" style="border-left: 4px solid #2563EB;">
                <h4 style="color:#0F172A; margin-top:0; font-size:1.05rem;">What AirSight AI Establishes</h4>
                <ul style="color:#475569; font-size:0.86rem; line-height:1.7;">
                    <li><strong>Inputs</strong>: Standardized sub-indices for CO, Ozone, NO2, and PM2.5.</li>
                    <li><strong>Function</strong>: Synthesizes multi-pollutant non-linear relationships to identify the governing composite AQI.</li>
                    <li><strong>Validation</strong>: Evaluated rigorously on 4,693 held-out test stations and 5-fold cross-validation.</li>
                    <li><strong>Designation</strong>: Diagnostic Sub-Index Regressor.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
            <div class="content-card" style="border-left: 4px solid #D97706;">
                <h4 style="color:#0F172A; margin-top:0; font-size:1.05rem;">What AirSight AI Does NOT Claim</h4>
                <ul style="color:#475569; font-size:0.86rem; line-height:1.7;">
                    <li><strong>No Tomorrow's Prediction</strong>: We do not claim to forecast future air quality for tomorrow or next week.</li>
                    <li><strong>No Fabricated Timelines</strong>: The Kaggle dataset contains no temporal timestamps; no synthetic dates are created.</li>
                    <li><strong>Missing Temporal Variables</strong>: Chronological forecasting requires time-series sensor streams, meteorological vectors (wind, boundary layer height, inversions), and emission schedules.</li>
                    <li><strong>Designation</strong>: Spatial Cross-Sectional Benchmark.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Scenario Simulation Reference Date</div>', unsafe_allow_html=True)
    st.caption("Selecting a date below serves solely as a scenario annotation. The underlying Random Forest model does not take dates as an input feature.")

    sim_date = st.date_input("Simulation Reference Date")
    st.info(f"Selected reference date: **{sim_date.strftime('%B %d, %Y')}**. Notice: The model inputs are strictly `[CO, Ozone, NO2, PM2.5]`. Date selection does not alter estimated AQI.")


# ==============================================================================
# 10. GLOBAL FOOTER
# ==============================================================================
st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; color:#94A3B8; font-size:0.78rem; padding: 18px 0; border-top: 1px solid #E2E8F0; margin-top: 36px;">
        <div>AirSight AI &copy; 2026 &bull; Environmental AQI Intelligence Platform</div>
        <div>Standard: US-EPA 40 CFR Part 58 &bull; Scikit-Learn &amp; Plotly</div>
    </div>
""", unsafe_allow_html=True)
