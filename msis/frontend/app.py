import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Market Shock Intelligence System",
    layout="wide"
)

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent.parent / "ml_outputs" / "regimes.csv"
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()
    return pd.read_csv(data_path)


df = load_data()

# ---------------------------------
# Regime Labels + Colors
# ---------------------------------
REGIME_META = {
    0: {"label": "Low Volatility / Stable Market", "color": "#1f77b4"},
    1: {"label": "Trend-Driven Growth Regime", "color": "#2ecc71"},
    2: {"label": "Crisis / Stress Regime", "color": "#e74c3c"}
}

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("Market Controls")

selected_regime = st.sidebar.selectbox(
    "Select Market Regime",
    options=sorted(df["regime"].unique()),
    format_func=lambda x: REGIME_META[x]["label"]
)

regime_color = REGIME_META[selected_regime]["color"]
regime_label = REGIME_META[selected_regime]["label"]

# ---------------------------------
# Dynamic CSS
# ---------------------------------
st.markdown(
    f"""
    <style>
    .regime-title {{
        color: {regime_color};
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }}

    .left-bar {{
        border-left: 6px solid {regime_color};
        padding-left: 14px;
        margin-top: 10px;
        margin-bottom: 20px;
    }}

    .sidebar-indicator {{
        background-color: {regime_color}20;
        border-left: 5px solid {regime_color};
        padding: 10px;
        font-weight: 600;
        border-radius: 4px;
        margin-top: 10px;
    }}

    section[data-testid="stSidebar"] select {{
        border: 2px solid {regime_color} !important;
        border-radius: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Header
# ---------------------------------
st.markdown(
    "<h2 style='margin-bottom:4px;'>Market Shock Intelligence System (MSIS)</h2>",
    unsafe_allow_html=True
)

st.caption(
    "Unsupervised detection of market regimes using returns, volatility, and drawdowns."
)

# ---------------------------------
# Sidebar Indicator
# ---------------------------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-indicator">
        Selected Regime<br>
        {regime_label}
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Main Regime Section
# ---------------------------------
st.markdown(
    f"""
    <div class="left-bar">
        <div class="regime-title">{regime_label}</div>
    </div>
    """,
    unsafe_allow_html=True
)

regime_df = df[df["regime"] == selected_regime]

# ---------------------------------
# Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Average Daily Return", f"{regime_df['return'].mean():.4f}")
col2.metric("Average Volatility", f"{regime_df['volatility'].mean():.4f}")
col3.metric("Average Drawdown", f"{regime_df['drawdown'].mean():.4f}")

# ---------------------------------
# Feature Distributions (BAR OUTLINES ONLY)
# ---------------------------------
st.subheader("Regime Feature Distributions")

def histogram_with_bar_edges(df, x, title):
    fig = px.histogram(
        df,
        x=x,
        nbins=40,
        title=title,
        opacity=0.8,
        color_discrete_sequence=[regime_color]
    )
    fig.update_traces(
        marker_line_width=0.6,
        marker_line_color="rgba(0,0,0,0.6)"
    )
    return fig

c1, c2, c3 = st.columns(3)

with c1:
    st.plotly_chart(
        histogram_with_bar_edges(regime_df, "return", "Return Distribution"),
        width="stretch",
    )

with c2:
    st.plotly_chart(
        histogram_with_bar_edges(regime_df, "volatility", "Volatility Distribution"),
        width="stretch"
    )

with c3:
    st.plotly_chart(
        histogram_with_bar_edges(regime_df, "drawdown", "Drawdown Distribution"),
        width="stretch"
    )

# ---------------------------------
# Interpretation
# ---------------------------------
st.subheader("Regime Interpretation")

if selected_regime == 0:
    st.write("This regime represents calm market periods with low volatility and shallow drawdowns.")
elif selected_regime == 1:
    st.write("This regime captures sustained growth trends with positive returns and moderate volatility.")
else:
    st.write("This regime reflects market stress periods characterized by high volatility and deep drawdowns.")
