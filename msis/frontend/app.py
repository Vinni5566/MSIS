import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")

# ---------------------------------
# Load Regime Data
# ---------------------------------
@st.cache_data
def load_data():
    path = Path(__file__).resolve().parent.parent / "ml_outputs" / "regimes.csv"
    if not path.exists():
        st.error(f"Regime data not found: {path}")
        st.stop()
    return pd.read_csv(path)

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
# Sidebar: Market Regime Selection
# ---------------------------------

st.markdown(
    """
    <style>
    /* Remove all top padding/margin in sidebar */
    section[data-testid="stSidebar"] .css-1d391kg {  /* main sidebar inner container */
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* Optional: remove extra spacing before first widget */
    section[data-testid="stSidebar"] div.stSelectbox, 
    section[data-testid="stSidebar"] div.stSlider, 
    section[data-testid="stSidebar"] div.stButton {
        margin-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("Market Controls")
selected_regime = st.sidebar.selectbox(
    "Select Market Regime",
    options=sorted(df["regime"].unique()),
    format_func=lambda x: REGIME_META[x]["label"]
)

# Manual reload (clears cache if CSV updated)
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    
# Assign label and color after selection
regime_label = REGIME_META[selected_regime]["label"]
regime_color = REGIME_META[selected_regime]["color"]

# ---------------------------------
# Dynamic CSS for selected regime
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
st.markdown("<h1 style='margin-bottom:4px;'>Market Shock Intelligence System (MSIS)</h1>", unsafe_allow_html=True)
st.caption("Unsupervised detection of market regimes using returns, volatility, and drawdowns.")

# Sidebar indicator
st.sidebar.markdown(f"""
<div class="sidebar-indicator">
    Selected Regime<br>
    {regime_label}
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------
# Main Regime Section
# ---------------------------------
st.markdown(f"""
<div class="left-bar">
    <div class="regime-title">{regime_label}</div>
</div>
""", unsafe_allow_html=True)

regime_df = df[df["regime"] == selected_regime]

# ---------------------------------
# Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Average Daily Return", f"{regime_df['return'].mean():.4f}")
col2.metric("Average Volatility", f"{regime_df['volatility'].mean():.4f}")
col3.metric("Average Drawdown", f"{regime_df['drawdown'].mean():.4f}")

# ---------------------------------
# Feature Distributions (with bar outlines)
# ---------------------------------
st.subheader("Regime Feature Distributions")

def histogram_with_bar_edges(df, x, title):
    fig = px.histogram(
        df, x=x, nbins=40, title=title, opacity=0.8, color_discrete_sequence=[regime_color]
    )
    fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(0,0,0,0.6)")
    return fig

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(histogram_with_bar_edges(regime_df, "return", "Daily Return Distribution"))
with c2:
    st.plotly_chart(histogram_with_bar_edges(regime_df, "volatility", "Volatility Distribution"))
with c3:
    st.plotly_chart(histogram_with_bar_edges(regime_df, "drawdown", "Drawdown Distribution"))

# ---------------------------------
# Explanation for Users
# ---------------------------------
st.subheader("Understanding This Regime")

st.markdown(f"""
- **Daily Return**: <span style='color:#2ecc71'>Positive = gains</span>, <span style='color:#e74c3c'>Negative = losses</span>
- **Volatility**: <span style='color:#f39c12'>Higher = more uncertainty</span>
- **Drawdown**: <span style='color:#e74c3c'>Peak-to-trough loss</span>, indicates risk exposure

**Interpretation for this regime**:
""", unsafe_allow_html=True)

if selected_regime == 0:
    st.write("- Calm market: returns are stable, volatility is low, and drawdowns are shallow.")
elif selected_regime == 1:
    st.write("- Growth trend: positive returns dominate, volatility is moderate, drawdowns are controlled.")
else:
    st.write("- Crisis/stress: high volatility, frequent losses, and deep drawdowns — risk is elevated.")

st.markdown("""
> These visualizations help users quickly understand market conditions and risk in the selected regime.  
> They provide context for strategy performance, risk management, and decision-making.
""")


# ---------------------------------
# Drawdown Risk Predictor
# ---------------------------------
st.markdown("---")
st.subheader("Next-Day Drawdown Risk Insights")

# Short explanation for users
st.info("""
**What is a drawdown?**  
Drawdown measures how much a strategy or market has fallen from a previous peak.  
- Example: If your portfolio goes from $100 → $90, the drawdown is 10%.  
High drawdowns indicate higher risk periods in the market.

**How this model helps:**  
The model predicts the probability of significant drawdowns for the next day based on historical market conditions.  
This helps traders and risk managers identify potential stress periods and adjust positions accordingly.
""")

# Load Drawdown Risk CSV
@st.cache_data
def load_drawdown_data():
    path = Path(__file__).resolve().parent.parent / "ml_outputs" / "drawdown_risk.csv"
    if not path.exists():
        st.error(f"Drawdown risk file not found: {path}")
        st.stop()
    return pd.read_csv(path)

risk_df = load_drawdown_data()

# Sidebar: filter by regime
selected_risk_regime = st.sidebar.selectbox(
    "Select regime for Drawdown Risk Insights",
    options=sorted(risk_df["regime"].unique())
)
filtered_risk_df = risk_df[risk_df['regime'] == selected_risk_regime]

# Compute metrics
avg_risk = filtered_risk_df['risk_probability'].mean()
high_risk_days = (filtered_risk_df['drawdown_risk'].sum() / len(filtered_risk_df))
col1, col2 = st.columns(2)
col1.metric("Average Next-Day Risk Probability", f"{avg_risk:.2f}")
col2.metric("Proportion of High-Risk Days", f"{high_risk_days:.0%}")

# Line chart: Predicted risk over time
st.markdown(f"""
<div style="padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px; background-color:{regime_color}10;">
<b>Predicted Drawdown Risk Over Time:</b>
<p style="margin-top:5px;">
The line chart shows the predicted probability of a significant drawdown each day.<br>
Higher values indicate increased likelihood of losses, allowing users to monitor potential market stress.
</p>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(px.line(
    filtered_risk_df, y='risk_probability',
    title=f"Predicted Drawdown Risk Over Time — {selected_risk_regime} Regime",
    color_discrete_sequence=[REGIME_META[selected_regime]["color"]],
    labels={'y': 'Risk Probability'},
    hover_data={"drawdown_risk": True}
), use_container_width=True)

# Histogram: Distribution of predicted risks
st.markdown(f"""
<div style="padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px; background-color:{regime_color}10;">
<b>Distribution of Predicted Risk:</b>
<ul style="margin-top:5px;">
    <li>Most values near 0 → generally stable market</li>
    <li>Many values near 1 → frequent high-risk days</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(px.histogram(
    filtered_risk_df, x='risk_probability', nbins=30,
    title=f"Distribution of Predicted Risk — {selected_risk_regime} Regime",
    color_discrete_sequence=[REGIME_META[selected_regime]["color"]]
), use_container_width=True)

# Collapsible section: Model features
with st.expander("Model Features / Inputs for Drawdown Prediction"):
    st.markdown("""
    - **Recent Returns & Volatility**: Capture short-term market movements  
    - **Recent Drawdowns**: Past losses help predict near-future risk  
    - **Regime Information**: Identifies whether the market is calm, trending, or in stress  
    - **Rolling Statistics (Momentum, Volatility)**: Capture local patterns and market shocks
    """)

# Insights
st.markdown("**Insights:**")
if avg_risk > 0.7:
    st.write("High predicted likelihood of drawdowns tomorrow. Consider caution in trading or reducing exposure.")
elif avg_risk > 0.4:
    st.write("Moderate risk. Monitor positions carefully; market may be unstable.")
else:
    st.write("Low predicted risk. Market appears stable for the next day.")

# ---------------------------------
# Strategy Failure Predictor
# ---------------------------------
st.markdown("---")
st.subheader("Strategy Failure Insights")

# Explanation for users
st.info("""
**What is a Strategy Failure?**  
A strategy failure occurs when the trading strategy experiences a large negative return (loss) the next day.  
- Example: If the strategy loses more than the bottom 5% of daily returns, it is considered a failure.  

**How this helps:**  
By predicting next-day failure probability, users can anticipate risk periods and adjust their positions or hedge strategies.  
""")

# Model Overview
st.markdown(f"""
<div style="background-color:{regime_color}10; padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px;">
<b>Models Overview:</b><br>
- <span style='color:#1f77b4'>Logistic Regression</span>: Linear model predicting next-day failure based on historical features<br>
- <span style='color:#2ecc71'>Random Forest</span>: Tree-based model capturing non-linear patterns<br>
- <span style='color:#e74c3c'>XGBoost</span>: Gradient boosted trees optimized for rare events
</div>
""", unsafe_allow_html=True)


# Load CSV
@st.cache_data
def load_strategy_failure():
    path = Path(__file__).resolve().parent.parent / "ml_outputs" / "strategy_failure.csv"
    if not path.exists():
        st.error(f"Strategy failure output file not found: {path}")
        st.stop()
    return pd.read_csv(path, parse_dates=["Date"])

strategy_df = load_strategy_failure()

# Sidebar selections
selected_sf_regime = st.sidebar.selectbox(
    "Select regime for Strategy Failure Insights",
    options=sorted(strategy_df["regime"].unique())
)
selected_model = st.sidebar.selectbox(
    "Choose model",
    options=["Logistic Regression", "Random Forest", "XGBoost"]
)

filtered_sf_df = strategy_df[strategy_df["regime"] == selected_sf_regime]

# Key Metrics
avg_failure_prob = filtered_sf_df[f"failure_prob_{selected_model}"].mean()
avg_strategy_ret = filtered_sf_df["strategy_ret"].mean()
high_risk_days = (filtered_sf_df["failure"] == 1).mean()

col1, col2, col3 = st.columns(3)
col1.metric("Avg. Predicted Next-Day Failure Probability", f"{avg_failure_prob:.2f}")
col2.metric("Avg. Strategy Return", f"{avg_strategy_ret:.4f}")
col3.metric("% of Days Strategy Actually Failed", f"{high_risk_days:.0%}")

# Failure probability trend
st.markdown(f"""
<div style="background-color:{regime_color}10; padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px;">
<b>Predicted Failure Probability Over Time:</b><br>
- Line chart shows probability of large loss each day<br>
- Peaks = higher expected risk<br>
- Hover to see actual strategy return & failures
</div>
""", unsafe_allow_html=True)

st.plotly_chart(px.line(
    filtered_sf_df, x="Date", y=f"failure_prob_{selected_model}",
    title=f"{selected_model} Predicted Failure Probability — Regime {selected_sf_regime}",
    color_discrete_sequence=[REGIME_META[selected_regime]["color"]],
    labels={"failure_prob_"+selected_model: "Predicted Failure Probability"},
    hover_data=["strategy_ret", "failure"]
), use_container_width=True)

# Strategy returns distribution
st.markdown(f"""
<div style="background-color:{regime_color}10; padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px;">
<b>Strategy Returns Distribution:</b><br>
- Boxplot shows daily strategy returns<br>
- Points represent individual days<br>
- Compare returns vs predicted failure probability to validate model
</div>
""", unsafe_allow_html=True)

st.plotly_chart(px.box(
    filtered_sf_df, x="regime", y="strategy_ret", points="all",
    title=f"Strategy Returns Distribution — Regime {selected_sf_regime}",
    color_discrete_sequence=[REGIME_META[selected_regime]["color"]],
    labels={"strategy_ret": "Strategy Return"}
), use_container_width=True)

# Model features (optional, collapsible)
with st.expander("Model Features / Inputs for Strategy Failure Prediction"):
    st.markdown("""
    - **Return & Volatility**: Measures recent market movements  
    - **Drawdowns**: Recent losses indicating market stress  
    - **Regime Information**: Captures calm, growth, or stress periods  
    - **Rolling Momentum & Lagged Returns**: Detects short-term trends  
    - **Volatility Ratio & Drawdown Change**: Captures sudden market shocks
    """)

# Insights / Interpretation
st.markdown(f"""
<div style="background-color:{regime_color}10; padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px;">
<b>Insights:</b><br>
<ul style="margin-top:5px;">
    <li style="color:#e74c3c;">High risk → reduce exposure</li>
    <li style="color:#f39c12;">Moderate risk → monitor carefully</li>
    <li style="color:#2ecc71;">Low risk → market stable</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ---------------------------------
# Strategy Risk Analysis Visualizations
# ---------------------------------
st.markdown("---")
st.subheader("Strategy Risk Analysis Visualizations")

st.markdown("""
This section provides visual insights into strategy risk over time and across market regimes.  
- **Heatmap**: Shows predicted failure probability for each regime over time.  
- **Scatter Plot**: Shows relationship between predicted failure probability and actual strategy returns.
""")

# 1️⃣ Failure Probability Heatmap by Regime vs. Time
st.markdown("### Failure Probability Heatmap by Regime vs. Time")
heatmap_df = strategy_df.pivot(index="Date", columns="regime", values=f"failure_prob_{selected_model}")
fig_heatmap = px.imshow(
    heatmap_df.T,  # transpose to have regimes on y-axis
    labels=dict(x="Date", y="Regime", color="Failure Prob"),
    x=heatmap_df.index,
    y=[REGIME_META[i]["label"] for i in heatmap_df.columns],
    color_continuous_scale="RdYlGn_r",
    aspect="auto"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown(f"""
<div style="padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px; background-color:{regime_color}10;">
<b>Interpretation:</b>
<ul style="margin-top:5px; margin-left:15px;">
<li>Darker colors indicate higher predicted failure probability.</li>
<li>Helps identify periods when specific market regimes are riskier for the strategy.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# 2️⃣ Strategy Return vs. Predicted Failure Probability Scatter Plot
st.markdown("### Strategy Return vs. Predicted Failure Probability")
fig_scatter = px.scatter(
    filtered_sf_df,
    x=f"failure_prob_{selected_model}",
    y="strategy_ret",
    color=filtered_sf_df["regime"].map(lambda x: REGIME_META[x]["label"]),
    color_discrete_sequence=[REGIME_META[i]["color"] for i in sorted(filtered_sf_df["regime"].unique())],
    hover_data=["Date", "regime"],
    labels={
        f"failure_prob_{selected_model}": "Predicted Failure Probability",
        "strategy_ret": "Strategy Return"
    }
)
fig_scatter.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode="markers"))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(f"""
<div style="padding:10px; border-left:5px solid {regime_color}; border-radius:5px; margin-bottom:10px; background-color:{regime_color}10;">
<b>Interpretation:</b>
<ul style="margin-top:5px; margin-left:15px;">
<li>Points toward the top-right indicate days with high predicted failure probability and negative returns.</li>
<li>This helps users visually validate model predictions and identify risky periods.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Collapsible Feature Info
# ---------------------------------
with st.expander("Model Features / Inputs"):
    st.markdown("""
    - **Volatility, Drawdown, Regime**: Core market conditions  
    - **Rolling Momentum & Lagged Returns**: Capture short-term trends  
    - **Rolling Volatility / Drawdown Stats**: Capture local risk patterns  
    - **Strategy Signals / Returns**: Connect market regime to strategy performance  
    - **Regime Shift Indicator**: Detect transitions between calm, trend, or crisis markets  
""")
