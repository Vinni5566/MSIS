# Market Shock Intelligence System (MSIS)

A market intelligence web application that uses machine learning to analyze historical financial data and explain how market risk, volatility, and model behavior change across different market conditions.

---

## 🚩 Problem 

Financial markets generate massive amounts of data, but during periods of stress, understanding *market behavior* and *risk dynamics* is often more valuable than point price predictions.

Traditional ML models often:
- Treat all market periods as identical
- Break down during crises
- Offer little explanation of risk behavior

This project focuses on **quantifying market regimes** and understanding **when and why markets behave differently**.

---

## 🎯 Project Objective

To build an explainable market intelligence system that:
- Identifies different market regimes (calm, stressed, crisis)
- Analyzes regime transitions and volatility buildup
- Evaluates how models and strategies perform across regimes
- Emphasizes interpretation over raw prediction accuracy

---

## ❓ Key Questions We Answer

- Can historical market data be grouped into distinct behavioral regimes?
- How does volatility evolve before market stress events?
- How do simple models and strategies behave across regimes?
- When do predictive models fail, and why?
  

---

## 🧠 Models Overview

| **Model**                               | **Core Question**                    | **Approach**                                              | **Key Output**                                        | **Why It Matters**                                              |
| --------------------------------------- | ------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| **Model 1: Market Regime Discovery**    | What types of market behavior exist? | KMeans clustering on **returns, volatility, drawdown**    | **Regime label per day** (calm / stressed / crisis)   | Provides **structural context** for all downstream analysis     |
| **Model 2A: Drawdown Risk Prediction**  | How risky is the next trading day?   | **Logistic Regression** with **regime-aware features**    | **Probability of next-day drawdown**                  | Shows how risk **changes across regimes** in an explainable way |
| **Model 2B: Strategy Failure Analysis** | When do models & strategies fail?    | Logistic, Random Forest, **XGBoost** + imbalance handling | **Failure probability** & regime-wise stress behavior | Evaluates **robustness**, not just prediction accuracy          |



---

## 🏗️ System Design

MSIS is built as a **decoupled analytics system**:

```
Offline Analysis Engine (Python)
│
├─ Data ingestion
├─ Feature computation
├─ Regime detection
└─ Stress & robustness analysis
        ↓
Precomputed Outputs
│
├─ CSV files
└─ JSON files
        ↓
Frontend Dashboard (Streamlit)
│
├─ Interactive charts
├─ Regime exploration
└─ Insight explanation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|----|----|
| Data | CSV (Yahoo Finance / Kaggle) |
| Feature Engineering | Python (pandas, numpy) |
| Analysis & ML | Python (scikit-learn: KMeans, Logistic Regression, Random Forest, XGBoost) |
| Visualization | matplotlib, seaborn |
| Frontend & App Layer | Streamlitt |
| Charts | Plotly |
| Deployment | Streamlit |

---

## 📦 Project Structure
```
MSIS/
│
├── msis/                         # Core application package
│   │
│   ├── ml_models/                 # Model research & experimentation
│   │   ├── strategy_failure_predictor.ipynb
│   │   ├── regime_exploration.ipynb
│   │   └── drawdown_risks_predictor.ipynb
│   │
│   ├── frontend/                 # Frontend visualization layer
│   │   ├── app.py                # Streamlit dashboard
│   │   └── streamlit_dashboard.md
│   │
│   └── outputs/                  # Persisted model outputs
│       ├── strategy_failure_predictor.csv
│       ├── strategy_failure_predictor.json
│       ├── regimes.csv
│       └── regimes.json
│
├── LICENSE
├── README.md                     # Project overview & usage
├── requirements.txt              # Python dependencies
└── runtime.txt                   # Deployment/runtime configuration
 
```

---

## 🌍 Real-World Relevance

This system reflects how:
- Financial institutions analyze risk
- Hedge funds perform regime-aware modeling
- Regulators study market stress

The focus is on **understanding uncertainty**, not short-term trading.


