# 💻Technology
A market intelligence dashboard that uses an ML pipeline to detect market regimes from historical data and explains how risk, volatility, and model performance change across different market conditions.

---
# 🏗️ Project Flow/ Low Level Architecture

```
Market Data (CSV, Yahoo Finance)
      ↓
Feature Engineering (returns, volatility, trends, rolling stats, drawdowns)
      ↓
Market Regime Detection (ML)
      ↓
Regime Transition & Model Robustness Analysis (ML)
      ↓
Insights + Visualizations (plots, overlays, regime map)
      ↓
Dashboard / Report UI (ttabs, filters, isights, conclusions)
```
---
# 📚Project Layers

| Layer           | What it does            |
| --------------- | ----------------------- |
| Data layer      | Load, clean, transform  |
| Analytics layer | Stats, trends           |
| ML layer        | Clustering / prediction |
| Insight layer   | Interpretation          |
| UI layer        | Visualization           |

## 1️⃣ Data Layer
- Static financial data
- Format: CSV / Parquet
- Source: Yahoo Finance / Kaggle
- Lives locally in repo

> No databases. No real-time feeds.

---

## 2️⃣ Analysis & ML Layer (Offline)

- Python scripts + notebooks
- Runs once (or occasionally)
- Outputs saved results

### Examples of outputs:
- Regime labels per date
- Volatility metrics
- Strategy performance tables

### Saved as:
- CSV
- JSON

> This is NOT a service. It’s an engine.

---

## 3️⃣ API Layer (Thin, Dumb)

- FastAPI
- Purpose: serve precomputed results

### What it does:
- /market-overview
- /regimes
- /transitions
- /stress-analysis

### What it does NOT do:
- Train models
- Compute ML
- Authenticate users

> Think of it as a file reader with HTTP

---

## 4️⃣ Frontend Layer (Product)

- Next.js (React)
- Tailwind CSS
- Charting library

### What it does:
- Loads data from API
- Renders charts
- Explains insights
- Controls UX flow

---

# 🛠️ Tech Stack & Responsibility

| System Layer                  | Responsibility                              | Tech Used                          | Why This Choice                 |
| ----------------------------- | ------------------------------------------- | ---------------------------------- | ------------------------------- |
| **Data Layer**                | Store historical market data                | CSV (Yahoo Finance / Kaggle)       | Free, transparent, reproducible |
| **Analysis Engine (Offline)** | Clean data, compute metrics, validate logic | Python (pandas, numpy, matplotlib) | Fast iteration, correctness     |
| **ML Engine (Offline)**       | Detect market regimes                       | scikit-learn (KMeans / GMM / HMM)  | Interpretable, stable           |
| **Insight Engine (Offline)**  | Transitions, robustness, stress analysis    | Python scripts                     | Quantitative reasoning          |
| **Output Layer**              | Persist computed results                    | CSV / JSON                         | Decouples ML from UI            |
| **Backend API**               | Serve precomputed insights                  | FastAPI                            | Thin, predictable               |
| **Frontend Dashboard**        | Interactive charts & UX                     | Next.js + React                    | Real product UI                 |
| **Charting (Frontend)**       | Visualize time-series & regimes             | Recharts / Chart.js / ECharts      | Interactive & fast              |
| **Deployment (Demo)**         | Run locally                                 | localhost                          | Zero infra risk                 |
