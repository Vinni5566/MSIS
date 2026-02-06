## 📊 ML MODEL 1: Market Regime Detection

**What it does**: Learns hidden market states from data.
> “Is the market behaving like a calm bull, volatile sideways, or crisis mode?”

**Type**: Unsupervised ML

**Examples**
- Gaussian Mixture Model (GMM)
- KMeans
- HMM as variants

**Input**
- Returns
- Volatility
- Drawdown

**Output**
- Regime label per day
- Regime probabilities

No regime labels exist in real markets, making unsupervised learning necessary.

---

## 📊 ML MODEL 2: Strategy Performance Prediction / Failure Detection

**What it does**: Learns when a strategy/model will fail, conditioned on market regime.
> “Given the current regime, how risky is this strategy?”

**Type**: Supervised ML

**Inputs**
- Regime label (from Model 1)
- Volatility
- Trend strength
- Momentum indicators

**Target (any ONE)**
- Next-day drawdown risk (binary)
- Strategy return bucket
- Error magnitude

**Model Options**
- Logistic Regression (clean & explainable)
- Random Forest (non-linear)
- XGBoost (if confident)

---

## 🔗 How They Connect (Important)

Model Dependency & Pipeline Flow

```bash
Market Data
   ↓
Feature Engineering
   ↓
ML Model 1 → Market Regimes
   ↓
Regime-aware Features
   ↓
ML Model 2 → Risk / Failure Prediction
```

This forms a **hierarchical ML pipeline**, where the output of the unsupervised model directly conditions the supervised learning task.

## 📚

> Built a two-stage machine learning pipeline.
>
> The first model identifies hidden market regimes using unsupervised learning.
>
> The second model uses those regimes as features to predict strategy risk and failure probability.

