## ML Model 1: Market Regime Detection (Unsupervised)

This notebook implements the first stage of MSIS: detecting hidden market regimes
from historical S&P 500 data using unsupervised learning.

### What the model does
- Computes daily market behavior metrics:
  - Returns
  - Rolling volatility
  - Drawdowns
- Uses KMeans clustering to group days with similar behavior
- No labels are used — regimes are discovered from data

### Why unsupervised learning
Real markets do not provide labels like "crisis" or "calm".
Clustering allows us to identify these regimes objectively.

### Outputs generated
- `regimes.csv`: daily return, volatility, drawdown, regime label
- `regimes.json`: same data formatted for frontend/API use

These outputs are later served by the backend API and visualized in the dashboard.
