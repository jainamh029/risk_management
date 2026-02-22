# 📊 S&P 500 Investment System: VaR-ES Risk Engine & ML Portfolio Optimizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge)

**A full-stack quantitative investment system replicating core infrastructure used in professional asset management — covering risk analytics, portfolio construction, and ML-based alpha generation across 468+ S&P 500 securities.**

[Overview](#-project-overview) · [Features](#-system-architecture) · [Quickstart](#-quickstart) · [Methodology](#-methodology) · [Results](#-key-outputs) · [Contact](#-contact)

</div>

---

## 🎯 Project Overview

This system was built to understand — and replicate — how quantitative hedge funds and asset managers think about **risk**, **portfolio construction**, and **alpha generation** at a technical level.

It is not a class project. It is a working framework applied to real market data covering the full S&P 500 universe, built with the same mathematical foundations (VaR, ES, mean-variance optimization, ensemble ML) that institutional investors use in production.

### What it solves:
| Problem | Solution |
|---|---|
| How much can I lose on a position? | Multi-method VaR/ES engine at 95% & 99% confidence |
| Which stocks deserve capital allocation? | 47+ risk-adjusted metrics ranked across 468+ securities |
| How should I construct my portfolio? | Mean-variance optimization: Max Sharpe, Min Vol, Risk Parity |
| Where are returns likely headed? | Ensemble ML models forecasting 63-day forward returns |

---

## 🏗 System Architecture

The system is built as six modular components that pipeline into each other:

```
 ┌─────────────────────────────────────────────────────────┐
 │  1. SP500DataFetcher  →  468+ stocks, 5yr history       │
 │           ↓                                              │
 │  2. VaRESEngine       →  Risk at position level         │
 │           ↓                                              │
 │  3. ComprehensiveRiskAnalyzer  →  47+ metrics/stock     │
 │           ↓                                              │
 │  4. AdvancedMLPredictor  →  63-day return forecasts     │
 │           ↓                                              │
 │  5. PortfolioOptimizer   →  Optimal weight allocation   │
 │           ↓                                              │
 │  6. AdvancedAnalysis     →  Visualizations & reports    │
 └─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Features In Depth

### 1. Data Infrastructure — `SP500DataFetcher`
- Fetches **468+ S&P 500 stocks** and major ETFs (SPY, QQQ, IWM, EFA, EEM, AGG) via yfinance
- Covers all 11 GICS sectors: Technology, Healthcare, Financials, Industrials, Energy, Materials, Utilities, Real Estate, Consumer Discretionary, Consumer Staples, Communication Services
- Robust batch fetching (20 stocks/batch), automatic error handling and fallback mechanisms
- Outputs: clean price series, return series, and volume data aligned across all tickers

---

### 2. VaR-ES Risk Engine — `VaRESEngine`
The core risk measurement module. Computes portfolio and individual security risk using three distinct methodologies:

| Method | Description | When it's best |
|---|---|---|
| **Historical Simulation** | Uses actual return distribution — no distributional assumptions | When returns are non-normal |
| **Parametric (Normal)** | Assumes Gaussian returns; fast and analytically clean | For large, liquid portfolios |
| **Cornish-Fisher** | Adjusts for skewness and excess kurtosis (fat tails) | Best for individual stocks |

**Outputs at 95% and 99% confidence:**
- `VaR` — Maximum expected loss at confidence level
- `ES (CVaR)` — Average loss *beyond* the VaR threshold (tail risk expectation)

---

### 3. Comprehensive Risk Analyzer — `ComprehensiveRiskAnalyzer`
Computes **47+ metrics per security**, organized into six categories:

**Return Metrics**
- Daily, annual, and period returns (1M / 3M / 6M / 1Y)

**Volatility Metrics**
- Annual volatility, downside deviation, downside volatility

**Risk-Adjusted Ratios**
- `Sharpe Ratio` — Excess return per unit of total risk
- `Sortino Ratio` — Excess return per unit of *downside* risk only
- `Calmar Ratio` — Return per unit of maximum drawdown

**Drawdown Analysis**
- Maximum drawdown, current drawdown, drawdown duration

**Market Sensitivity**
- Beta vs S&P 500, correlation to benchmark

**Technical Indicators**
- RSI (momentum), SMA 20/50/200 (trend), Price-to-SMA ratios (deviation from trend)

**Investment Score**
- Composite long-term quality metric synthesizing risk/return/momentum signals

---

### 4. ML Predictor — `AdvancedMLPredictor`
Generates **63-day forward return forecasts** using ensemble machine learning:

**Feature Engineering (50+ features)**
```
Statistical Moments:     Mean returns, volatility, skewness, kurtosis
Momentum Indicators:     1M, 3M, 6M, 12M momentum signals
Moving Averages:         SMA ratios, EMA crossovers
Market Correlation:      Rolling beta, correlation to SPY
Relative Strength:       Cross-sectional momentum ranks
```

**Models**
- `Random Forest Regressor` — Robust to overfitting, captures non-linear relationships
- `Gradient Boosting Regressor` — Sequential error correction, strong out-of-sample performance

**Validation**
- Time-series cross-validation (no data leakage — future data never seen in training)
- `SelectKBest` feature selection to reduce noise
- Reported metrics: MSE, MAE, R²

---

### 5. Portfolio Optimizer — `PortfolioOptimizer`
Implements **mean-variance optimization** via scipy to construct three distinct portfolios:

| Portfolio Type | Objective | Use Case |
|---|---|---|
| **Maximum Sharpe** | Maximize return per unit of risk | Growth-oriented allocation |
| **Minimum Volatility** | Minimize portfolio variance | Capital preservation |
| **Risk Parity** | Equal risk contribution from each asset | Diversification-focused |

Constraints applied:
- No short selling (long-only)
- Individual position limits (prevents over-concentration)
- Leverage constraints consistent with typical fund mandates

---

### 6. Visualizations — `AdvancedAnalysis`
Interactive and static charts built with Plotly and Matplotlib:
- Efficient frontier with portfolio overlays
- VaR/ES distribution histograms
- Correlation heatmaps by sector
- Drawdown time series
- Performance attribution by sector and factor
- ML feature importance rankings

---

## 🚀 Quickstart

### Prerequisites
```bash
pip install numpy pandas yfinance scipy scikit-learn matplotlib seaborn plotly
```

### Run the System
```bash
# Clone the repo
git clone https://github.com/jainamh029/risk_management.git
cd risk_management

# Open the notebook
jupyter notebook Risk_Management_and_Var_es_engine_for_stocks.ipynb
```

### Step-by-Step Usage

**Step 1 — Fetch Data**
```python
fetcher = SP500DataFetcher()
prices = fetcher.fetch_data(start_date='2019-01-01', end_date='2024-01-01')
returns = fetcher.returns
```

**Step 2 — Run Risk Engine**
```python
var_engine = VaRESEngine(returns, portfolio_value=100000)
var_es_results = var_engine.calculate_var_es(
    confidence_levels=[0.95, 0.99],
    window=250
)
```

**Step 3 — Compute Risk Metrics**
```python
risk_analyzer = ComprehensiveRiskAnalyzer(returns, prices, var_es_results)
metrics = risk_analyzer.calculate_all_metrics()
```

**Step 4 — Generate ML Forecasts**
```python
ml_predictor = AdvancedMLPredictor(returns, prices, metrics)
predictions = ml_predictor.predict_returns(
    selected_tickers=metrics['Ticker'].head(50).tolist(),
    train_size=0.7
)
```

**Step 5 — Optimize Portfolio**
```python
optimizer = PortfolioOptimizer(returns, metrics)

max_sharpe = optimizer.optimize_max_sharpe(selected_tickers=top_stocks)
min_vol    = optimizer.optimize_min_volatility(selected_tickers=top_stocks)
```

**Step 6 — Visualize & Report**
```python
analyzer = AdvancedAnalysis(returns, prices, metrics, predictions)
analyzer.create_comprehensive_report()
```

---

## 📐 Methodology

### Why Three VaR Methods?
No single VaR method dominates in all conditions. Historical simulation captures real return distributions but is backward-looking. Parametric VaR is fast but assumes normality — which stock returns violate, especially in tail events. The Cornish-Fisher expansion explicitly corrects for the skewness and excess kurtosis observed in equity returns, making it particularly appropriate for individual stock-level risk measurement.

### Why Sortino Over Sharpe?
The Sharpe ratio penalizes upside and downside volatility equally. For investment decision-making, only downside volatility matters — investors don't object to upside surprises. The Sortino ratio isolates the downside deviation, giving a cleaner picture of risk-adjusted return quality.

### Why Time-Series Cross-Validation for ML?
Standard k-fold cross-validation randomly shuffles data, creating data leakage when applied to time series (future information contaminates training). Time-series CV respects temporal ordering — training always precedes validation — preserving the causal structure of financial data.

### Why Risk Parity?
Mean-variance optimization tends to produce highly concentrated portfolios dominated by low-volatility assets. Risk parity allocates capital so that each position contributes equally to total portfolio risk, producing more robust diversification — particularly important in regime changes when correlations shift.

---

## 📊 Data Coverage

| Sector | Approx. Count |
|---|---|
| Technology | 50+ |
| Healthcare | 63+ |
| Financials | 61+ |
| Industrials | 72+ |
| Consumer Discretionary | 56+ |
| Consumer Staples | 37+ |
| Communication Services | 23+ |
| Energy | 24+ |
| Materials | 26+ |
| Real Estate | 31+ |
| Utilities | 31+ |
| ETFs (SPY, QQQ, IWM…) | 10 |
| **Total** | **~468** |

---

## 📁 Repository Structure

```
risk_management/
│
├── Risk_Management_and_Var_es_engine_for_stocks.ipynb   # Main notebook
│   ├── Part 1: SP500DataFetcher
│   ├── Part 2: VaRESEngine
│   ├── Part 3: ComprehensiveRiskAnalyzer
│   ├── Part 4: AdvancedMLPredictor
│   ├── Part 5: PortfolioOptimizer
│   └── Part 6: AdvancedAnalysis
│
└── README.md
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `yfinance` | Market data fetching (S&P 500 historical prices) |
| `scipy` | Portfolio optimization (minimize, efficient frontier) |
| `scikit-learn` | ML models, feature selection, cross-validation |
| `matplotlib` / `seaborn` | Static visualizations |
| `plotly` | Interactive dashboards and charts |

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not financial advice and does not constitute a recommendation to buy or sell any security. Past performance does not guarantee future results. Always conduct independent research and consult qualified financial professionals before making investment decisions.

---

## 📬 Contact

**Jainam Shah**
MS Quantitative Finance · Northeastern University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/s-jainam)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jainamh029)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shah.jainamh@northeastern.edu)

---

<div align="center">
<sub>Built with Python · February 2026 · Boston, MA</sub>
</div>
