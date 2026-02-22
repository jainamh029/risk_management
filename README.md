# 📊 S&P 500 Investment System: VaR-ES Risk Engine & ML Portfolio Optimizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge)

<br/>

**A full-stack quantitative investment system replicating core infrastructure used in professional asset management — covering risk analytics, portfolio construction, and ML-based alpha generation across 468+ S&P 500 securities.**

<br/>

[🎯 Overview](#-project-overview) · [🏗 Architecture](#-system-architecture) · [⚙️ Features](#️-features-in-depth) · [🚀 Quickstart](#-quickstart) · [📐 Methodology](#-methodology) · [📬 Contact](#-contact)

</div>

---

## 🎯 Project Overview

This system was built to understand — and replicate — how quantitative hedge funds and asset managers think about **risk**, **portfolio construction**, and **alpha generation** at a technical level.

It is not a class project. It is a working framework applied to real market data covering the full S&P 500 universe, built with the same mathematical foundations — VaR, ES, mean-variance optimization, ensemble ML — that institutional investors use in production.

### What it solves

| Problem | Solution |
|---|---|
| How much can I lose on a position? | Multi-method VaR/ES engine at 95% & 99% confidence |
| Which stocks deserve capital allocation? | 47+ risk-adjusted metrics ranked across 468+ securities |
| How should I construct my portfolio? | Mean-variance optimization: Max Sharpe, Min Vol, Risk Parity |
| Where are returns likely headed? | Ensemble ML models forecasting 63-day forward returns |

---

## 🏗 System Architecture

Six modular components that pipeline sequentially into each other:

```
┌──────────────────────────────────────────────────────────────┐
│  1. SP500DataFetcher       →  468+ stocks, 5-year history    │
│            ↓                                                  │
│  2. VaRESEngine            →  Risk at position level         │
│            ↓                                                  │
│  3. ComprehensiveRiskAnalyzer  →  47+ metrics per stock      │
│            ↓                                                  │
│  4. AdvancedMLPredictor    →  63-day return forecasts        │
│            ↓                                                  │
│  5. PortfolioOptimizer     →  Optimal weight allocation      │
│            ↓                                                  │
│  6. AdvancedAnalysis       →  Visualizations & reports       │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Features In Depth

### 1. Data Infrastructure — `SP500DataFetcher`

- Fetches **468+ S&P 500 stocks** and major ETFs (`SPY`, `QQQ`, `IWM`, `EFA`, `EEM`, `AGG`) via `yfinance`
- Covers all **11 GICS sectors**: Technology, Healthcare, Financials, Industrials, Energy, Materials, Utilities, Real Estate, Consumer Discretionary, Consumer Staples, Communication Services
- Robust batch fetching (20 stocks/batch) with automatic error handling and fallback mechanisms
- Outputs: clean price series, return series, and volume data aligned across all tickers

---

### 2. VaR-ES Risk Engine — `VaRESEngine`

The core risk measurement module. Computes portfolio and individual security risk using three distinct methodologies:

| Method | Description | When It's Best |
|---|---|---|
| **Historical Simulation** | Uses actual return distribution — no distributional assumptions | When returns are non-normal |
| **Parametric (Normal)** | Assumes Gaussian returns; fast and analytically clean | For large, liquid portfolios |
| **Cornish-Fisher** | Adjusts for skewness and excess kurtosis (fat tails) | Best for individual stocks |

**Outputs at 95% and 99% confidence:**
- `VaR` — Maximum expected loss at the given confidence level
- `ES / CVaR` — Average loss *beyond* the VaR threshold (tail risk expectation)

---

### 3. Comprehensive Risk Analyzer — `ComprehensiveRiskAnalyzer`

Computes **47+ metrics per security**, organized into six categories:

| Category | Metrics |
|---|---|
| **Return Metrics** | Daily, annual, and period returns (1M / 3M / 6M / 1Y) |
| **Volatility** | Annual volatility, downside deviation, downside volatility |
| **Risk-Adjusted Ratios** | Sharpe, Sortino, Calmar ratios |
| **Drawdown Analysis** | Maximum drawdown, current drawdown, drawdown duration |
| **Market Sensitivity** | Beta vs S&P 500, rolling correlation to benchmark |
| **Technical Indicators** | RSI, SMA 20/50/200, Price-to-SMA deviation ratios |

> **Investment Score:** A composite long-term quality metric synthesizing risk, return, and momentum signals into a single ranked score per security.

---

### 4. ML Predictor — `AdvancedMLPredictor`

Generates **63-day forward return forecasts** using ensemble machine learning.

**Feature Engineering (50+ features)**

```
Statistical Moments   →  Mean returns, volatility, skewness, kurtosis
Momentum Indicators   →  1M, 3M, 6M, 12M momentum signals
Moving Averages       →  SMA ratios, EMA crossovers
Market Correlation    →  Rolling beta, correlation to SPY
Relative Strength     →  Cross-sectional momentum ranks
```

**Models**

| Model | Strength |
|---|---|
| `Random Forest Regressor` | Robust to overfitting; captures non-linear relationships |
| `Gradient Boosting Regressor` | Sequential error correction; strong out-of-sample performance |

**Validation**
- Time-series cross-validation — future data is never seen during training (no leakage)
- `SelectKBest` feature selection to eliminate noise
- Evaluation metrics: MSE, MAE, R²

---

### 5. Portfolio Optimizer — `PortfolioOptimizer`

Implements **mean-variance optimization** via `scipy` to construct three distinct portfolios:

| Portfolio Type | Objective | Best For |
|---|---|---|
| **Maximum Sharpe** | Maximize return per unit of risk | Growth-oriented allocation |
| **Minimum Volatility** | Minimize portfolio variance | Capital preservation |
| **Risk Parity** | Equal risk contribution from each asset | Robust diversification |

**Constraints applied:**
- Long-only (no short selling)
- Individual position limits to prevent over-concentration
- Leverage constraints consistent with typical fund mandates

---

### 6. Visualizations — `AdvancedAnalysis`

Interactive and static charts built with `Plotly` and `Matplotlib`:

- Efficient frontier with Max Sharpe, Min Vol, and Risk Parity portfolio overlays
- VaR/ES return distribution histograms
- Sector-level correlation heatmaps
- Drawdown time series per security
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

# Launch the notebook
jupyter notebook Risk_Management_and_Var_es_engine_for_stocks.ipynb
```

### Step-by-Step Usage

**Step 1 — Fetch Data**
```python
fetcher = SP500DataFetcher()
prices  = fetcher.fetch_data(start_date='2019-01-01', end_date='2024-01-01')
returns = fetcher.returns
```

**Step 2 — Run Risk Engine**
```python
var_engine     = VaRESEngine(returns, portfolio_value=100_000)
var_es_results = var_engine.calculate_var_es(
    confidence_levels=[0.95, 0.99],
    window=250
)
```

**Step 3 — Compute Risk Metrics**
```python
risk_analyzer = ComprehensiveRiskAnalyzer(returns, prices, var_es_results)
metrics       = risk_analyzer.calculate_all_metrics()
```

**Step 4 — Generate ML Forecasts**
```python
ml_predictor = AdvancedMLPredictor(returns, prices, metrics)
predictions  = ml_predictor.predict_returns(
    selected_tickers=metrics['Ticker'].head(50).tolist(),
    train_size=0.7
)
```

**Step 5 — Optimize Portfolio**
```python
optimizer  = PortfolioOptimizer(returns, metrics)
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

**Why three VaR methods?**
No single method dominates across all conditions. Historical simulation captures real distributions but is backward-looking. Parametric VaR is analytically clean but assumes normality — which equity returns routinely violate in the tails. The Cornish-Fisher expansion explicitly corrects for skewness and excess kurtosis, making it the most appropriate method for individual stock-level risk measurement.

**Why Sortino over Sharpe?**
The Sharpe ratio penalizes upside and downside volatility equally. Investors don't object to positive surprises — only losses matter. The Sortino ratio isolates downside deviation, producing a more accurate picture of risk-adjusted return quality for portfolio selection.

**Why time-series cross-validation for ML?**
Standard k-fold CV randomly shuffles data, allowing future information to contaminate training — a critical flaw in financial applications. Time-series CV enforces strict temporal ordering so that training always precedes validation, preserving the causal structure of market data.

**Why Risk Parity?**
Mean-variance optimization tends to produce highly concentrated portfolios dominated by low-volatility assets. Risk parity ensures each position contributes equally to total portfolio risk, delivering more robust diversification — particularly important during regime changes when asset correlations shift dramatically.

---

## 📊 Data Coverage

| Sector | Approx. Stocks |
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
| ETFs (SPY, QQQ, IWM, VTI…) | 10 |
| **Total** | **~468** |

---

## 📁 Repository Structure

```
risk_management/
│
├── Risk_Management_and_Var_es_engine_for_stocks.ipynb
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
| `yfinance` | Market data fetching — S&P 500 historical prices |
| `scipy` | Portfolio optimization via constrained minimization |
| `scikit-learn` | ML models, feature selection, time-series cross-validation |
| `matplotlib` / `seaborn` | Static visualizations and statistical plots |
| `plotly` | Interactive dashboards and charts |

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not financial advice and does not constitute a recommendation to buy or sell any security. Past performance does not guarantee future results. Always conduct independent research and consult a qualified financial professional before making investment decisions.

---

## 📬 Contact

**Jainam Shah** · MS Quantitative Finance, Northeastern University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/s-jainam)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jainamh029)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shah.jainamh@northeastern.edu)

---

<div align="center">
<sub>Built with Python · February 2026 · Boston, MA</sub>
</div>
