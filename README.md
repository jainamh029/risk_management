# S&P 500 Complete Investment System with VaR-ES Engine & ML Predictions

A comprehensive Python-based investment analytics and risk management system for analyzing S&P 500 stocks with advanced risk metrics, machine learning predictions, and portfolio optimization.

## 📋 Project Overview

This project provides a complete framework for:
- **Data Fetching**: Retrieving comprehensive historical data for ~468 S&P 500 stocks and ETFs
- **Risk Analysis**: Value at Risk (VaR), Expected Shortfall (ES), and multiple risk metrics
- **Machine Learning**: Advanced predictions using ensemble methods
- **Portfolio Optimization**: Risk-adjusted portfolio construction and analysis
- **Visualizations**: Interactive and static plots for market insights

## 🎯 Key Feature

### 1. **Comprehensive Data Fetcher (SP500DataFetcher)**
   - Fetches data for 468+ S&P 500 stocks across all sectors
   - Covers multiple sectors: Technology, Finance, Healthcare, Energy, Industrials, Materials, Real Estate, Utilities, Consumer Discretionary, Consumer Staples, Communication Services
   - Includes market ETFs: SPY, QQQ, DIA, IWM, VTI, VOO, IVV, EFA, EEM, AGG
   - Robust error handling with fallback mechanisms
   - Data includes: Close prices, Volume, High/Low prices
   - Automatic data cleaning and alignment

### 2. **VaR-ES Risk Engine (VaRESEngine)**
   - **Value at Risk (VaR)**: Estimates potential losses at specified confidence levels (95%, 99%)
   - **Expected Shortfall (ES)**: Average loss beyond the VaR threshold
   - **Multiple Calculation Methods**:
     - Historical simulation
     - Parametric (assuming normal distribution)
     - Cornish-Fisher adjustment for skewness/kurtosis
   - Risk metrics per stock and portfolio-level

### 3. **Comprehensive Risk Analyzer (ComprehensiveRiskAnalyzer)**
   - **Return Metrics**: Daily, annual, and period returns (1M, 3M, 6M, 1Y)
   - **Volatility Metrics**: Daily/annual volatility, downside deviation
   - **Risk-Adjusted Returns**:
     - Sharpe Ratio: Return per unit of risk
     - Sortino Ratio: Return per unit of downside risk
     - Calmar Ratio: Return per unit of drawdown
   - **Drawdown Analysis**: Maximum and current drawdown
   - **Higher Moments**: Skewness and Kurtosis
   - **Technical Analysis**: Support/Resistance (SMA 20, 50, 200), RSI
   - **Beta Calculation**: Market-relative risk
   - **Investment Score**: Composite long-term investment quality metric

### 4. **Advanced ML Predictor (AdvancedMLPredictor)**
   - **Feature Engineering**: 50+ engineered features including:
     - Statistical moments (mean, volatility, skew, kurtosis)
     - Momentum indicators
     - Moving averages
     - Relative strength
     - Market correlations
   - **Ensemble Models**:
     - Random Forest Regressor
     - Gradient Boosting Regressor
   - **Time-Series Cross-Validation**: Maintains temporal integrity
   - **Feature Selection**: Uses SelectKBest for optimal features
   - **Predictions**: 63-day forward return forecasts
   - **Model Performance**: MSE, MAE, R² metrics

### 5. **Portfolio Optimizer (PortfolioOptimizer)**
   - **Mean-Variance Optimization**: Efficient frontier calculation
   - **Optimization Objectives**:
     - Maximum Sharpe Ratio portfolio
     - Minimum volatility portfolio
     - Risk parity portfolio
   - **Constraints**: Leverage limits, individual position limits
   - **Weight Allocation**: Optimized for risk-adjusted returns

### 6. **Advanced Analysis & Visualizations (AdvancedAnalysis)**
   - Risk heatmaps and correlation matrices
   - Efficient frontier plots
   - VaR/ES distribution analysis
   - Performance attribution
   - Interactive Plotly visualizations
   - Sector and category breakdowns

## 📊 Data Coverage

| Sector | Stocks |
|--------|--------|
| Technology | 50+ |
| Communication Services | 23+ |
| Healthcare | 63+ |
| Financial | 61+ |
| Consumer Discretionary | 56+ |
| Consumer Staples | 37+ |
| Energy | 24+ |
| Industrials | 72+ |
| Materials | 26+ |
| Real Estate | 31+ |
| Utilities | 31+ |
| **ETFs** | **10** |
| **Total** | **~468** |

## 🛠️ Technical Stack

### Required Libraries
```
numpy              # Numerical computations
pandas             # Data manipulation and analysis
yfinance          # Yahoo Finance data fetching
scipy             # Statistical functions and optimization
scikit-learn      # Machine learning models
matplotlib        # Static visualizations
seaborn           # Statistical visualization
plotly            # Interactive visualizations
```

### Python Version
- Python 3.7+

## 📦 Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/project
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas yfinance scipy scikit-learn matplotlib seaborn plotly
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Risk_Management_and_Var_es_engine_for_stocks.ipynb
   ```

## 🚀 Usage Guide

### Step 1: Data Fetching
```python
# Initialize the data fetcher
fetcher = SP500DataFetcher()

# Fetch data (default: last 5 years)
prices = fetcher.fetch_data(start_date='2019-01-01', end_date='2024-01-01')

# Access the data
returns = fetcher.returns
volumes = fetcher.volume
```

### Step 2: Risk Analysis with VaR-ES
```python
# Initialize VaR-ES engine
var_engine = VaRESEngine(returns, portfolio_value=100000)

# Calculate VaR and ES
var_es_results = var_engine.calculate_var_es(
    confidence_levels=[0.95, 0.99],
    window=250
)
```

### Step 3: Comprehensive Risk Metrics
```python
# Initialize risk analyzer
risk_analyzer = ComprehensiveRiskAnalyzer(returns, prices, var_es_results)

# Calculate all metrics
metrics = risk_analyzer.calculate_all_metrics()

# View metrics
metrics.head()
```

### Step 4: ML Predictions
```python
# Initialize ML predictor
ml_predictor = AdvancedMLPredictor(returns, prices, metrics)

# Train and predict
predictions = ml_predictor.predict_returns(
    selected_tickers=metrics['Ticker'].head(50).tolist(),
    train_size=0.7
)
```

### Step 5: Portfolio Optimization
```python
# Initialize portfolio optimizer
optimizer = PortfolioOptimizer(returns, metrics)

# Get optimized portfolios
max_sharpe_portfolio = optimizer.optimize_max_sharpe(
    selected_tickers=top_stocks
)

min_vol_portfolio = optimizer.optimize_min_volatility(
    selected_tickers=top_stocks
)
```

### Step 6: Advanced Analysis & Visualization
```python
# Initialize advanced analyzer
analyzer = AdvancedAnalysis(returns, prices, metrics, predictions)

# Generate comprehensive analysis
analyzer.create_comprehensive_report()
```

## 📈 Key Metrics Explained

### Value at Risk (VaR) & Expected Shortfall (ES)
- **VaR 95%**: Maximum loss at 95% confidence level (5% tail risk)
- **VaR 99%**: Maximum loss at 99% confidence level (1% tail risk)
- **ES**: Average loss if VaR is exceeded (tail risk expectation)

### Risk-Adjusted Return Ratios
- **Sharpe Ratio**: Excess return per unit of total risk
- **Sortino Ratio**: Excess return per unit of downside risk (penalizes losses only)
- **Calmar Ratio**: Return per unit of maximum drawdown

### Volatility Metrics
- **Annual Volatility**: Annualized standard deviation of returns
- **Downside Volatility**: Volatility of negative returns only
- **Beta**: Correlation-adjusted market sensitivity

### Drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Current Drawdown**: Current distance from recent peak

### Technical Indicators
- **RSI (Relative Strength Index)**: Momentum indicator (0-100)
- **SMA (Simple Moving Average)**: Trend indicators for 20, 50, 200-day periods
- **Price-to-SMA Ratio**: Deviation from trend

## 🔄 Data Flow

```
1. Data Fetching (yfinance)
   ↓
2. Data Cleaning & Alignment
   ↓
3. Returns Calculation
   ↓
4. ├─ VaR-ES Engine (Risk Metrics)
   ├─ Comprehensive Risk Analyzer (47+ Metrics)
   └─ ML Predictor (Feature Engineering)
   ↓
5. Portfolio Optimizer
   ↓
6. Advanced Analysis & Visualization
```

## 📊 Output Examples

The system generates:
- **Risk Metrics DataFrame**: 47+ risk and return metrics per stock
- **VaR-ES Results**: Multiple confidence levels and calculation methods
- **ML Predictions**: 63-day forward returns with confidence intervals
- **Optimized Portfolios**: Weights for maximum Sharpe, minimum volatility, risk parity
- **Interactive Visualizations**: Efficient frontiers, heatmaps, time series plots
- **Performance Reports**: Comprehensive analysis with insights

## 💡 Notable Features

1. **Robust Error Handling**: Graceful fallback for failed data fetches
2. **Time-Series Integrity**: Proper handling of temporal data structures
3. **Scalability**: Handles 468+ stocks with efficient batch processing
4. **Statistical Rigor**: Multiple VaR calculation methods, proper resampling
5. **ML Best Practices**: Train-test split, cross-validation, feature scaling
6. **Interactive Visualization**: Plotly-based interactive charts
7. **Comprehensive Metrics**: 47+ calculated risk/return metrics per stock

## ⚡ Performance Considerations

- **Data Fetching**: In batches of 20 stocks for efficiency (~5-10 minutes for full dataset)
- **ML Training**: Time-series cross-validation on 468+ stocks (~10-20 minutes)
- **Optimization**: Efficient scipy minimize for portfolio weights (~1-2 minutes)
- **Memory**: Requires ~500MB-1GB for full dataset

## 🎓 Educational Value

This project demonstrates:
- Financial data analysis and processing
- Risk management in practice (VaR, ES, portfolio theory)
- Machine learning applied to time-series prediction
- Portfolio optimization algorithms
- Statistical analysis and visualization
- Object-oriented Python design patterns
- Data pipeline architecture

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice or a recommendation to buy/sell securities. Past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.

## 📝 Project Structure

```
Risk_Management_and_Var_es_engine_for_stocks.ipynb
├── Part 1: Data Fetcher (SP500DataFetcher class)
├── Part 2: VaR-ES Engine (VaRESEngine class)
├── Part 3: Risk Analyzer (ComprehensiveRiskAnalyzer class)
├── Part 4: ML Predictor (AdvancedMLPredictor class)
├── Part 5: Portfolio Optimizer (PortfolioOptimizer class)
└── Part 6: Advanced Analysis (AdvancedAnalysis class)
```

## 🔧 Customization

You can customize the system by:
- Modifying the ticker list in `SP500DataFetcher`
- Adjusting confidence levels in `VaRESEngine`
- Changing optimization objectives in `PortfolioOptimizer`
- Tweaking ML parameters in `AdvancedMLPredictor`
- Selecting specific sectors or cap sizes for analysis

## 📧 Support & Contribution

For issues, suggestions, or improvements, please review the code comments and docstrings within the notebook for detailed explanation of each component.

## 📄 License

This project is provided as-is for educational purposes. Feel free to modify and use for your learning.

---

**Last Updated**: February 2026  
**Python Version**: 3.7+  
**Status**: Production Ready
