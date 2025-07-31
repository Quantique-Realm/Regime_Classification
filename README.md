# Market Regimes Classification using Hidden Markov Models (HMM)

> **Domain:** Finance and Analytics  
> **Problem Statement (PS) Name:** Market Regimes Classification using HMM  
> **PS Number:** FA-9  
> **Preference Number:** 2

## 📌 Overview

This project focuses on classifying market regimes (e.g., bull, bear, high volatility) using **Hidden Markov Models (HMM)**. By analyzing financial time-series data, we identify latent states in market behavior through unsupervised learning techniques. This enables regime-aware trading strategies and better risk-adjusted performance.

---

## 📊 Key Objectives

- Acquire and preprocess high-frequency OHLCV financial data.
- Engineer features including technical indicators, volatility measures, and correlation metrics.
- Train HMMs with Gaussian and GMM emissions to detect latent market states.
- Perform unsupervised learning using the **Baum-Welch** algorithm.
- Predict future market regimes using the **Viterbi** algorithm.
- Build interactive visualizations and backtest regime-specific trading strategies.

---

## 🧠 Features

- ✅ **Data Acquisition**: Yahoo Finance, Alpha Vantage, Quandl APIs.
- ✅ **Preprocessing**: Missing values, outlier handling, resampling.
- ✅ **Feature Engineering**: RSI, MACD, Bollinger Bands, volatility, returns, correlations.
- ✅ **HMM Modeling**: Gaussian and GMM emissions using `hmmlearn`.
- ✅ **Regime Detection**: Unsupervised state learning (Baum-Welch), forward-looking prediction (Viterbi).
- ✅ **Visualization**: Regime plots, heatmaps, transition matrices using `Matplotlib`, `Seaborn`, `Plotly`.
- ✅ **Backtesting**: Regime-aware strategies in `Backtrader`.
- ✅ **Performance Metrics**: Sharpe Ratio, Sortino Ratio, drawdown analysis.

---

## 🛠 Tech Stack

| Category              | Tools/Libraries Used |
|-----------------------|----------------------|
| Language              | Python               |
| Data APIs             | Yahoo Finance,       |
| Preprocessing         | Pandas, NumPy        |
| Feature Engineering   | SciPy                 |
| Modeling              | hmmlearn, scikit-learn |
| Visualization         | Matplotlib, Plotly, Seaborn |
| Backtesting           | Backtrader           |

---

## 🗂 Project Structure

```plaintext
Regime_Classification/
├── data/                  # Raw & processed data
├── notebooks/             # Jupyter notebooks for each stage
├── src/                   # Core source code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── hmm_model.py
│   ├── visualization.py
│   └── strategy.py
├── results/               # Plots, results, output files
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── LICENSE
