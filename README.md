# Comparative Study of Machine Learning vs Deep Learning Models in Financial Return Prediction

**Authors:** AZNAK Meryam — EL BOUHALI Nouhaila  
**Institution:** Faculty of Sciences, University Mohammed V, Rabat  
**Course:** BIG DATA

---

## Project Overview

This project compares four models for predicting next-day financial returns across four major assets (^GSPC, AAPL, MSFT, GOOGL) using daily data from 2018 to 2025.

| Model | Type | R² Score |
|---|---|---|
| Linear Regression | Classical ML | 0.7003 |
| Random Forest | Ensemble ML | **0.7722** (best) |
| ANN | Deep Learning | 0.6181 |
| LSTM | Deep Learning | -0.0293 (worst) |

---

## Project Files

```
├── MLvsDL_financialReturnPrediction.py  ← main code (all models + visualizations)
├── clean_multi_asset_dataset.csv  ← dataset (8040 rows, 4 assets, 2018–2025)
├── MLvsDL_financialReturnPrediction.pdf  ← IEEE paper
└── README.md                      ← this file
```

---

## How to Run

**Requirements:** Python 3.8+

**Install dependencies:**
```bash
pip install scikit-learn tensorflow pandas numpy matplotlib seaborn
```

**Run:**
```bash
python MLvsDL_financialReturnPrediction.py
```

Make sure `clean_multi_asset_dataset.csv` is in the same folder as `MLvsDL_financialReturnPrediction.py`.

---

## What the Code Does (step by step)

| Step | What happens |
|---|---|
| 1 | Loads `clean_multi_asset_dataset.csv` |
| 2 | Engineers 20 features (lagged returns, volatility, momentum, volume, intraday gaps) |
| 3 | Splits data: train (80%) / test (20%), time-ordered — no data leakage |
| 4 | Scales features using StandardScaler |
| 5 | Builds LSTM sequences (10 consecutive days per input) |
| 6 | Trains Linear Regression and Random Forest (sklearn) |
| 7 | Trains ANN: Dense(128) → Dense(64) → Dense(32) → Dense(1) |
| 8 | Trains LSTM: LSTM(64) → LSTM(32) → Dense(1) |
| 9 | Evaluates all models (MAE, RMSE, R²) |
| 10 | Prints per-asset breakdown (^GSPC, AAPL, MSFT, GOOGL separately) |
| 11 | Saves all figures as PNG files |

---

## Output Files Generated

After running, the following files will be saved:

- `model_metrics.csv` — MAE, RMSE, R² for all models
- `comparison_linear_regression.png` — actual vs predicted time series
- `comparison_random_forest.png`
- `comparison_ann.png`
- `comparison_lstm.png`
- `scatter_all_models.png` — scatter plots for all 4 models
- `diagnostics.png` — residuals, feature importances, loss curves
- `per_asset_predictions.png` — predictions per asset per model
- `r2_per_asset.png` — R² bar chart by asset

---

## Dataset Description

**File:** `clean_multi_asset_dataset.csv`  
**Rows:** 8,040 (one per asset per trading day)  
**Period:** January 2, 2018 — December 30, 2025  
**Source:** Yahoo Finance via yfinance library  

| Column | Description |
|---|---|
| Date | Trading date |
| Close | Closing price |
| High | Daily high price |
| Low | Daily low price |
| Open | Opening price |
| Volume | Trading volume |
| Asset_ID | Ticker symbol (AAPL, MSFT, GOOGL, ^GSPC) |
| Asset_ID_encoded | Numeric encoding of Asset_ID |

---

## Key Findings

- **Random Forest** achieved the best performance (R²=0.7722) by effectively capturing non-linear interactions among engineered features
- **Linear Regression** performed surprisingly well (R²=0.7003), suggesting strong linear relationships in intraday price features
- **LSTM** underperformed (R²=−0.0293) due to insufficient training sequences — approximately 1,262 per asset, far below what LSTM needs to learn temporal patterns
- **Open–Close gap** and **High–Low range** were consistently the most important predictors across all models

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥1.3 | Data manipulation |
| numpy | ≥1.21 | Numerical computing |
| scikit-learn | ≥1.0 | LR, RF, scaling, metrics |
| tensorflow | ≥2.10 | ANN, LSTM |
| matplotlib | ≥3.5 | Visualizations |
| seaborn | ≥0.11 | Visualizations |