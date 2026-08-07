# Uncertainty Quantification & Conformal Prediction in Financial Time Series

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Model-Amazon--Chronos--2-orange.svg)](https://github.com/amazon-science/chronos-forecasting)
[![Focus](https://img.shields.io/badge/Focus-Conformal%20Prediction%20%26%20ACI-green.svg)](#methodology)

An end-to-end research and engineering framework evaluating uncertainty quantification and distribution-free risk calibration for zero-shot time series foundation models (**Amazon Chronos-2**). Applied directly to major financial indices (**BSE Sensex** & **NSE Nifty 50**), this project diagnoses probabilistic miscalibration and guarantees distribution-free coverage bounds using **Fixed Conformal Prediction** and **Adaptive Conformal Inference (ACI)**.

---

## 📌 Overview

Traditional probabilistic point and interval forecasts in financial markets frequently suffer from **overconfidence** during periods of high volatility and regime shifts. While foundational time series models like Amazon's Chronos-2 provide powerful zero-shot predictions, their out-of-the-box prediction intervals often fail to maintain valid statistical coverage guarantees when market distributions shift.

This project addresses those failure modes by:
1. **Diagnosing Model Miscalibration:** Identifying overconfidence using **Probability Integral Transform (PIT) histograms** and empirical coverage metrics across multiple prediction horizons.
2. **Guaranteed Coverage via Conformal Prediction:** Applying distribution-free Split Conformal Prediction to produce valid $1 - \alpha$ prediction bands.
3. **Dynamic Adaptation in Non-Stationary Markets:** Implementing **Adaptive Conformal Inference (ACI)** to dynamically recalibrate interval widths following sudden regime shifts.
4. **Analyzing the "Horizon Reversal Effect":** Evaluating how forecasting horizon lengths interact with volatility regimes and interval widening/narrowing dynamics.

---

## ⚙️ Key Features

* **Foundation Model Integration:** Uses Amazon's **Chronos-2** pretrained model as the underlying probabilistic forecaster.
* **Miscalibration Diagnostics:**
  * Empirical coverage tracking against nominal target confidence (e.g., $90\%$ target coverage).
  * Probability Integral Transform (PIT) histogram generation to detect underdispersion, overdispersion, or asymmetric bias.
* **Conformal Methods Implemented:**
  * **Fixed Split Conformal Prediction:** Computes non-conformity scores over calibration windows to yield distribution-free prediction bands.
  * **Adaptive Conformal Inference (ACI):** Dynamically updates the nominal level $\alpha_t$ step-by-step using online learning to maintain target coverage under distribution shifts.
* **Regime Shift & Horizon Analytics:**
  * Empirical tracking of **recovery speed** following volatility spikes.
  * Quantitative analysis of the **Horizon Reversal Effect** across varying forecast horizons ($h \in \{1, 5, 10, 20\}$ days).

---

## 🛠️ Methodology & Workflow

┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│  Market Data    │     │   Chronos-2 Forecaster│     │ Calibration Metrics  │
│ (Sensex / Nifty)│ ──> │ (Zero-Shot Inference) │ ──> │ (PIT, Coverage Rate) │
└─────────────────┘     └───────────────────────┘     └──────────┬───────────┘
│
▼
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ Adaptive Bands  │     │ Conformal Recalibrate │     │ Miscalibration       │
│  (ACI Update)   │ <── │ (Fixed Split vs ACI)  │ <── │ Diagnostic Trigger   │
└─────────────────┘     └───────────────────────┘     └──────────────────────┘

### 1. Conformal Prediction Formulation
For a target confidence level $1 - \alpha = 0.90$, we define non-conformity scores $s_i$ over a calibration set using residuals between actual asset prices/returns $Y_i$ and model quantiles $\hat{q}$:

$$s_i = \max \left( \hat{q}_{\alpha/2}(X_i) - Y_i, \; Y_i - \hat{q}_{1 - \alpha/2}(X_i) \right)$$

The conformal quantile $Q_{1-\alpha}(s)$ is then added to widen or narrow intervals, guaranteeing valid empirical coverage regardless of underlying data distribution.

### 2. Adaptive Conformal Inference (ACI)
To handle time-series non-stationarity, ACI adjusts the effective alpha $\alpha_t$ dynamically at each step $t$:

$$\alpha_{t+1} = \alpha_t + \gamma (\alpha - \mathbf{1}_{\{Y_t \notin \hat{C}_t\}})$$

where $\gamma$ is the learning rate governing adaptation speed to sudden volatility spikes.

---

## 📂 Repository Structure

```text
conformal-prediction_project/
├── data/
│   ├── raw/                # Historical BSE Sensex & NSE Nifty 50 data
│   └── processed/          # Formatted time series & returns split
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chronos_baseline_forecasting.ipynb
│   ├── 03_pit_diagnostics_coverage.ipynb
│   └── 04_conformal_and_aci_benchmarks.ipynb
├── src/
│   ├── models/
│   │   └── chronos_wrapper.py     # Chronos-2 inference interface
│   ├── conformal/
│   │   ├── split_conformal.py     # Fixed Conformal Prediction implementation
│   │   └── aci.py                 # Adaptive Conformal Inference engine
│   ├── metrics/
│   │   └── calibration.py         # Empirical coverage & PIT histogram calculations
│   └── visualization/
│       └── plots.py               # Interval and diagnostic plotting utilities
├── docs/
│   ├── project_guide.pdf          # Full technical guide & mathematical details
│   └── blog_series/               # Explanatory guides on calibration diagnostics
├── requirements.txt               # Dependencies
├── LICENSE                        # License file
└── README.md                      # Project documentation
