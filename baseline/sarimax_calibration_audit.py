"""
SARIMAX Baseline Calibration Audit Pipeline
=============================================
Runs the same rolling-window calibration audit used for Chronos-2 (coverage-by-horizon,
PIT histograms, regime-conditional coverage, bootstrap validation) on your existing
fitted SARIMAX models, so their calibration can be directly compared to Chronos-2's.

HOW TO USE
----------
Paste this into a new notebook cell (or new notebook, e.g. 14_baseline_sarimax_calibration.ipynb)
in the SAME kernel where you already have `sensex_df` / `nifty_df` and their USD/INR exogenous
series loaded (from your SARIMAX notebooks). Then call `run_full_audit(...)` once per index,
using the exact (order, seasonal_order) you already found via grid search:

    Sensex: order=(2,1,2), seasonal_order=(2,0,2,5)
    Nifty:  order=(3,1,3), seasonal_order=(1,0,1,5)

Example:
    sensex_results = run_full_audit(
        price series = sensex_df['price'],
        exog         = sensex_df['USDINR'],
        order        = (2, 1, 2),
        seasonal_order = (2, 0, 2, 5),
        index_name   = "Sensex",
    )

    nifty_results = run_full_audit(
        price series = nifty_df['price'],
        exog         = nifty_df['USDINR'],
        order        = (3, 1, 3),
        seasonal_order = (1, 0, 1, 5),
        index_name   = "Nifty",
    )

Each call returns a dict with:
    - 'coverage_by_horizon'   : DataFrame, empirical coverage at each horizon 1..H
    - 'pit_values'            : dict {horizon: array of PIT values}, for histograms
    - 'regime_coverage'       : DataFrame, coverage split by low/high vol regime, each horizon
    - 'bootstrap_ci'          : DataFrame, 95% CI on regime coverage difference at each horizon
    - 'raw'                   : the underlying per-window per-horizon L/U/y/regime table

RUNTIME NOTE
------------
SARIMAX is refit every `refit_every` windows (default 15) rather than at every single
origin date, using .append() to fold in new observations between refits. This keeps
runtime manageable. Full run is still likely to take a while (expect tens of minutes
to a few hours depending on your machine and window count) -- consider running it
in the background / overnight, or reducing `stride` to produce fewer, more widely
spaced windows if you need a faster first pass.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------------
# 1. Rolling-window quantile forecasting with periodic refit
# ---------------------------------------------------------------------------

def rolling_quantile_forecast_sarimax(
    price: pd.Series,
    exog: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    window_len: int = 512,
    horizon: int = 30,
    stride: int = 1,
    refit_every: int = 15,
    alpha: float = 0.2,
):
    """
    For each rolling origin t (stepping by `stride` days), forecast `horizon` days ahead
    with a (1-alpha) prediction interval, refitting the SARIMAX model every `refit_every`
    windows and using .append() to update state cheaply in between.

    Returns a long-format DataFrame with columns:
        origin_idx, horizon, L, U, y, origin_date
    """
    n = len(price)
    origins = list(range(window_len, n - horizon, stride))

    records = []
    model_fit = None
    last_refit_origin = None

    for w_i, t in enumerate(origins):
        # Decide whether to do a full refit or append new obs to existing fit
        needs_refit = (model_fit is None) or (w_i % refit_every == 0)

        train_price = price.iloc[:t]
        train_exog = exog.iloc[:t]

        if needs_refit:
            model = SARIMAX(
                train_price,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit(disp=False)
            last_refit_origin = t
        else:
            # Append only the NEW observations since the last refit/append
            new_price = price.iloc[last_refit_origin:t]
            new_exog = exog.iloc[last_refit_origin:t]
            if len(new_price) > 0:
                model_fit = model_fit.append(new_price, exog=new_exog, refit=False)
            last_refit_origin = t

        # Forecast `horizon` steps ahead with exogenous values held at last known level
        # (persistence assumption for the exogenous USD/INR series over the forecast window)
        future_exog = pd.DataFrame(
            {exog.name: [exog.iloc[t - 1]] * horizon},
            index=price.index[t: t + horizon] if t + horizon <= n else None,
        )
        fc = model_fit.get_forecast(steps=horizon, exog=future_exog.values)
        ci = fc.conf_int(alpha=alpha)  # columns: lower, upper

        realized = price.iloc[t: t + horizon].values
        origin_date = price.index[t]

        for h in range(horizon):
            if h < len(realized):
                records.append({
                    "origin_idx": t,
                    "origin_date": origin_date,
                    "horizon": h,  # 0-indexed: day 0 = 1 day ahead
                    "L": ci.iloc[h, 0],
                    "U": ci.iloc[h, 1],
                    "y": realized[h],
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Coverage and PIT diagnostics (mirrors your Chronos-2 notebook 02/07 logic)
# ---------------------------------------------------------------------------

def compute_coverage_by_horizon(results_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.copy()
    results_df["covered"] = (results_df["y"] >= results_df["L"]) & (results_df["y"] <= results_df["U"])
    cov = results_df.groupby("horizon")["covered"].mean().reset_index()
    cov.columns = ["horizon", "empirical_coverage"]
    cov["horizon_days_ahead"] = cov["horizon"] + 1
    return cov


def compute_pit_values(results_df: pd.DataFrame, horizon: int, dist_std_col: str = None):
    """
    Approximate PIT using a Normal CDF assumption around the interval midpoint,
    matching standard practice when only [L, U] (not the full predictive CDF) is stored.
    sigma is backed out from the interval width assuming symmetric normal quantiles
    at the given alpha.
    """
    from scipy.stats import norm

    sub = results_df[results_df["horizon"] == horizon].copy()
    mid = (sub["L"] + sub["U"]) / 2
    # For an (1-alpha) interval from a Normal, U - mid = z * sigma
    z = norm.ppf(1 - 0.2 / 2)  # alpha=0.2 -> 90th percentile z-score
    sigma = (sub["U"] - mid) / z
    pit = norm.cdf(sub["y"], loc=mid, scale=sigma)
    return pit.values


# ---------------------------------------------------------------------------
# 3. Regime labeling (reuse the same 20-day rolling vol median-split logic
#    you already used for Chronos-2, notebook 08)
# ---------------------------------------------------------------------------

def assign_volatility_regime(price: pd.Series, lookback: int = 20) -> pd.Series:
    returns = np.log(price / price.shift(1))
    rolling_vol = returns.rolling(lookback).std()
    median_vol = rolling_vol.median()
    regime = np.where(rolling_vol >= median_vol, "high", "low")
    return pd.Series(regime, index=price.index)


def compute_regime_conditional_coverage(results_df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    results_df = results_df.copy()
    results_df["covered"] = (results_df["y"] >= results_df["L"]) & (results_df["y"] <= results_df["U"])
    results_df["regime"] = results_df["origin_date"].map(regime_series)

    grouped = results_df.groupby(["horizon", "regime"])["covered"].agg(["mean", "count"]).reset_index()
    pivot = grouped.pivot(index="horizon", columns="regime", values=["mean", "count"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["horizon_days_ahead"] = pivot["horizon"] + 1
    return pivot


# ---------------------------------------------------------------------------
# 4. Bootstrap validation of the regime coverage difference
#    (same procedure as your Chronos-2 notebook 09/10)
# ---------------------------------------------------------------------------

def bootstrap_regime_difference(results_df: pd.DataFrame, regime_series: pd.Series,
                                 horizon: int, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    results_df = results_df.copy()
    results_df["covered"] = (results_df["y"] >= results_df["L"]) & (results_df["y"] <= results_df["U"])
    results_df["regime"] = results_df["origin_date"].map(regime_series)

    sub = results_df[results_df["horizon"] == horizon]
    low_covered = sub[sub["regime"] == "low"]["covered"].values
    high_covered = sub[sub["regime"] == "high"]["covered"].values

    diffs = []
    for _ in range(n_boot):
        low_sample = rng.choice(low_covered, size=len(low_covered), replace=True)
        high_sample = rng.choice(high_covered, size=len(high_covered), replace=True)
        diffs.append(low_sample.mean() - high_sample.mean())

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    point_estimate = low_covered.mean() - high_covered.mean()
    return {
        "horizon": horizon,
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": not (ci_low <= 0 <= ci_high),
    }


# ---------------------------------------------------------------------------
# 5. Full pipeline wrapper
# ---------------------------------------------------------------------------

def run_full_audit(price: pd.Series, exog: pd.Series, order: tuple, seasonal_order: tuple,
                    index_name: str, window_len: int = 512, horizon: int = 30,
                    stride: int = 1, refit_every: int = 15, alpha: float = 0.2):
    print(f"[{index_name}] Running rolling-window quantile forecasts...")
    raw = rolling_quantile_forecast_sarimax(
        price, exog, order, seasonal_order,
        window_len=window_len, horizon=horizon, stride=stride,
        refit_every=refit_every, alpha=alpha,
    )
    print(f"[{index_name}] Done. {raw['origin_idx'].nunique()} windows, "
          f"{len(raw)} total (window x horizon) observations.")

    coverage = compute_coverage_by_horizon(raw)

    regime = assign_volatility_regime(price)
    regime_coverage = compute_regime_conditional_coverage(raw, regime)

    boot_day0 = bootstrap_regime_difference(raw, regime, horizon=0)
    boot_last = bootstrap_regime_difference(raw, regime, horizon=horizon - 1)
    bootstrap_ci = pd.DataFrame([boot_day0, boot_last])

    return {
        "index_name": index_name,
        "coverage_by_horizon": coverage,
        "regime_coverage": regime_coverage,
        "bootstrap_ci": bootstrap_ci,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# 6. Comparison table builder: SARIMAX vs your existing Chronos-2 numbers
# ---------------------------------------------------------------------------

def build_comparison_table(sarimax_results: dict, chronos_coverage_day1: float,
                            chronos_coverage_day16: float, chronos_coverage_day30: float):
    cov = sarimax_results["coverage_by_horizon"]

    def get_cov(day_ahead):
        row = cov[cov["horizon_days_ahead"] == day_ahead]
        return row["empirical_coverage"].values[0] if len(row) else np.nan

    comparison = pd.DataFrame({
        "Days ahead": [1, 16, 30],
        "Chronos-2 coverage": [chronos_coverage_day1, chronos_coverage_day16, chronos_coverage_day30],
        f"SARIMAX ({sarimax_results['index_name']}) coverage": [
            get_cov(1), get_cov(16), get_cov(30)
        ],
    })
    return comparison
