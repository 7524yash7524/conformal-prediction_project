# Chronos-2 Calibration Audit: Sensex & Nifty 50

Do the prediction intervals of a zero-shot time-series foundation model (Amazon **Chronos-2**) cover as often as they claim? I tested this on the BSE Sensex and NSE Nifty 50, then tried to fix it with conformal prediction.

**Paper:** [SSRN 7300198](https://doi.org/10.2139/ssrn.7300198) &nbsp;|&nbsp; **Blog series:** [Medium](https://medium.com/@ysrivastava470) &nbsp;|&nbsp; **Follow-up project:** [coverage-aware-guardrails](https://github.com/7524yash7524/coverage-aware-guardrails)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Model](https://img.shields.io/badge/Model-amazon%2Fchronos--2-orange.svg)
![Topic](https://img.shields.io/badge/Topic-Conformal%20Prediction-green.svg)

## Main findings

All coverage numbers below are for the model's 10th to 90th percentile interval, which claims **80%** coverage.

1. **The intervals under-cover, and this gets worse after the first day.**

   | Days ahead | Sensex coverage | Nifty coverage |
   |---:|---:|---:|
   | 1 | 80.7% | 80.6% |
   | 2 | 73.6% | 75.4% |
   | 16 | 74.1% | 70.7% |
   | 30 | 74.4% | 75.4% |

   Sensex: 352 forecast windows. Nifty: 403 forecast windows.

2. **Calibration depends on the volatility regime, and the pattern flips with the horizon (I call this the "Horizon Reversal Effect").**

   | | Low-vol coverage | High-vol coverage |
   |---|---:|---:|
   | Sensex, day 1 | 83.8% (n=197) | 76.8% (n=155) |
   | Sensex, day 30 | 70.1% | 80.0% |
   | Nifty, day 1 | 85.7% (n=223) | 74.4% (n=180) |
   | Nifty, day 30 | 71.7% | 80.0% |

   Low-vol minus high-vol coverage gap, with 95% bootstrap intervals (2,000 resamples):

   | | Day 1 | Day 30 |
   |---|---|---|
   | Sensex | +6.9 pts [-1.1, 15.1] | -9.9 pts [-18.9, -0.8] |
   | Nifty | +11.2 pts [3.6, 18.9] | -8.2 pts [-16.9, 0.1] |

   The sign flips within the first few days on both indices, and the two gap curves are correlated at 0.52. Not every gap is statistically strong: Sensex day 1 and Nifty day 30 have intervals that touch zero.

3. **Conformal correction helps, but does not reach the target.** On Sensex at 16 days ahead (211 calibration windows, 141 test windows):

   | Method | Coverage | Average width |
   |---|---:|---:|
   | Raw Chronos-2 | 75.2% | 5,152 |
   | Split conformal | 78.0% | 5,387 (+4.6%) |
   | Adaptive conformal (ACI, gamma = 0.01) | 78.7% | 5,630 |

   Both methods stay below 80% because market data is not exchangeable, so the finite-sample guarantee does not hold exactly.

4. **ACI showed no clear advantage right after regime shifts in this data.** Fixed and adaptive conformal both reached 80.0% coverage after the 26 detected shift points, and 86.7% after the 3 clean shifts (a very small sample).

5. **High-volatility intervals are much wider in relative terms.** At 30 days ahead, Sensex intervals are 15.3% of price in high-vol periods versus 10.7% in low-vol periods (+44%). Nifty is similar (15.8% vs 10.9%).

## Method

- **Forecaster:** `amazon/chronos-2`, zero-shot, 512-day context, 30-day horizon, 11 quantile levels (0.05 to 0.95), rolling windows with a step of 10 days.
- **Diagnostics:** empirical coverage by horizon and PIT histograms.
- **Regimes:** a day is "high-vol" if its 20-day rolling volatility is above the median, otherwise "low-vol".
- **Split conformal:** chronological 60/40 calibration/test split. Non-conformity score `s = max(q_lo - y, y - q_hi)`, and the intervals are widened by the target quantile of `s`.
- **ACI:** online update `alpha_t+1 = alpha_t + gamma * (alpha - miss_t)`.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_load_forecast` | Runs Chronos-2 over rolling Sensex windows |
| `02_calibration_check` | Coverage and PIT histograms at days 1, 2, 16 and 30 |
| `03_conformal_prediction` | Split conformal correction |
| `04_adaptive_conformal` | ACI |
| `05_final_comparison` | Side-by-side comparison |
| `06`, `07` | Same forecasts and calibration check for Nifty |
| `08` to `10` | Volatility regimes, regime-conditional coverage, bootstrap significance |
| `11` | Crossover analysis of the Horizon Reversal Effect |
| `12`, `13` | Recovery after regime shifts (all shifts, then clean shifts only) |
| `14` | Check that results hold in percent-of-price terms |

`baseline/` holds a classical SARIMAX forecaster and a script that runs the same calibration audit on it for comparison.

## Reproduce

```bash
pip install numpy pandas matplotlib torch chronos-forecasting jupyter
jupyter notebook notebooks/
```

Run the notebooks in order from inside the `notebooks/` folder. The forecasts used in the analysis are saved in `data/` as `.npy` files, so you can inspect the results without re-running the model. Notebooks 01 and 06 download the model and are slow on CPU.

## Limitations

- One model (Chronos-2), two indices, one chronological split, and small test sets (141 windows for the conformal comparison).
- The conformal guarantee assumes exchangeable data, which price series are not. That is why the corrected coverage still falls short of 80%.
- Forecast windows start every 10 days but look 30 days ahead, so neighbouring windows overlap. They are not independent, and the confidence intervals above are probably narrower than they should be.
- Regimes use a simple median-volatility rule.

## Author

Yash Chitransh, M.Sc. Mathematics & Scientific Computing, NIT Warangal. [LinkedIn](https://www.linkedin.com/in/yash-chitransh-4b108b2aa)

## License

MIT, see `LICENSE`.
