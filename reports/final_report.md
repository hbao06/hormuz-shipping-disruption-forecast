# Strait of Hormuz Maritime Disruption — Final Report

**Project:** Hormuz Shipping Disruption Forecasting
**Date:** June 2026
**Author:** [Your Name]
**Status:** Analysis Complete

---

## 1. Executive Summary

The Strait of Hormuz has experienced a near-total collapse in daily ship transits since a geopolitical crisis began in late February 2026 — from a stable baseline of ~103 transits/day to approximately 5–8 transits/day, a 94% reduction. Forecasting models project continued suppression over the next 15 days with no statistically observable recovery signal. Stakeholders in maritime insurance, shipping operations, and energy supply chains should plan around a sustained disruption scenario and establish weekly model refresh cycles as the primary early-warning mechanism.

---

## 2. Background

The Strait of Hormuz is the world's most critical maritime oil chokepoint, with approximately 20–21% of global oil supply and significant LNG volumes transiting daily. A regional escalation beginning in February 2026 triggered a series of vessel attacks, carrier suspensions, and routing changes that effectively closed the strait to commercial traffic.

This analysis was conducted on 125 daily observations (January 1 – May 5, 2026) covering ship transit volumes, energy throughput, oil prices, security incident counts, and carrier operational status.

> **Note:** This dataset represents a hypothetical scenario constructed for analytical and portfolio demonstration purposes.

---

## 3. Data Overview

| Attribute               | Detail                                              |
| ----------------------- | --------------------------------------------------- |
| Observations            | 125 daily records                                   |
| Date range              | Jan 1, 2026 – May 5, 2026                           |
| Target variable         | `daily_ship_transits`                               |
| Total columns           | 26                                                  |
| Missing values (target) | 0                                                   |
| Period types            | `pre_war` (early period), `war_crisis` (post-onset) |

Key variable groups: ship/vessel traffic, energy throughput (oil/LNG), price indices (Brent, WTI), security incidents, carrier operational status.

---

## 4. Key Trends

### Pre-War Period

- Daily transits: stable at ~103/day (std = 4.3)
- Negligible volatility; highly predictable
- Energy throughput consistent with full operational capacity

### Crisis Period

- Daily transits: collapsed to ~5.8/day (std = 5.9 on a near-zero base)
- Absolute volatility remained elevated despite low absolute levels
- Brent crude rose approximately 35% from pre-crisis baseline
- All major carriers (Maersk, CMA CGM, others) suspended or rerouted operations
- Vessel attacks accumulating at approximately 3–4 per spike day

### Structural Break

- The transition is a discrete step-change, not a gradual decline
- Statistically confirmed by ADF test (p > 0.05 on original, p < 0.05 on first difference) and PELT change-point detection (detected break aligns with `war_onset` event date)
- Weekly seasonality is present but negligible relative to the break magnitude

---

## 5. Methodology

### Analytical Pipeline

1. **Exploratory Data Analysis** — missingness, quality checks, distribution analysis, rolling statistics
2. **Time Series Diagnostics** — ADF, KPSS, ACF/PACF, STL decomposition, structural break detection (PELT), cross-correlation analysis
3. **Feature Engineering** — lag features, rolling statistics, EWM features, calendar features, exogenous regressors (regime dummy, attack counts)
4. **Modeling** — 8 models benchmarked on a proper chronological train/test split (110 train / 15 test)
5. **Evaluation** — MAE, RMSE, MAPE, Directional Accuracy; walk-forward cross-validation; residual diagnostics (Q-Q, Ljung-Box)
6. **Forecasting** — 15-day forward forecast with 95% confidence intervals, crisis-continuation scenario

### Models Evaluated

| #   | Model                  | Notes                                  |
| --- | ---------------------- | -------------------------------------- |
| 1   | Naive                  | Last-value-forward baseline            |
| 2   | Moving Average (7d)    | Rolling-mean baseline                  |
| 3   | Simple Exp. Smoothing  | Level-adaptive, no trend               |
| 4   | Holt-Winters           | Trend + seasonal components            |
| 5   | ARIMA(1,1,1)           | Classical autoregressive               |
| 6   | SARIMA(1,1,1)(1,1,1,7) | Adds weekly seasonal terms             |
| 7   | Auto-ARIMA             | Stepwise AIC-optimized order           |
| 8   | SARIMAX + exog         | Regime dummy + attack count regressors |

---

## 6. Forecast Results

**Selected model:** SARIMAX(1,1,1) with exogenous regressors (war dummy + cumulative attack count)

**Test set performance (15-day held-out period):**
| Metric | Value |
|---|---|
| MAE | [run notebook 04 for value] |
| RMSE | [run notebook 04 for value] |
| MAPE | [run notebook 04 for value] |
| Directional Accuracy | [run notebook 04 for value] |

**15-Day Forward Forecast (Crisis-Continuation Scenario):**

- Forecast range: 3–10 transits/day
- Mean forecast: ~5–7 transits/day
- 95% CI: wide, reflecting genuine crisis-period uncertainty
- No recovery signal present in the statistical forecast

Key chart: `images/15day_forward_forecast.png`

---

## 7. Business Risks

| Risk                                 | Likelihood               | Impact          | Notes                                                                |
| ------------------------------------ | ------------------------ | --------------- | -------------------------------------------------------------------- |
| Sustained closure (4+ weeks)         | High (based on forecast) | Severe          | Current baseline scenario                                            |
| Sudden de-escalation                 | Moderate                 | High (positive) | Cannot be forecast statistically — monitor geopolitical intelligence |
| Further escalation                   | Moderate                 | Severe          | Wider CI upper bound; stress-test budgets accordingly                |
| Fleet repositioning lag at reopening | Near-certain             | Moderate        | Plan 5–10 day lag even after reopening announcement                  |

---

## 8. Strategic Recommendations

**Maritime Insurers**

- Maintain war-risk premium elevation consistent with sub-10 transit/day forecasts
- Implement a weekly model refresh and pricing review cycle
- Use daily attack count as a trigger: 7+ days of zero new attacks → initiate premium review

**Shipping Operators**

- Commit to Cape of Good Hope rerouting for minimum 3–4 additional weeks
- Do not pre-position vessels for strait transit without confirmed de-escalation
- Build 5–10 day repositioning lag into any recovery scenario logistics planning

**Energy & Supply Chain**

- Activate alternative sourcing immediately; do not wait for multi-week trend confirmation
- Stress-test Q3/Q4 procurement budgets against $100–110/bbl sustained Brent
- Drawdown from strategic reserves should be authorized now rather than reactively

**Risk & Intelligence Teams**

- Deploy the SARIMAX model on a weekly refresh cadence (new data → refit → re-forecast)
- Add a "resolution scenario" branch (war dummy = 0 in 2 weeks) for range-of-outcomes planning
- Integrate real-time carrier status API feeds as complementary exogenous signals

---

## 9. Limitations

1. **Dataset is a constructed scenario** — not live operational data; findings should be calibrated against real data before operational deployment
2. **125-day window** — limited history for estimating seasonal patterns or long-run behavior
3. **Cannot forecast black swans** — sudden ceasefire, escalation, or third-party intervention are not predictable from the time series alone
4. **Exogenous assumptions in forward forecast** — future attack counts are extrapolated at the recent average rate; actual trajectory may differ
5. **Single-scenario forecast** — results reflect crisis continuation only; a scenario analysis with resolution and escalation branches is recommended for decision support

---

## 10. Next Steps

- [ ] Integrate live transit count data feed for weekly model refresh
- [ ] Add resolution scenario: re-run with war dummy flipping to 0 at various future dates
- [ ] Build Streamlit dashboard for weekly stakeholder distribution
- [ ] Extend feature engineering with Prophet-style holiday/event markers for major incident dates
- [ ] Collect 12+ months of historical data (including previous Hormuz tension episodes) to improve seasonal estimation
- [ ] Add XGBoost/LightGBM with full lag feature matrix as alternative production model

---

_Report generated from notebooks/04_forecast_and_evaluation.ipynb_
_Full methodology: see notebooks 01–04 and src/ modules_
