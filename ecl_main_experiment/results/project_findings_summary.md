# Project Findings Summary

**Project:** Comparative evaluation of forecasting models for multivariate electricity demand  
**Audience:** Avanade mentor / business-facing project review  
**Dataset:** ECL electricity benchmark, 321 client series, hourly observations  
**Evaluation setup:** 20 representative clients, horizons of 24h, 48h and 168h

---

## 1. Executive Summary

No single model is universally best. Model choice depends on the business
constraint: forecast horizon, available training resources, number of related
series, and the level of interpretability required for client trust.

The most important result is horizon-dependent. TabPFN-TS, used as an
exploratory zero-shot foundation-model baseline, achieved the lowest RMSE at
24h and 48h without task-specific training. At 168h, trained models regained
the advantage: LSTM performed best, with iTransformer very close behind.

For client-facing advisory work, the strongest practical recommendation is not
"use the most accurate model". It is to match the model to the operating
context. XGBoost remains attractive when interpretability and implementation
simplicity matter. iTransformer is most relevant when many related time series
must be forecast together. TabPFN-TS is compelling for rapid prototyping and
short-horizon forecasting, but should be positioned as an exploratory
foundation-model option rather than a traditional baseline.

---

## 2. Key Quantitative Findings

### Forecast Accuracy by Horizon

| Horizon | Best model | RMSE | Interpretation |
|---:|---|---:|---|
| 24h | TabPFN-TS | 1,153.0 | Best short-horizon accuracy with zero task-specific training |
| 48h | TabPFN-TS | 1,363.6 | Maintains short-horizon advantage |
| 168h | LSTM | 3,324.4 | Trained sequence model performs best at one-week horizon |

At 168h, iTransformer was very close to LSTM:

| Model | 168h RMSE |
|---|---:|
| LSTM | 3,324.4 |
| iTransformer | 3,361.2 |
| TabPFN-TS | 3,597.9 |

This suggests that zero-shot forecasting may be especially valuable for short
horizons, while longer-horizon forecasting benefits from task-specific training
on the target dataset.

### Training Cost

| Model | Total training time | Interpretation |
|---|---:|---|
| TabPFN-TS | 0.0s | Zero-shot; no model training |
| XGBoost | 15.5s | Fastest trained baseline across 20 client models |
| ARIMA | 230.8s | Transparent but slower and less accurate |
| iTransformer | 1,096.9s | One shared model trained on all 321 series |
| LSTM | 4,544.1s | 20 independent client-specific models |

iTransformer has a larger single-model training cost than XGBoost or ARIMA, but
its cost should be interpreted differently: it trains one shared multivariate
model across all 321 client series, while ARIMA, XGBoost and LSTM are trained
as separate per-client models.

---

## 3. Weather Extension Experiment

To test whether the model rankings generalise when richer input information is available, the same five models were re-run on the ECL dataset augmented with 15 exogenous features: five Lisbon weather variables (temperature, humidity, precipitation, heating degree, cooling degree) fetched via the Open-Meteo API, plus ten calendar features (hour-of-day, day-of-week, month, weekend flag, and their cyclical sin/cos encodings).

ARIMA cannot accept exogenous covariates in the configuration used, so it serves as the load-only baseline. The four other models received the weather and calendar features as additional inputs alongside the historical load series.

### Weather Extension — RMSE by Model and Horizon

Values from corrected result files (raw_metrics/, 2026-07-25/26). Old (misaligned 2012-axis) files superseded; old XGBoost 168h was 5,109.7 kWh.

| Model | RMSE 24h | RMSE 48h | RMSE 168h | Setting |
|---|---:|---:|---:|---|
| ARIMA | 4,352.5 | 4,707.9 | 7,277.5 | Load-only baseline |
| XGBoost | 1,473.0 | 2,204.0 | 4,822.6 | Past weather + calendar exog |
| LSTM | 2,110.2 | 2,094.4 | 3,322.7 | Past weather + calendar exog |
| iTransformer | **1,132.3** | **1,233.4** | **3,067.3** | Past weather + calendar exog |
| TabPFN-TS | 1,035.3 | 1,274.0 | 3,329.0 | Past weather + calendar exog |

### What changed vs the main ECL experiment

Weather covariates reduced RMSE for XGBoost, iTransformer, and TabPFN-TS at all horizons. LSTM's RMSE increased at 24h and 48h under the weather setting — weather features did not help LSTM at short horizons (negative values = improvement; positive = degradation):

| Model | 24h RMSE change | 48h RMSE change | 168h RMSE change |
|---|---:|---:|---:|
| XGBoost | −1,662.1 (−53%) | −1,534.2 (−41%) | −1,511.1 (−24%) |
| LSTM | **+397.9 (+23%)** | **+261.5 (+14%)** | −1.7 (0%) |
| iTransformer | −249.6 (−18%) | −312.4 (−20%) | −293.9 (−9%) |
| TabPFN-TS | −117.7 (−10%) | −89.5 (−7%) | −268.9 (−7%) |

XGBoost gains the most from weather features at all horizons. iTransformer and TabPFN-TS improve consistently. LSTM deteriorates at short horizons, likely because the sequential architecture struggles to incorporate the additional exogenous channels from its training window alignment.

### Main conclusion

Under the weather extension, the ranking shifts at 48h compared to the main ECL experiment:
- **TabPFN-TS** remains best at 24h (RMSE 1,035.3).
- **iTransformer** is best at 48h (RMSE 1,233.4, beats TabPFN 1,274.0) and at 168h (RMSE 3,067.3).
- **ARIMA** (load-only) is comfortably the weakest across all horizons.
- **LSTM** is the only model that performed worse with weather covariates at 24h and 48h.

This suggests that iTransformer benefits most from multi-variate context at medium-to-long horizons, while TabPFN-TS retains its short-horizon advantage even in the weather setting.

---

## 4. Business Recommendations

| Business situation | Recommended model | Rationale |
|---|---|---|
| Rapid prototype or low training resource | TabPFN-TS | Zero task-specific training; strongest 24h/48h performance in this experiment |
| Long-horizon planning, one week ahead | LSTM or iTransformer | Trained models outperform TabPFN-TS at 168h |
| Many related customer or asset series | iTransformer | One shared model can use cross-series structure across 321 variables |
| Need explainable client-facing drivers | XGBoost | SHAP can explain lag, rolling and calendar feature contributions |
| Audit-first or highly regulated context | ARIMA | Most transparent coefficients and assumptions, despite weaker accuracy |
| Accuracy-focused deployment with less need for explanation | LSTM | Strong long-horizon accuracy, but harder to explain |

Recommended positioning for Avanade-style client work:

1. Use **TabPFN-TS** as a rapid benchmark or short-horizon prototype.
2. Use **XGBoost** when the client needs an interpretable, fast and robust
   baseline.
3. Use **iTransformer** when the client has many related series and wants a
   scalable multivariate forecasting architecture.
4. Use **LSTM/iTransformer** for longer-horizon forecasting where task-specific
   learning is more important.
5. Keep **ARIMA** as a transparent statistical baseline, not as the main
   accuracy-driven recommendation.

---

## 5. Interpretability Implications

Interpretability is not uniform across models. The strongest business-facing
model is not always the model with the lowest RMSE; it is the model whose
predictions can be explained at the level the client requires.

| Model | Interpretability approach | Business readability | Practical caveat |
|---|---|---|---|
| ARIMA | AR/MA coefficients and residual diagnostics | High | Transparent but lower accuracy on this dataset |
| XGBoost | SHAP feature attribution on lag, rolling and calendar features | Medium-High | Best balance between accuracy, speed and explanation |
| LSTM | Input-window or permutation sensitivity analysis | Low | Accurate, but explanations are indirect |
| iTransformer | Cross-series ablation or attention diagnostics | Medium | Attention should be treated as diagnostic, not causal proof |
| TabPFN-TS | Context-length sensitivity analysis | Medium | Explain through historical context dependence, not standard feature importance |

### Most client-ready interpretability option

**XGBoost is the most practical explainable model for client-facing work.**
It supports SHAP analysis, which can translate technical features into business
drivers such as:

- demand 24 hours ago,
- demand one week ago,
- recent rolling average,
- hour of day,
- weekend effect.

This makes XGBoost useful even when it is not the most accurate model, because
it can answer the client question: **"Why did the model forecast this?"**

### Most transparent baseline

**ARIMA is the most transparent model**, because its parameters and assumptions
are directly inspectable. However, in this experiment it had the weakest
accuracy across all horizons. It is therefore best used as an audit-friendly
baseline rather than the recommended production model.

### Most important interpretability warning

For **iTransformer**, attention diagnostics can be useful for investigating
cross-series relationships, but they should not be presented as causal
explanations. A safer framing is:

> The diagnostic indicates which series the model appears to use more strongly
> during forecasting, but it does not prove that one client's demand causes
> another client's demand.

For **TabPFN-TS**, standard feature importance is not the right explanation in
the current setup because the model is used through historical context rather
than a manually engineered feature table. A more defensible explanation is
context sensitivity: testing how forecast performance changes when the
historical context window is shortened or lengthened.

---

## 6. Robustness and Quality Assurance

Client 313 is an extreme high-load client and was the hardest case across all
models. Removing Client 313 reduced average RMSE, but it did not change the
model ranking at any horizon. This supports the conclusion that the main
findings are not driven by a single outlier client.

The evaluation also identified and corrected a forecast-origin alignment issue
in two pipelines, where predictions were initially generated from the end of
the training split instead of the boundary immediately before the test split.
This was corrected by using train+validation history as the forecast context.
The remaining model scripts were audited for the same issue.

This matters for client work because time-series evaluation bugs can silently
produce incorrect metrics without throwing runtime errors. Forecast-origin
audits should be part of any production forecasting validation process.

---

## 7. Caveats

- The ECL dataset is a public electricity benchmark, not a proprietary Avanade
  client dataset. Results should be validated on client-specific data before
  making deployment decisions.
- TabPFN-TS is an exploratory zero-shot foundation-model baseline, not a
  traditional model and not part of the original dissertation baseline set.
- Training-time comparisons are hardware-dependent. The iTransformer and LSTM
  runs were affected by available GPU hardware, so cost conclusions should be
  interpreted directionally rather than as universal timings.
- Accuracy metrics are based on RMSE and MAE. Commercial deployments may also
  require cost-weighted error metrics, service-level penalties or peak-demand
  risk measures.
- Interpretability requirements vary by client. For regulated clients,
  a slightly less accurate but more explainable model may be preferable.

---

## 8. Recommended Next Steps

1. **Run XGBoost SHAP analysis**  
   This is the highest-value interpretability extension because it provides
   client-readable explanations of forecast drivers.

2. **Add context sensitivity for TabPFN-TS**  
   Compare context windows such as 24h, 48h, 96h and 168h to explain how much
   history the zero-shot model needs.

3. **Add cross-series diagnostics for iTransformer**  
   Use ablation or attention-based diagnostics carefully, framing them as
   dependency diagnostics rather than causal explanations.

4. **Validate on a client-style dataset**  
   Repeat the model comparison on a dataset closer to Avanade client use cases,
   especially where business costs and interpretability requirements are known.

5. **Create a client-facing model selection playbook**  
   Convert the decision matrix into a reusable consulting artifact: business
   constraints in, recommended model family out.

