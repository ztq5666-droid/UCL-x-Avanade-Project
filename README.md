# UCL × Avanade Project — Multivariate Electricity Demand Forecasting

**Author:** Tianqi (MSc Business Analytics, UCL × Avanade, 2026)  
**Dataset:** ECL Electricity Load Diagrams 2016–2019 (321-client benchmark version)

---

## Overview

This repository contains the full experimental code and results for a dissertation comparing five forecasting pipelines on the ECL electricity benchmark:

| Pipeline | Paradigm |
|---|---|
| Seasonal Naïve | Deterministic benchmark |
| ARIMA | Classical statistical |
| XGBoost | Gradient boosting |
| LSTM | Recurrent neural network |
| iTransformer | Cross-variate Transformer |
| TabPFN-TS | Zero-shot foundation model |

**Two experiments:**  
- **Experiment 1** — Load history + calendar features (all six pipelines)  
- **Experiment 2** — + past Lisbon weather covariates (four pipelines; ARIMA and seasonal naïve unchanged)

**Three forecast horizons:** 24h, 48h, 168h (one week)  
**Evaluation panel:** 20 clients — top 10 by mean consumption + 10 randomly selected (seed = 42)

---

## Repository Structure

```
ecl_main_experiment/          Experiment 1: load history comparison
├── data/README.md            How to obtain electricity.csv (not tracked — 100 MB)
├── eda/                      8-step exploratory data analysis
│   ├── build_notebook.py
│   ├── eda_ecl.ipynb
│   └── figures/              8 EDA output figures
├── models/                   Training scripts (one per pipeline)
│   ├── arima/train_arima.py
│   ├── lstm/train_lstm.py
│   ├── itransformer/train_itransformer.py
│   ├── tabpfn/train_tabpfn.py
│   └── xgboost/train_xgboost.py
├── analysis/                 Post-training evaluation scripts
│   ├── 01b_seasonal_naive_pointwise.py
│   └── create_comparison_figures.py
└── results/
    ├── figures_final/        Final dissertation figures (Figs 4.1–4.6)
    ├── raw_metrics/          Per-client RMSE/MAE/NMAE CSVs for all pipelines
    └── summary_tables/       Aggregated summary tables

ecl_weather_covariates_experiment/   Experiment 2: weather covariates extension
├── README.md
├── data/README.md            How to regenerate the merged weather dataset
├── scripts/                  Dataset build script (fetches Open-Meteo, merges ECL)
├── models/                   Training scripts (exogenous variants)
├── analysis/                 Evaluation and figure-generation scripts
└── outputs/
    ├── figures_final/        Final dissertation figures (weather experiment)
    ├── raw_metrics/          Result CSVs including bootstrap comparisons
    └── model_checkpoints/    iTransformer column index (model weights gitignored)

dissertation/
└── dissertation_final.docx   Final dissertation submitted to UCL

project_summary/
└── summary.md                Project overview and key findings
```

---

## Data

The raw ECL dataset (`electricity.csv`, ~100 MB) is not tracked by this repository. Download the benchmark-ready file from the [official iTransformer repository](https://github.com/thuml/iTransformer) and place it at:

```
ecl_main_experiment/data/ECL/electricity.csv
```

See `ecl_weather_covariates_experiment/data/README.md` for instructions on regenerating the merged weather dataset (requires internet access; uses the free Open-Meteo API, no key needed).

---

## Key Results

| Horizon | Best RMSE (Exp 1) | Best RMSE (Exp 2) |
|---|---|---|
| 24h | TabPFN-TS | iTransformer |
| 48h | TabPFN-TS | iTransformer |
| 168h | LSTM | iTransformer |

The weekly seasonal naïve benchmark remains competitive at all horizons, including outperforming several trained pipelines under normalised error (NMAE). iTransformer's advantage is clearest at shorter horizons when weather covariates are available.

---

## Reproduction

Each training script is self-contained. Run from the repository root with the ECL dataset in place:

```bash
# Experiment 1 — example
python ecl_main_experiment/models/arima/train_arima.py
python ecl_main_experiment/models/lstm/train_lstm.py

# Experiment 2 — first build the merged dataset
cd ecl_weather_covariates_experiment
python scripts/build_ecl_lisbon_weather_features.py
python models/xgboost/train_xgboost_exog_past.py
```

Deep-learning pipelines (LSTM, iTransformer) require GPU. The experiments were run on an Azure ML instance with a Tesla M60 GPU. See `ecl_main_experiment/models/itransformer/` for the iTransformer configuration; the official THUML implementation must be cloned separately (see Appendix C of the dissertation).
