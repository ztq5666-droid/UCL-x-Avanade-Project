# ECL + Lisbon Weather Covariates Experiment

## 1. Purpose

This experiment extends the ECL electricity load forecasting dataset by adding
real Lisbon weather and calendar covariates fetched from the Open-Meteo
Historical Weather API.

The target remains electricity consumption. The new independent variables are
shared regional weather and calendar predictors that create a richer
weather-augmented forecasting task, enabling a side-by-side model comparison
under richer input conditions.

Research focus:

```text
When the ECL forecasting task is augmented with shared regional weather and
calendar covariates, do Transformer-based models handle the richer
multivariate/exogenous setting better than traditional approaches?
```

This experiment is intentionally separate from the main ECL baseline
(`ecl_main_experiment/`). The existing load-only model scripts and result
files are not modified.

## 2. Why Lisbon Weather Is Used

The ECL dataset is based on Portuguese electricity consumption, but the public
benchmark version does not provide exact client-level geographic locations.
Lisbon weather is therefore used as a representative regional weather proxy.

These weather variables should be interpreted as shared regional covariates,
not client-specific local weather measurements. This is a reasonable but
imperfect modelling assumption: it allows a controlled exogenous-variable
experiment while acknowledging that local weather variation across clients
cannot be captured.

## 3. Added Variables

Weather variables downloaded from the Open-Meteo Historical Weather API:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`

Derived weather variables:

- `heating_degree` = `max(0, 18 - temperature_2m)`
- `cooling_degree` = `max(0, temperature_2m - 22)`

Calendar variables:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `hour_sin`, `hour_cos`
- `dayofweek_sin`, `dayofweek_cos`
- `month_sin`, `month_cos`

Total exogenous features: **15**

## 4. Data Pipeline

Run `scripts/build_ecl_lisbon_weather_features.py` to build the dataset:

```bash
python scripts/build_ecl_lisbon_weather_features.py
```

What it does:

1. Load the original ECL dataset from
   `../ecl_main_experiment/data/ECL/electricity.csv`.
2. Align the 26,304-row sequence to the standard 2012-01-01 to 2014-12-31
   hourly physical period.
3. Download real hourly Lisbon weather from Open-Meteo for that physical period.
4. Compute derived weather features and calendar encodings.
5. Merge everything into a single output file with one `date` column
   (physical 2012-2014 period).
6. Write outputs to `data/`:
   - `data/lisbon_weather_hourly.csv` — weather-only intermediate file
   - `data/electricity_lisbon_weather.csv` — final merged dataset (337 cols)

The original ECL file is never modified.

## 5. Experimental Framing

The original ECL experiment is a load-only time series forecasting task.

This weather-augmented experiment changes the input setting to:

```text
past load + past shared regional weather/calendar covariates -> future load
```

To keep the model comparison fair, all active model scripts use:

- the same weather-augmented dataset (`data/electricity_lisbon_weather.csv`)
- the same chronological 70% / 10% / 20% split
- the same selected 20 clients
- the same forecast horizons: 24h, 48h, 168h
- the same metrics: MSE, MAE, RMSE
- **past exogenous context only** — no future observed weather values

ARIMA does not consume weather/calendar covariates and serves as the
load-only statistical benchmark.

## 6. Folder Structure

```
ecl_weather_covariates_experiment/
├── data/                          # Generated dataset files (gitignored)
│   ├── electricity_lisbon_weather.csv
│   └── lisbon_weather_hourly.csv
├── models/
│   ├── common/common.py           # Shared dataset loader and utilities
│   ├── arima/                     # Load-only ARIMA baseline
│   ├── xgboost/                   # XGBoost with past weather features
│   ├── lstm/                      # LSTM with weather input channels
│   ├── itransformer/              # iTransformer with all variables
│   └── tabpfn/                    # TabPFN-TS with past weather features
├── scripts/
│   └── build_ecl_lisbon_weather_features.py
├── analysis/
│   ├── generate_result_figures.py
│   └── generate_weather_figures.py
├── outputs/
│   ├── raw_metrics/               # Per-model result CSVs
│   ├── figures/                   # All generated figures
│   └── validation_summary.md
└── README.md
```

## 7. Limitations

Lisbon weather is a regional proxy. It does not represent exact client-level
weather conditions because the ECL benchmark does not provide client locations.

The focused weather feature set (temperature, humidity, precipitation, derived
degree days, calendar) keeps the experiment interpretable. Variables such as
cloud cover or wind speed could be added in a future robustness extension.

## 8. Regenerating Figures

```bash
# Rebuild dataset (only needed if data/ is missing)
python scripts/build_ecl_lisbon_weather_features.py

# Regenerate comparison figures
python analysis/generate_result_figures.py

# Regenerate weather extension figures
python analysis/generate_weather_figures.py
```
