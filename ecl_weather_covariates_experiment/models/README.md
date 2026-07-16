# Exogenous Model Scripts

These scripts mirror the structure of `ecl_main_experiment/models/` while remaining
fully inside the independent `ecl_weather_covariates_experiment/` folder.

The active modelling objective is to compare model families on the same
weather-augmented ECL forecasting task, not simply to test whether weather
improves accuracy.

Active setting:

```text
past load + past shared regional weather/calendar covariates -> future load
```

No original `ecl_main_experiment/` model scripts are modified.

## Shared Design

All scripts use:

- `data/electricity_lisbon_weather.csv`
- `date` column (physical 2012-2014 period) for chronological ordering
- 70% / 10% / 20% chronological train/validation/test split
- 20 clients selected with the same rule as the main experiment:
  top 10 by train-set mean load plus 10 random remaining clients with seed 42
- Forecast horizons: 24h, 48h, 168h
- Metrics: MSE, MAE, RMSE in original electricity-load units

Shared utilities live in:

- `common/common.py`

The shared loader checks that:

- total columns = 337 (1 date + 321 load + 15 exog)
- electricity load columns = 321
- exogenous columns = 15
- `date` is continuous hourly (2012-2014 physical period)
- forecast origin is aligned at the val/test boundary

## Critical Forecast-Origin Rule

Fixed-origin forecasts must start at the beginning of the test split. Because
the data split is `[train] -> [validation] -> [test]`, the model context for
test evaluation must include the validation period.

Using only the end of the training split as context would forecast the
validation period, then compare those predictions against test actuals. That
bug produces valid-looking but time-misaligned RMSE/MAE values.

The shared dry-run check asserts:

```text
val_end + 1 hour == test_start
```

## Scripts

| Model | Script | Main Difference From Load-Only Main Experiment Version |
|---|---|---|
| ARIMA load-only | `arima/train_arima_load_only.py` | Statistical load-only benchmark using the same split, clients and horizons. |
| XGBoost past-weather | `xgboost/train_xgboost_exog_past.py` | Uses past weather/calendar lag/rolling features. |
| LSTM | `lstm/train_lstm_exog.py` | Adds historical weather/calendar covariates as input channels. |
| iTransformer-style | `itransformer/train_itransformer_exog.py` | Uses all load variables plus exogenous variables in an inverted-variable Transformer. |
| TabPFN | `tabpfn/train_tabpfn_exog_past.py` | Uses tabular past load/weather/calendar lag and rolling features. |

## Dry Run

Before training, each script can validate paths, split sizes, selected clients
and datetime ranges without fitting a model:

```bash
python models/arima/train_arima_load_only.py --dry-run
python models/xgboost/train_xgboost_exog_past.py --dry-run
python models/lstm/train_lstm_exog.py --dry-run
python models/itransformer/train_itransformer_exog.py --dry-run
python models/tabpfn/train_tabpfn_exog_past.py --dry-run
```

## Fair Information Setting

Use this setting when comparing model structures:

- `xgboost/train_xgboost_exog_past.py`
- `lstm/train_lstm_exog.py`
- `itransformer/train_itransformer_exog.py`
- `tabpfn/train_tabpfn_exog_past.py`

These scripts do not use future observed exogenous values from the test period.
The current active setting uses past exogenous context only, including
historical weather and historical calendar covariates.

`arima/train_arima_load_only.py` is also active, but it is labelled separately
as a load-only statistical benchmark because ARIMA does not consume the
weather/calendar covariates in the fair past-information setting.

The Lisbon weather variables are shared regional covariates, not
client-specific local weather measurements.

## Interpretability Outputs

The active XGBoost exogenous script writes feature importance under
`outputs/raw_metrics/`:

- `xgboost_exog_past_weather_feature_importance.csv`

Result figures can be regenerated with:

```bash
python analysis/generate_result_figures.py
python analysis/generate_weather_figures.py
```
