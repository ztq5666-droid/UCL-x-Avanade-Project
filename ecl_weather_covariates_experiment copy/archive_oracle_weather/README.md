# Archived Oracle-Weather Extensions

This folder contains scripts and outputs that use future observed Lisbon
weather from the test period.

They are archived because the active weather-augmented experiment is now framed
as a fair model-comparison task:

```text
past load + past weather/calendar covariates -> future load
```

Oracle-weather scripts answer a different question:

```text
If future observed weather were available, what upper-bound performance could
some models achieve?
```

That is useful as a business or weather-forecast-informed extension, but it
should not be directly compared with active past-weather-only results as if the
models received the same information.

Archived scripts:

- `models/arima/train_arimax_exog.py`
- `models/xgboost/train_xgboost_exog.py`
- `models/tabpfn/train_tabpfn_exog.py`

Archived outputs:

- `raw_metrics/arimax_exog_oracle_weather_results.csv`
- `raw_metrics/xgboost_exog_oracle_weather_results.csv`
- `raw_metrics/xgboost_exog_oracle_weather_feature_importance.csv`

The TabPFN oracle script was not run in the current local environment because
the required TabPFN packages were not available.
