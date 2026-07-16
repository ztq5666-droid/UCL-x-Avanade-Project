# ECL + Lisbon Weather Validation Summary

## Scope

This validation summary applies only to the independent weather covariate
experiment in `ecl_weather_covariates_experiment/`.

No files inside the original `ecl_main_experiment/` project were modified by
the weather data generation or merge process.

## Output Dataset

The build script (`scripts/build_ecl_lisbon_weather_features.py`) internally
aligns Lisbon weather to the standard 2012-2014 ECL physical period, then
writes a single merged file with one datetime column.

Final output file: `data/electricity_lisbon_weather.csv`

| Field | Value |
|---|---|
| Rows | 26,304 |
| Columns | 337 (1 date + 321 load + 15 exog) |
| Date column | `date` (physical 2012-01-01 to 2014-12-31, hourly) |
| Duplicate dates | 0 |
| Hourly continuity | Yes |
| Missing values in exog covariates | 0 |
| Total missing values | 0 |

## Weather Data

Weather source:

- Open-Meteo Historical Weather API
- Location: Lisbon, Portugal
- Latitude: 38.7223
- Longitude: -9.1393
- Timezone: Europe/Lisbon
- Period: 2012-01-01 00:00:00 to 2014-12-31 23:00:00
- Frequency: hourly

Weather intermediate file: `data/lisbon_weather_hourly.csv`

| Field | Value |
|---|---:|
| Rows | 26,304 |
| Columns | 6 |
| Datetime column | `physical_datetime` (internal, not in final merged output) |
| Duplicate datetimes | 0 |
| Hourly continuity | Yes |
| Missing values | 0 |

## Added Variables

Weather variables:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `heating_degree` (`max(0, 18 - temperature_2m)`)
- `cooling_degree` (`max(0, temperature_2m - 22)`)

Calendar variables:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `dayofweek_sin`
- `dayofweek_cos`
- `month_sin`
- `month_cos`

## Interpretation Note

The Lisbon weather variables are shared regional exogenous covariates. They are
not client-specific local weather measurements because the public ECL benchmark
does not provide exact client locations.

This design is suitable for creating a richer weather-augmented ECL forecasting
task for model-family comparison. It should not be interpreted as a precise
customer-level weather exposure model or as a standalone test of whether
weather improves forecasting accuracy.
