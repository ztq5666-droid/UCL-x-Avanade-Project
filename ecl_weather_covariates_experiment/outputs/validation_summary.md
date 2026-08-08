# ECL + Lisbon Weather Validation Summary

## Scope

This validation summary applies only to the independent weather covariate
experiment in `ecl_weather_covariates_experiment/`.

No files inside the original `ecl_main_experiment/` project were modified by
the weather data generation or merge process.

## Output Dataset

The build script (`scripts/build_ecl_lisbon_weather_features.py`) parses the
real ECL timestamps from the source file, detects their date range
(2016-07-01 to 2019-07-02), and fetches matching Lisbon weather from
Open-Meteo for the same period. All merging is done on the actual ECL
timestamps.

Final output file: `data/electricity_lisbon_weather.csv`

| Field | Value |
|---|---|
| Rows | 26,304 |
| Columns | 337 (1 date + 321 load + 15 exog) |
| Date column | `benchmark_datetime` (ECL actual timestamps, 2016-07-01 to 2019-07-02, hourly) |
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
- Period: 2016-07-01 to 2019-07-02 (aligned to ECL benchmark timestamps)
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
