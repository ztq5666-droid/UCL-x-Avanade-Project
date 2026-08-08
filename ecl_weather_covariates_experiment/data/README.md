# Data — ECL Weather Covariates Experiment

This folder holds two generated dataset files that are **not tracked by Git**
because of file size (54 MB and 1.2 MB respectively). Both files can be fully
reproduced by running one script.

---

## Files in this folder

### `electricity_lisbon_weather.csv` (54 MB — not on GitHub)

The primary merged dataset for this experiment. It contains all 321 ECL client
load series (26,304 hourly rows) plus 15 exogenous weather and calendar columns.

This file is generated automatically — do not download it separately.

**Why it is not committed:** At 54 MB it would consume a disproportionate share
of the repository's storage and is a fully deterministic output of the build
script. Committing generated files that can be reproduced in under a minute
adds no reproducibility value and makes the repository harder to clone.

### `lisbon_weather_hourly.csv` (1.2 MB — not on GitHub)

Intermediate file containing only the raw Lisbon weather series fetched from
the Open-Meteo Historical Weather API. Stored as a build artefact so that
the API does not need to be called again during subsequent runs.

**Why it is not committed:** It is a raw API download that can be re-fetched
at any time. Keeping fetched API responses out of version control is standard
practice.

---

## How to regenerate both files

You need the main ECL dataset first. If you have not already done so, follow
the instructions in `../ecl_main_experiment/data/README.md` to place
`electricity.csv` at `ecl_main_experiment/data/ECL/electricity.csv`.

Then, from the repository root, run:

```bash
cd ecl_weather_covariates_experiment
python scripts/build_ecl_lisbon_weather_features.py
```

The script will:

1. Load `../ecl_main_experiment/data/ECL/electricity.csv`
2. Assign the correct physical datetime axis (2016-07-01 to 2019-07-01) that
   matches the actual ECL benchmark timestamps
3. Fetch real hourly Lisbon weather from the Open-Meteo API for that period
4. Compute heating/cooling degree days and cyclic calendar encodings
5. Write `data/lisbon_weather_hourly.csv` (weather intermediate)
6. Write `data/electricity_lisbon_weather.csv` (final merged dataset, 338 cols)

Expected runtime: under 60 seconds (dominated by the Open-Meteo API request).
No API key is required — Open-Meteo is a free public API.

---

## Column layout of `electricity_lisbon_weather.csv`

| Columns | Content |
|---|---|
| `date` | Hourly timestamp (physical datetime axis, 2016-07-01 to 2019-07-01) |
| `MT_001` … `MT_321` | ECL client load series (321 columns, kWh) |
| `temperature_2m` | Air temperature at 2 m (°C) |
| `relative_humidity_2m` | Relative humidity at 2 m (%) |
| `precipitation` | Hourly precipitation (mm) |
| `heating_degree` | max(0, 18 − temperature) |
| `cooling_degree` | max(0, temperature − 22) |
| `hour`, `day_of_week`, `month`, `is_weekend` | Calendar features |
| `hour_sin/cos`, `dayofweek_sin/cos`, `month_sin/cos` | Cyclic encodings |

Total: 338 columns, 26,304 rows.
