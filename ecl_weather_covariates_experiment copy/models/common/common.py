"""
Shared utilities for ECL + Lisbon weather exogenous model scripts.

These helpers are intentionally local to ecl_weather_covariates_experiment.
They do not read from or write to the original dissertation model folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = EXPERIMENT_ROOT / "outputs" / "electricity_lisbon_weather.csv"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
RAW_RESULTS_DIR = OUTPUT_ROOT / "raw_metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures" / "model_diagnostics"

HORIZONS = [24, 48, 168]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
N_TOP = 10
N_RANDOM = 10
RANDOM_SEED = 42

DATETIME_COLUMNS = ["benchmark_datetime", "physical_datetime"]
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "heating_degree",
    "cooling_degree",
]
CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]
EXOG_COLUMNS = WEATHER_COLUMNS + CALENDAR_COLUMNS
EXPECTED_LOAD_COLUMNS = 321
EXPECTED_TOTAL_COLUMNS = EXPECTED_LOAD_COLUMNS + len(DATETIME_COLUMNS) + len(EXOG_COLUMNS)


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    load_columns: list[str]
    exog_columns: list[str]


def ensure_output_dirs() -> None:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_exog_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Weather-augmented dataset not found: {path}. "
            "Run build_ecl_lisbon_weather_features.py first."
        )

    df = pd.read_csv(path)
    if len(df.columns) != EXPECTED_TOTAL_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_COLUMNS} total columns "
            f"({EXPECTED_LOAD_COLUMNS} load + {len(DATETIME_COLUMNS)} datetime + "
            f"{len(EXOG_COLUMNS)} exogenous), got {len(df.columns)}."
        )

    missing = [c for c in DATETIME_COLUMNS + EXOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["benchmark_datetime"] = pd.to_datetime(df["benchmark_datetime"], errors="raise")
    df["physical_datetime"] = pd.to_datetime(df["physical_datetime"], errors="raise")
    df = df.sort_values("physical_datetime").reset_index(drop=True)

    if df["physical_datetime"].duplicated().any():
        raise ValueError("Duplicate physical_datetime values detected.")
    if df["benchmark_datetime"].duplicated().any():
        raise ValueError("Duplicate benchmark_datetime values detected.")
    if not (df["physical_datetime"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("physical_datetime is not continuous hourly.")

    load_columns = [
        c for c in df.columns
        if c not in set(DATETIME_COLUMNS + EXOG_COLUMNS)
    ]
    if not load_columns:
        raise ValueError("No electricity load columns were detected.")
    if len(load_columns) != EXPECTED_LOAD_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_LOAD_COLUMNS} electricity load columns, got "
            f"{len(load_columns)}. Check DATETIME_COLUMNS and EXOG_COLUMNS so "
            "weather/calendar variables are not misclassified as clients."
        )

    df[load_columns] = df[load_columns].apply(pd.to_numeric, errors="coerce")
    df[EXOG_COLUMNS] = df[EXOG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df[EXOG_COLUMNS].isna().any().any():
        missing_counts = df[EXOG_COLUMNS].isna().sum()
        raise ValueError(
            "Missing values detected in exogenous covariates:\n"
            f"{missing_counts[missing_counts > 0].to_string()}"
        )

    return df, load_columns, EXOG_COLUMNS.copy()


def chronological_split(df: pd.DataFrame, load_columns: list[str], exog_columns: list[str]) -> SplitData:
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return SplitData(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
        load_columns=load_columns,
        exog_columns=exog_columns,
    )


def load_and_split(path: Path = DATA_PATH) -> SplitData:
    df, load_columns, exog_columns = load_exog_dataset(path)
    return chronological_split(df, load_columns, exog_columns)


def select_clients(train_df: pd.DataFrame, load_columns: list[str]) -> list[str]:
    means = train_df[load_columns].mean(axis=0).sort_values(ascending=False)
    top_clients = [str(c) for c in means.head(N_TOP).index]
    remaining = [str(c) for c in load_columns if str(c) not in top_clients]
    rng = np.random.default_rng(RANDOM_SEED)
    random_clients = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    return top_clients + random_clients


def series_values(df: pd.DataFrame, client_id: str) -> np.ndarray:
    return pd.Series(df[client_id].values).ffill().bfill().values.astype(float)


def exog_values(df: pd.DataFrame, exog_columns: list[str]) -> np.ndarray:
    return df[exog_columns].values.astype(float)


def physical_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df["physical_datetime"])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def validate_forecast_origin(split: SplitData) -> dict[str, pd.Timestamp]:
    """Assert that fixed-origin forecasts start at the test boundary.

    The context for fixed-origin forecasting must end at the final validation
    row, not at the final training row. Otherwise the model forecasts the
    validation period and the metrics compare it against test actuals.
    """
    train_end = split.train["physical_datetime"].iloc[-1]
    val_end = split.val["physical_datetime"].iloc[-1]
    test_start = split.test["physical_datetime"].iloc[0]

    expected_test_start = val_end + pd.Timedelta(hours=1)
    if expected_test_start != test_start:
        raise ValueError(
            "Forecast origin mismatch: train+val context does not end one hour "
            f"before test_start. val_end={val_end}, test_start={test_start}"
        )

    if train_end + pd.Timedelta(hours=1) == test_start:
        raise ValueError(
            "Forecast origin check is ambiguous: train end is adjacent to test start, "
            "so the validation split may have been skipped unexpectedly."
        )

    return {
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
    }


def dry_run_report(split: SplitData, script_name: str) -> None:
    selected = select_clients(split.train, split.load_columns)
    origin = validate_forecast_origin(split)
    print(f"{script_name} dry run")
    print(f"Dataset: {DATA_PATH}")
    print(f"Train rows: {len(split.train)}")
    print(f"Validation rows: {len(split.val)}")
    print(f"Test rows: {len(split.test)}")
    print(f"Total columns: {len(split.train.columns)}")
    print(f"Load columns: {len(split.load_columns)}")
    print(f"Exogenous columns: {len(split.exog_columns)}")
    print(f"Exogenous column names: {split.exog_columns}")
    print(f"Selected clients: {selected}")
    print(
        "Physical datetime range: "
        f"{split.train['physical_datetime'].min()} to {split.test['physical_datetime'].max()}"
    )
    print(
        "Benchmark datetime range: "
        f"{split.train['benchmark_datetime'].min()} to {split.test['benchmark_datetime'].max()}"
    )
    print(
        "Forecast origin check: OK "
        f"(train_end={origin['train_end']}, val_end={origin['val_end']}, "
        f"test_start={origin['test_start']})"
    )
