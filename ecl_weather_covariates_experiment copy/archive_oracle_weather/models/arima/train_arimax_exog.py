"""
ARIMAX/SARIMAX exogenous baseline for the independent ECL weather experiment.

Reference pattern: dissertation/models/arima/train_arima.py

This script keeps the same selected-client and horizon design, but replaces
load-only ARIMA with SARIMAX using Lisbon weather/calendar covariates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.common import (  # noqa: E402
    HORIZONS,
    RAW_RESULTS_DIR,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    exog_values,
    load_and_split,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "arimax_exog_oracle_weather_results.csv"
TRAIN_WINDOW = 2000
ORDER = (2, 1, 2)


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> list[dict]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)
    train_val_vals = np.concatenate([train_vals, val_vals])

    train_exog = exog_values(split.train, split.exog_columns)
    val_exog = exog_values(split.val, split.exog_columns)
    test_exog = exog_values(split.test, split.exog_columns)
    train_val_exog = np.vstack([train_exog, val_exog])

    y_window = train_val_vals[-TRAIN_WINDOW:]
    x_window = train_val_exog[-TRAIN_WINDOW:]

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_window.reshape(-1, 1)).ravel()
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_window)
    test_x_scaled = x_scaler.transform(test_exog)

    t0 = time.time()
    model = SARIMAX(
        y_scaled,
        exog=x_scaled,
        order=ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    train_time = time.time() - t0

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue
        t1 = time.time()
        forecast_scaled = model.forecast(steps=horizon, exog=test_x_scaled[:horizon])
        inference_time = time.time() - t1
        forecast = y_scaler.inverse_transform(np.asarray(forecast_scaled).reshape(-1, 1)).ravel()
        actual = test_vals[:horizon]
        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "model": "arimax_exog_oracle_weather",
            "exog_setting": "future_observed_weather_oracle",
            "order": str(ORDER),
            "train_window": TRAIN_WINDOW,
            "uses_future_exog": True,
        })
        print(f"  Horizon {horizon:3d}h -> RMSE={metrics['RMSE']:.3f}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate data path/splits without training.")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "ARIMAX exogenous oracle-weather")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients)))

    pd.DataFrame(all_records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
