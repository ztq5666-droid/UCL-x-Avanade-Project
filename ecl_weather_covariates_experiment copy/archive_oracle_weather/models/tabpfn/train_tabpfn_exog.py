"""
TabPFN exogenous baseline for the independent ECL weather experiment.

Reference pattern: dissertation/models/tabpfn/train_tabpfn.py

TabPFN-TS does not expose a stable covariate interface in all environments, so
this exogenous version uses a tabular supervised framing: lag/rolling load
features plus Lisbon weather/calendar covariates. It remains an exploratory
foundation-model baseline rather than a traditional trained time-series model.
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
    physical_index,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "tabpfn_exog_oracle_weather_results.csv"
CONTEXT_ROWS = 2048
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]


def build_feature_frame(target_scaled, timestamps, exog_scaled, exog_columns):
    s = pd.Series(target_scaled, index=timestamps)
    df = pd.DataFrame(index=timestamps)
    for lag in LAG_FEATURES:
        df[f"lag_t{lag}"] = s.shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"roll_{window}"] = s.shift(1).rolling(window=window, min_periods=1).mean()
    for idx, col in enumerate(exog_columns):
        df[f"exog_{col}"] = exog_scaled[:, idx]
    df["target"] = target_scaled
    return df.dropna()


def build_next_row(history_scaled, future_exog_scaled, exog_columns, feature_cols):
    arr = np.asarray(history_scaled, dtype=float)
    row = {}
    for lag in LAG_FEATURES:
        row[f"lag_t{lag}"] = float(arr[-lag])
    for window in ROLLING_WINDOWS:
        row[f"roll_{window}"] = float(arr[-window:].mean())
    for idx, col in enumerate(exog_columns):
        row[f"exog_{col}"] = float(future_exog_scaled[idx])
    return pd.DataFrame([row])[feature_cols]


def make_tabpfn_regressor():
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor()
    except Exception as exc:
        raise RuntimeError(
            "Could not import tabpfn.TabPFNRegressor. Install/activate the "
            "same environment used for the TabPFN baseline before running."
        ) from exc


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> list[dict]:
    print(f"\nEvaluating client {client_idx}/{total_clients} (id={client_id})...")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)
    train_val_vals = np.concatenate([train_vals, val_vals])

    train_exog = exog_values(split.train, split.exog_columns)
    val_exog = exog_values(split.val, split.exog_columns)
    test_exog = exog_values(split.test, split.exog_columns)
    train_val_exog = np.vstack([train_exog, val_exog])

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(train_val_vals.reshape(-1, 1)).ravel()
    x_scaler = StandardScaler()
    exog_scaled = x_scaler.fit_transform(train_val_exog)
    test_exog_scaled = x_scaler.transform(test_exog)

    features = build_feature_frame(
        y_scaled,
        physical_index(pd.concat([split.train, split.val])),
        exog_scaled,
        split.exog_columns,
    ).tail(CONTEXT_ROWS)
    feature_cols = [c for c in features.columns if c != "target"]

    model = make_tabpfn_regressor()
    t0 = time.time()
    model.fit(features[feature_cols].values.astype(np.float32), features["target"].values.astype(np.float32))
    fit_time = time.time() - t0

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue
        history = list(y_scaled)
        preds_scaled = []
        t1 = time.time()
        for step in range(horizon):
            row = build_next_row(history, test_exog_scaled[step], split.exog_columns, feature_cols)
            pred = float(model.predict(row.values.astype(np.float32))[0])
            preds_scaled.append(pred)
            history.append(pred)
        inference_time = time.time() - t1

        forecast = y_scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        actual = test_vals[:horizon]
        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": round(fit_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "model": "tabpfn_exog_oracle_weather",
            "exog_setting": "future_observed_weather_oracle",
            "context_rows": len(features),
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
        dry_run_report(split, "TabPFN exogenous oracle-weather")
        print("Training mode uses tabpfn.TabPFNRegressor with lag + exog tabular features.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients)))

    pd.DataFrame(all_records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
