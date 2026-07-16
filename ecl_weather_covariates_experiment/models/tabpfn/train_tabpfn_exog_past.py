"""
TabPFN-TS past-exogenous baseline for the independent ECL weather experiment.

Uses TabPFNTSPipeline (time-series native, LOCAL mode) with past Lisbon
weather and calendar covariates as additional channels in the context window.

Fair setting:
- Past load values and past weather/calendar features are provided as context.
- Future weather (temperature, humidity, etc.) is unknown: last observed value
  is persisted for the forecast horizon.
- Future calendar features (hour, day_of_week, etc.) are known deterministically
  and provided as-is.
- No future observed weather is used.

This approach produces a full H-step trajectory prediction, making RMSE
directly comparable to ARIMA, XGBoost, LSTM, and iTransformer results.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
from pathlib import Path

# macOS SSL fix — patch before any network call so HuggingFace downloads work.
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.common import (  # noqa: E402
    CALENDAR_COLUMNS,
    HORIZONS,
    RAW_RESULTS_DIR,
    WEATHER_COLUMNS,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    load_and_split,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "tabpfn_exog_past_weather_results.csv"
CONTEXT_ROWS = 1000
PAST_EXOG_COLUMNS = WEATHER_COLUMNS + CALENDAR_COLUMNS


def make_pipeline():
    try:
        from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
    except ImportError as exc:
        raise RuntimeError(
            "tabpfn_time_series not found. Install it before running this script."
        ) from exc
    return TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)


def _make_tsdf(df: pd.DataFrame) -> "TimeSeriesDataFrame":
    from tabpfn_time_series import TimeSeriesDataFrame
    return TimeSeriesDataFrame(df, id_column="item_id", timestamp_column="timestamp")


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> list[dict]:
    print(f"\nEvaluating client {client_idx}/{total_clients} (id={client_id})...")

    test_vals = series_values(split.test, client_id)
    last_val_row = split.val.iloc[-1]

    # Context: last CONTEXT_ROWS rows of train+val with load + all exog
    ctx_data = pd.concat([split.train, split.val]).tail(CONTEXT_ROWS).reset_index(drop=True)
    ctx_df = pd.DataFrame({
        "item_id": client_id,
        "timestamp": ctx_data["date"].values,
        "target": series_values(ctx_data, client_id),
    })
    for col in PAST_EXOG_COLUMNS:
        ctx_df[col] = ctx_data[col].values
    ctx_tsdf = _make_tsdf(ctx_df)

    pipe = make_pipeline()

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue

        # Future: calendar features are deterministically known; weather persisted.
        fut_data = split.test.iloc[:horizon].copy()
        fut_df = pd.DataFrame({
            "item_id": client_id,
            "timestamp": fut_data["date"].values,
            "target": np.nan,
        })
        for col in CALENDAR_COLUMNS:
            fut_df[col] = fut_data[col].values
        for col in WEATHER_COLUMNS:
            fut_df[col] = float(last_val_row[col])
        fut_tsdf = _make_tsdf(fut_df)

        t0 = time.time()
        result = pipe.predict(context_tsdf=ctx_tsdf, future_tsdf=fut_tsdf)
        elapsed = time.time() - t0

        forecast = result["target"].values[:horizon]
        actual = test_vals[:horizon]
        metrics = compute_metrics(actual, forecast)

        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": 0,
            "inference_time_sec": round(elapsed, 2),
            "model": "tabpfn_ts_exog_past_weather",
            "exog_setting": "past_exog_context_only",
            "context_rows": len(ctx_data),
            "uses_future_exog": False,
            "future_exog_strategy": "calendar_known_weather_persisted",
            "prediction_strategy": "tabpfn_ts_trajectory",
        })
        print(f"  Horizon {horizon:3d}h -> RMSE={metrics['RMSE']:.3f}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "TabPFN-TS exogenous past-weather")
        print("Setting: TabPFNTSPipeline LOCAL mode; past weather context + known future calendar.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients)))

    pd.DataFrame(all_records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
