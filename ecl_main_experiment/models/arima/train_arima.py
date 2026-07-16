"""
================================================================================
DISSERTATION NOTE: ARIMA Baseline Model — train_arima.py
================================================================================
Research topic: "When Do Transformer-Based Models Outperform Traditional
Approaches? A Comparative Study of Multivariate Time Series Forecasting"

Model: ARIMA (non-seasonal) via pmdarima.auto_arima, fitted per-client on the
       last 2000 training rows (~83 days). Seasonal ARIMA (m=24) was found to
       be computationally intractable (30-60 min/client); non-seasonal ARIMA is
       the standard comparative baseline in the time-series forecasting literature.
Dataset: ECL (Electricity Consuming Load), 321 clients, hourly 2012–2014

Sampling strategy (computational feasibility):
  - Running SARIMA on all 321 clients is computationally intractable.
  - We select 20 representative clients:
      * Top 10 clients by mean consumption (high-load segment)
      * 10 randomly selected clients (seed=42, general population sample)
  - This balances representativeness with runtime constraints, and mirrors
    the sampling strategy used for the other baseline models in this study.

Train/Val/Test split: 70% / 10% / 20% (strict chronological order)
Forecast horizons evaluated: 24h, 48h, 168h (1 week)
Normalisation: StandardScaler per client (fit on train only, inverse-transformed
               before metric computation to keep units interpretable as kWh).

Outputs:
  - dissertation/results/raw_metrics/arima_results.csv
  - dissertation/results/figures/model_diagnostics/arima_forecast_sample.png
  - dissertation/results/figures/model_diagnostics/arima_error_by_horizon.png
================================================================================
"""

import time
import warnings
import logging
import os
import sys

# Force line-buffered stdout so progress prints appear immediately when running
# in background or piped to a file (Python defaults to block-buffering otherwise).
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Paths — all relative to the repo root for portability.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_PATH    = os.path.join(ROOT, "data", "ECL", "electricity.csv")
RESULTS_DIR  = os.path.join(ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw_metrics")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures", "model_diagnostics")
RESULTS_CSV  = os.path.join(RAW_RESULTS_DIR, "arima_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Hyperparameters — centralised for easy reproduction.
# ---------------------------------------------------------------------------
SEASONAL_PERIOD = 24          # daily seasonality in hourly data
HORIZONS        = [24, 48, 168]
N_TOP           = 10          # top clients by mean consumption
N_RANDOM        = 10          # randomly sampled clients
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.10        # TEST_RATIO = 0.20 (implicit)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Data loading and chronological splitting.
# We never shuffle; future timestamps must not leak into training.
# ---------------------------------------------------------------------------
def load_and_split(path: str):
    log.info("Loading data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()  # guarantee chronological order

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    log.info(
        "Split sizes — Train: %d | Val: %d | Test: %d (total rows: %d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Client sampling strategy.
# Top-10 by mean consumption captures high-load industrial clients (harder to
# forecast). Random-10 gives a representative cross-section of the population.
# ---------------------------------------------------------------------------
def select_clients(train: pd.DataFrame) -> list:
    means = train.mean(axis=0).sort_values(ascending=False)
    top10 = list(means.head(N_TOP).index)

    rng = np.random.default_rng(RANDOM_SEED)
    remaining = [c for c in df_all.columns if c not in top10]
    random10  = list(rng.choice(remaining, size=N_RANDOM, replace=False))
    # convert numpy str to plain str for consistency
    random10  = [str(c) for c in random10]

    selected = top10 + random10
    log.info("Selected clients: top-10=%s | random-10=%s", top10, random10)
    return selected


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Metric helpers — inverse-transformed predictions so MSE/
# MAE/RMSE are in original kWh units, matching other models in the comparison.
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Per-client SARIMA training and multi-horizon evaluation.
# auto_arima selects optimal (p,d,q)(P,D,Q)[m] per client via stepwise search.
# We fit once on the training set, then use rolling/direct multi-step forecast
# for each horizon — the same forecast origin is used for all horizons so
# results are directly comparable across horizons.
# ---------------------------------------------------------------------------
def train_and_evaluate(client_id: str, train_series: np.ndarray,
                        val_series: np.ndarray, test_series: np.ndarray,
                        client_idx: int, total_clients: int) -> list[dict]:
    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    # DISSERTATION NOTE: Bug fix — use train+val as the fitting context.
    # Previously only train_series was used, so the ARIMA predicted from the end
    # of the training split, which is 2,631 rows (the val set) BEFORE the test
    # set starts. Comparing those predictions to test actuals gave astronomically
    # wrong errors. Combining train+val means predictions start exactly at the
    # test boundary, making the comparison valid.
    train_val_series = np.concatenate([train_series, val_series])

    # DISSERTATION NOTE: Fit the scaler on the same 2000-row window that ARIMA
    # will be trained on. Previously the scaler was fit on all 18k training rows
    # while ARIMA saw only the last 2000 — mismatched normalisation statistics
    # caused the inverse-transform to produce values inconsistent with the model's
    # internal scale. Fitting both on the same window ensures consistency.
    train_window = train_val_series[-2000:]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_window.reshape(-1, 1)).ravel()

    # --- Training ---
    t0_train = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                train_scaled,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
                max_p=5, max_q=5,
                max_d=2,
                trace=False,
            )
    except Exception as exc:
        log.error("auto_arima failed for client %s: %s", client_id, exc)
        return []

    train_time = time.time() - t0_train
    arima_order    = str(model.order)
    seasonal_order = str(model.seasonal_order)
    log.info(
        "Client %s fitted | order=%s seasonal=%s | train_time=%.1fs",
        client_id, arima_order, seasonal_order, train_time,
    )

    records = []
    for horizon in HORIZONS:
        # --- Inference ---
        n_test = min(horizon, len(test_series))
        if n_test < horizon:
            log.warning(
                "Client %s: test set too short for horizon %d (only %d steps). Skipping.",
                client_id, horizon, n_test,
            )
            continue

        t0_inf = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast_scaled, _ = model.predict(
                    n_periods=horizon, return_conf_int=True
                )
        except Exception as exc:
            log.error(
                "Inference failed for client %s horizon %d: %s",
                client_id, horizon, exc,
            )
            continue
        inference_time = time.time() - t0_inf

        # Inverse transform to original kWh units
        forecast = scaler.inverse_transform(
            forecast_scaled.reshape(-1, 1)
        ).ravel()
        actual = test_series[:horizon]

        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id":        client_id,
            "horizon":          horizon,
            "MSE":              round(metrics["MSE"],  4),
            "MAE":              round(metrics["MAE"],  4),
            "RMSE":             round(metrics["RMSE"], 4),
            "train_time_sec":   round(train_time,      2),
            "inference_time_sec": round(inference_time, 4),
            "arima_order":      arima_order,
            "seasonal_order":   seasonal_order,
            # stash for plotting
            "_actual":   actual,
            "_forecast": forecast,
            "_client_id": client_id,
        })
        print(
            f"  Horizon {horizon:3d}h → MSE={metrics['MSE']:.3f} | "
            f"MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | "
            f"Inference={inference_time:.3f}s"
        )

    return records


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Figures — two required outputs.
# Figure 1: actual vs predicted for one representative client at horizon 24h.
#           Gives a qualitative sense of forecast quality.
# Figure 2: bar chart of MSE/MAE averaged over 20 clients per horizon.
#           Directly feeds into the comparative results table in Chapter 4.
# ---------------------------------------------------------------------------
def save_sample_forecast_plot(actual, forecast, client_id, horizon,
                               out_path: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,   label="Actual",    linewidth=1.5)
    ax.plot(forecast, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(
        f"ARIMA Forecast — Client {client_id}, Horizon {horizon}h"
    )
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Electricity consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sample forecast plot → %s", out_path)


def save_error_by_horizon_plot(results_df: pd.DataFrame, out_path: str):
    summary = (
        results_df.groupby("horizon")[["RMSE", "MAE"]]
        .mean()
        .reindex(HORIZONS)
    )

    x = np.arange(len(HORIZONS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, summary["RMSE"], width, label="RMSE")
    bars2 = ax.bar(x + width / 2, summary["MAE"],  width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Error (kWh)")
    ax.set_title("ARIMA — Average RMSE and MAE by Forecast Horizon")
    ax.legend()

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved error-by-horizon plot → %s", out_path)


# ===========================================================================
# DISSERTATION NOTE: Main execution — orchestrates the full pipeline.
# ===========================================================================
if __name__ == "__main__":
    # --- Load data ---
    df_all = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df_all = df_all.sort_index()

    n = len(df_all)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df_all.iloc[:train_end]
    val_df   = df_all.iloc[train_end:val_end]
    test_df  = df_all.iloc[val_end:]

    log.info(
        "Split: Train=%d | Val=%d | Test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # --- Select 20 clients ---
    means   = train_df.mean(axis=0).sort_values(ascending=False)
    top10   = list(means.head(N_TOP).index)
    rng     = np.random.default_rng(RANDOM_SEED)
    remaining = [c for c in df_all.columns if c not in top10]
    random10  = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    selected_clients = top10 + random10

    log.info("Top-10 clients (by mean consumption): %s", top10)
    log.info("Random-10 clients (seed=%d): %s", RANDOM_SEED, random10)
    log.info("Total clients to train: %d", len(selected_clients))

    # --- Train and evaluate ---
    all_records   = []
    sample_record = None   # for plot: first successful 24h forecast

    for idx, cid in enumerate(selected_clients, start=1):
        # Check for excessive missing values (>50% NaN)
        client_series = df_all[cid].values
        nan_ratio = np.isnan(client_series).mean()
        if nan_ratio > 0.50:
            log.warning(
                "Client %s has %.1f%% missing values — skipping.", cid, nan_ratio * 100
            )
            continue

        # Forward-fill any remaining NaNs (minor gaps)
        train_vals = pd.Series(train_df[cid].values).ffill().bfill().values.astype(float)
        val_vals   = pd.Series(val_df[cid].values).ffill().bfill().values.astype(float)
        test_vals  = pd.Series(test_df[cid].values).ffill().bfill().values.astype(float)

        records = train_and_evaluate(
            client_id=cid,
            train_series=train_vals,
            val_series=val_vals,
            test_series=test_vals,
            client_idx=idx,
            total_clients=len(selected_clients),
        )

        for r in records:
            # Capture sample for Figure 1 (first client, horizon 24h)
            if sample_record is None and r["horizon"] == 24:
                sample_record = r.copy()

            # Remove plot data before storing to CSV
            r.pop("_actual",    None)
            r.pop("_forecast",  None)
            r.pop("_client_id", None)
            all_records.append(r)

    # --- Save CSV ---
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # --- Console summary table ---
    print("\n" + "=" * 70)
    print("ARIMA RESULTS SUMMARY — Average across clients")
    print("=" * 70)
    for h in HORIZONS:
        subset = results_df[results_df["horizon"] == h]
        if subset.empty:
            print(f"  Horizon {h:3d}h → no data")
            continue
        print(
            f"  Horizon {h:3d}h → "
            f"MSE: {subset['MSE'].mean():8.3f} | "
            f"MAE: {subset['MAE'].mean():8.3f} | "
            f"RMSE: {subset['RMSE'].mean():8.3f} | "
            f"Avg Train Time: {subset['train_time_sec'].mean():.1f}s"
        )
    print("=" * 70)

    # --- Figure 1: sample forecast ---
    if sample_record:
        save_sample_forecast_plot(
            actual    = sample_record["_actual"]   if "_actual"   in sample_record else None,
            forecast  = sample_record["_forecast"] if "_forecast" in sample_record else None,
            client_id = sample_record.get("_client_id", sample_record["client_id"]),
            horizon   = 24,
            out_path  = os.path.join(FIGURES_DIR, "arima_forecast_sample.png"),
        )

    # --- Figure 2: error by horizon ---
    if not results_df.empty:
        save_error_by_horizon_plot(
            results_df,
            out_path=os.path.join(FIGURES_DIR, "arima_error_by_horizon.png"),
        )

    print("\nDone. All outputs saved to:", RESULTS_DIR)
