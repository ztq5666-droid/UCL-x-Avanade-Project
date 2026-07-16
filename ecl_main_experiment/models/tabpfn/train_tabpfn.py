"""
================================================================================
DISSERTATION NOTE: TabPFN-TS Zero-Shot Baseline — train_tabpfn.py
================================================================================
Research topic: "When Do Transformer-Based Models Outperform Traditional
Approaches? A Comparative Study of Multivariate Time Series Forecasting"

Model: TabPFN-TS via tabpfn_time_series.pipeline.TabPFNTSPipeline
       (TabPFNMode.LOCAL — runs on CPU without a cloud account)
Dataset: ECL (Electricity Consuming Load), 321 clients, hourly 2012–2014

Zero-shot baseline: TabPFN-TS requires NO gradient training or fine-tuning.
  train_time_sec is recorded as 0 for all clients/horizons.

Sampling strategy (mirrors ARIMA baseline for direct comparability):
  - Top 10 clients by mean consumption on the training set (high-load segment)
  - 10 randomly selected remaining clients (seed=42, general population sample)

Context window: last 96 hours of the training set per client.
  Full training history (12k+ rows) would exceed TabPFN context limits and
  dramatically increase inference time with diminishing accuracy returns.
  96 h (4 days) captures recent daily cycles — sufficient for 24/48/168 h horizons.

Train/Val/Test split: 70% / 10% / 20% (strict chronological order, no shuffle)
Forecast horizons: 24h, 48h, 168h (fixed-origin from start of test set)
Normalisation: StandardScaler per client, fit on train only, inverse-transformed
               before metric computation to keep units interpretable as kWh.

Outputs:
  - dissertation/results/raw_metrics/tabpfn_results.csv
  - dissertation/results/figures/model_diagnostics/tabpfn_forecast_sample.png
  - dissertation/results/figures/model_diagnostics/tabpfn_error_by_horizon.png
================================================================================
"""

import time
import warnings
import logging
import os
import sys
import ssl

sys.stdout.reconfigure(line_buffering=True)

# macOS ships without the CA bundle that Python needs for HTTPS.
# Patch ssl globally before any network call so tabpfn's HuggingFace
# download succeeds without requiring the user to run Install Certificates.
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass  # certifi not installed — SSL may still work on some systems

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tabpfn_time_series.pipeline import TabPFNTSPipeline
from tabpfn_time_series import TabPFNMode

# ---------------------------------------------------------------------------
# Paths — relative to repo root for portability.
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_PATH   = os.path.join(ROOT, "data", "ECL", "electricity.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw_metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "model_diagnostics")
RESULTS_CSV = os.path.join(RAW_RESULTS_DIR, "tabpfn_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — centralised for reproducibility.
# ---------------------------------------------------------------------------
HORIZONS      = [24, 48, 168]
N_TOP         = 10
N_RANDOM      = 10
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10   # TEST_RATIO = 0.20 implicit

# DISSERTATION NOTE: 96-hour context window.
# TabPFN-TS is a zero-shot in-context learner — it treats the context as its
# "training set" at inference time. A 96-h window (4 complete daily cycles)
# gives the model enough seasonal signal for all three horizons while keeping
# per-client inference time under ~30 s on CPU.
CONTEXT_LENGTH = 96

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers — always computed on inverse-transformed (kWh) values.
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": rmse}


# ---------------------------------------------------------------------------
# Per-client inference and multi-horizon evaluation.
# ---------------------------------------------------------------------------
def evaluate_client(
    pipeline: TabPFNTSPipeline,
    client_id: str,
    train_series: np.ndarray,
    test_series: np.ndarray,
    client_idx: int,
    total_clients: int,
) -> list[dict]:
    print(f"\nEvaluating client {client_idx}/{total_clients} (id={client_id})...")

    # DISSERTATION NOTE: Normalise per client; scaler fit on train+val so its
    # statistics match the context window. Inverse-transform before metrics so
    # MSE/MAE/RMSE are in kWh — directly comparable to other baselines.
    scaler = StandardScaler()
    scaler.fit(train_series.reshape(-1, 1))

    # Context = last CONTEXT_LENGTH rows of train+val (= train_series here).
    # This ensures predictions start exactly at the test set boundary — no gap.
    context_vals = train_series[-CONTEXT_LENGTH:]
    context_scaled = scaler.transform(context_vals.reshape(-1, 1)).ravel()

    # Build a dummy hourly DatetimeIndex for the context (TabPFN-TS needs timestamps).
    # Absolute dates don't affect zero-shot accuracy; only relative spacing matters.
    context_timestamps = pd.date_range(
        start="2013-01-01", periods=CONTEXT_LENGTH, freq="h"
    )

    records = []
    for horizon in HORIZONS:
        n_test = len(test_series)
        if n_test < horizon:
            log.warning(
                "Client %s: test set too short for horizon %d (%d steps). Skipping.",
                client_id, horizon, n_test,
            )
            continue

        # Build context DataFrame expected by TabPFNTSPipeline.predict_df:
        #   columns: timestamp, target  (item_id optional for single series)
        context_df = pd.DataFrame({
            "timestamp": context_timestamps,
            "target":    context_scaled,
        })

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_df = pipeline.predict_df(
                    context_df=context_df,
                    prediction_length=horizon,
                    quantiles=[0.5],   # point forecast only — speeds up inference
                )
        except Exception as exc:
            log.error(
                "TabPFN-TS inference failed for client %s horizon %d: %s",
                client_id, horizon, exc,
            )
            continue
        inference_time = time.time() - t0

        # Extract point forecast (median at q=0.5; fallback to 'target' column).
        if "0.5" in pred_df.columns:
            forecast_scaled = pred_df["0.5"].values[:horizon]
        else:
            forecast_scaled = pred_df["target"].values[:horizon]

        # Inverse-transform predictions and compute metrics in kWh.
        forecast = scaler.inverse_transform(
            forecast_scaled.reshape(-1, 1)
        ).ravel()
        actual = test_series[:horizon]

        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id":          client_id,
            "horizon":            horizon,
            "MSE":                round(metrics["MSE"],  4),
            "MAE":                round(metrics["MAE"],  4),
            "RMSE":               round(metrics["RMSE"], 4),
            # DISSERTATION NOTE: train_time_sec = 0 because TabPFN-TS is
            # zero-shot — there is no gradient-based training step. All compute
            # is in-context inference, recorded separately as inference_time_sec.
            "train_time_sec":     0,
            "inference_time_sec": round(inference_time, 4),
            # stash arrays for Figure 1 (removed before CSV write)
            "_actual":    actual,
            "_forecast":  forecast,
            "_client_id": client_id,
        })
        print(
            f"  Horizon {horizon:3d}h → MSE={metrics['MSE']:.3f} | "
            f"MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | "
            f"Inference={inference_time:.3f}s"
        )

    return records


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_sample_forecast_plot(
    actual, forecast, client_id, horizon, out_path: str
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,   label="Actual",    linewidth=1.5)
    ax.plot(forecast, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(f"TabPFN-TS Forecast — Client {client_id}, Horizon {horizon}h")
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Electricity consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sample forecast plot → %s", out_path)


def save_error_by_horizon_plot(results_df: pd.DataFrame, out_path: str):
    # DISSERTATION NOTE: Plot RMSE and MAE (both in kWh) — not MSE.
    # MSE is in kWh² while RMSE and MAE are in kWh; mixing them on one axis
    # produces a misleading scale comparison. Using RMSE + MAE keeps units
    # consistent and makes the bar chart directly interpretable.
    summary = (
        results_df.groupby("horizon")[["RMSE", "MAE"]]
        .mean()
        .reindex(HORIZONS)
    )

    x     = np.arange(len(HORIZONS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, summary["RMSE"], width, label="RMSE")
    bars2 = ax.bar(x + width / 2, summary["MAE"],  width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Error (kWh)")
    ax.set_title("TabPFN-TS — Average RMSE and MAE by Forecast Horizon")
    ax.legend()

    for bar in list(bars1) + list(bars2):
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
# Main
# ===========================================================================
if __name__ == "__main__":
    t_total_start = time.time()

    # --- Load data ---
    log.info("Loading data from %s", DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)

    # Robustly rename the timestamp column to 'date' regardless of original name.
    first_col = df_all.columns[0]
    if first_col != "date":
        df_all.rename(columns={first_col: "date"}, inplace=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.set_index("date").sort_index()

    n         = len(df_all)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df_all.iloc[:train_end]
    val_df   = df_all.iloc[train_end:val_end]
    test_df  = df_all.iloc[val_end:]

    log.info(
        "Split: Train=%d | Val=%d | Test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # --- DISSERTATION NOTE: Client sampling — identical to ARIMA baseline.
    # Top-10 by mean consumption: captures high-load industrial clients that
    # are harder to forecast and more policy-relevant.
    # Random-10 (seed=42): representative cross-section of the full population.
    # Using the same 20 clients across all models ensures results are comparable.
    means     = train_df.mean(axis=0).sort_values(ascending=False)
    top10     = list(means.head(N_TOP).index)
    rng       = np.random.default_rng(RANDOM_SEED)
    remaining = [c for c in df_all.columns if c not in top10]
    random10  = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    selected_clients = top10 + random10

    log.info("Top-10 clients (by mean consumption): %s", top10)
    log.info("Random-10 clients (seed=%d): %s", RANDOM_SEED, random10)
    log.info("Total clients to evaluate: %d", len(selected_clients))

    # --- Initialise TabPFN-TS pipeline (LOCAL mode — no API account needed) ---
    log.info("Initialising TabPFNTSPipeline (LOCAL mode, CPU)...")
    pipeline = TabPFNTSPipeline(
        max_context_length=CONTEXT_LENGTH,
        tabpfn_mode=TabPFNMode.LOCAL,
    )
    log.info("Pipeline ready.")

    # --- Evaluate all selected clients ---
    all_records   = []
    sample_record = None

    for idx, cid in enumerate(selected_clients, start=1):
        # Skip clients with excessive missing values
        nan_ratio = np.isnan(df_all[cid].values).mean()
        if nan_ratio > 0.50:
            log.warning(
                "Client %s has %.1f%% missing values — skipping.", cid, nan_ratio * 100
            )
            continue

        # Forward-fill minor gaps; bfill handles leading NaNs.
        # ALIGNMENT FIX: context must end immediately before the test set.
        # train+val is used so that the last CONTEXT_LENGTH rows of (train+val)
        # are contiguous with the first row of test — no gap, no window mismatch.
        train_vals = (
            pd.Series(train_df[cid].values).ffill().bfill().values.astype(float)
        )
        val_vals = (
            pd.Series(val_df[cid].values).ffill().bfill().values.astype(float)
        )
        context_source = np.concatenate([train_vals, val_vals])
        test_vals = (
            pd.Series(test_df[cid].values).ffill().bfill().values.astype(float)
        )

        records = evaluate_client(
            pipeline=pipeline,
            client_id=cid,
            train_series=context_source,   # train+val → context ends at test boundary
            test_series=test_vals,
            client_idx=idx,
            total_clients=len(selected_clients),
        )

        for r in records:
            if sample_record is None and r["horizon"] == 24:
                sample_record = r.copy()

            r.pop("_actual",    None)
            r.pop("_forecast",  None)
            r.pop("_client_id", None)
            all_records.append(r)

    total_wall_time = time.time() - t_total_start

    # --- Save CSV ---
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # --- Console summary ---
    print("\n" + "=" * 75)
    print("TabPFN-TS RESULTS SUMMARY — Average across clients")
    print("=" * 75)
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
            f"Inference: {subset['inference_time_sec'].mean():.1f}s"
        )
    print(f"  Total wall-clock time: {total_wall_time:.1f}s")
    print("=" * 75)

    # --- Figure 1: sample forecast ---
    if sample_record and sample_record.get("_actual") is not None:
        save_sample_forecast_plot(
            actual    = sample_record["_actual"],
            forecast  = sample_record["_forecast"],
            client_id = sample_record.get("_client_id", sample_record["client_id"]),
            horizon   = 24,
            out_path  = os.path.join(FIGURES_DIR, "tabpfn_forecast_sample.png"),
        )

    # --- Figure 2: RMSE + MAE by horizon ---
    if not results_df.empty:
        save_error_by_horizon_plot(
            results_df,
            out_path=os.path.join(FIGURES_DIR, "tabpfn_error_by_horizon.png"),
        )

    print("\nDone. All outputs saved to:", RESULTS_DIR)
