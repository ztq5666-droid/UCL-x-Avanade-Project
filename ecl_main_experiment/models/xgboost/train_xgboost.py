"""
================================================================================
DISSERTATION NOTE: XGBoost Baseline Model — train_xgboost.py
================================================================================
Research topic: "When Do Transformer-Based Models Outperform Traditional
Approaches? A Comparative Study of Multivariate Time Series Forecasting"

Model: XGBoost (XGBRegressor), one model per client — univariate and directly
       comparable with the ARIMA baseline.
Dataset: ECL (Electricity Consuming Load), 321 clients, hourly 2016–2019

Sampling strategy:
  - 20 representative clients selected identically to the ARIMA baseline:
      * Top 10 by mean consumption in the training split (high-load segment)
      * 10 randomly selected from remaining clients (seed=42)
  - This ensures a fair, like-for-like comparison across all models.

Train/Val/Test split: 70% / 10% / 20% (strict chronological order, no shuffle)
Forecast horizons: 24h, 48h, 168h
Forecasting method: recursive (one-step-ahead, appending predictions)
Normalisation: StandardScaler per client, fit on training set only; metrics
               reported in original kWh units after inverse-transform.

Feature engineering (supervised framing of the univariate series):
  Lag features: t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-168
  Rolling features: 24h mean, 168h mean
  Calendar features: hour_of_day, day_of_week, month, is_weekend

Outputs:
  - dissertation/results/raw_metrics/xgboost_results.csv
  - dissertation/results/figures/model_diagnostics/xgboost_forecast_sample.png
  - dissertation/results/figures/model_diagnostics/xgboost_error_by_horizon.png
================================================================================
"""

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: macOS OpenMP workaround.
# XGBoost requires libomp.dylib (OpenMP runtime). On systems without Homebrew,
# we resolve it from scikit-learn's bundled copy before importing xgboost.
# ---------------------------------------------------------------------------
import os
import sys

_LIBOMP_DIRS = [
    "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/sklearn/.dylibs",
    "/opt/anaconda3/lib",
]
_existing = os.environ.get("DYLD_LIBRARY_PATH", "")
for _d in _LIBOMP_DIRS:
    if os.path.isdir(_d) and _d not in _existing:
        os.environ["DYLD_LIBRARY_PATH"] = _d + (":" + _existing if _existing else "")
        break

# Force line-buffered stdout so progress prints appear immediately.
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Paths — all relative to the repository root.
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_PATH   = os.path.join(ROOT, "data", "ECL", "electricity.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw_metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "model_diagnostics")
RESULTS_CSV = os.path.join(RAW_RESULTS_DIR, "xgboost_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Hyperparameters — centralised for reproducibility.
# ---------------------------------------------------------------------------
HORIZONS     = [24, 48, 168]
N_TOP        = 10
N_RANDOM     = 10
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.10   # TEST_RATIO = 0.20 (implicit)

# Lag and rolling window sizes (hours)
LAG_FEATURES     = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS  = [24, 168]

# XGBoost base parameters
XGB_PARAMS = dict(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = "reg:squarederror",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)
EARLY_STOPPING_ROUNDS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===========================================================================
# Feature engineering
# ===========================================================================

def _build_features_from_series(series: np.ndarray,
                                  timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    DISSERTATION NOTE: Build the full supervised feature matrix from a 1-D
    time series.  The function is self-contained so it can be called both
    during training (on the full scaler-transformed history) and during
    recursive inference (on the growing predicted history).

    Returns a DataFrame where each row is one time step and columns are:
      lag_t{k}  — value k steps ago
      roll_{w}  — rolling mean over the last w hours
      hour_of_day, day_of_week, month, is_weekend
    """
    s = pd.Series(series, index=timestamps if len(timestamps) == len(series) else range(len(series)))

    df = pd.DataFrame(index=s.index)

    # Lag features
    for k in LAG_FEATURES:
        df[f"lag_t{k}"] = s.shift(k)

    # Rolling mean features (min_periods=1 avoids NaN at start)
    for w in ROLLING_WINDOWS:
        df[f"roll_{w}"] = s.shift(1).rolling(window=w, min_periods=1).mean()

    # Calendar features (only valid when index is DatetimeIndex)
    if isinstance(s.index, pd.DatetimeIndex):
        df["hour_of_day"] = s.index.hour
        df["day_of_week"] = s.index.dayofweek
        df["month"]       = s.index.month
        df["is_weekend"]  = (s.index.dayofweek >= 5).astype(int)
    else:
        df["hour_of_day"] = 0
        df["day_of_week"] = 0
        df["month"]       = 1
        df["is_weekend"]  = 0

    return df


def build_supervised_dataset(series: np.ndarray,
                              timestamps: pd.DatetimeIndex):
    """
    DISSERTATION NOTE: Convert a scaled 1-D series to (X, y) arrays for
    supervised learning.  Rows with any NaN feature (from initial lags) are
    dropped — the maximum lag is 168, so the first 168 rows are discarded.
    """
    df = _build_features_from_series(series, timestamps)
    df["target"] = series

    df = df.dropna()
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    return X, y, feature_cols


# ===========================================================================
# Recursive forecasting
# ===========================================================================

def recursive_forecast(model: xgb.XGBRegressor,
                       history_scaled: np.ndarray,
                       history_timestamps: pd.DatetimeIndex,
                       test_timestamps: pd.DatetimeIndex,
                       horizon: int,
                       feature_cols: list) -> np.ndarray:
    """
    DISSERTATION NOTE: Recursive (autoregressive) multi-step forecasting.

    Strategy:
      - Start from a fixed forecast origin: the beginning of the test set.
        This mirrors the ARIMA baseline and avoids data leakage.
      - At each step, build features from the current history, predict one
        step ahead, and append the prediction to the history.
      - True test values are NEVER used to construct recursive features;
        only the model's own prior predictions extend the history.
      - Calendar features for future steps are computed from the known
        future timestamps (day/hour of week) — this is legitimate since
        calendar position is not a future unknown.
    """
    # DISSERTATION NOTE: Combined history used to compute lags/rolling stats.
    # We carry the full training+validation history so early lags (t-168) are
    # always available even at step 1 of the test set.
    running_series     = list(history_scaled)
    running_timestamps = list(history_timestamps)

    predictions = []

    for step in range(horizon):
        # Build feature matrix from current running history
        arr = np.array(running_series, dtype=np.float64)
        ts  = pd.DatetimeIndex(running_timestamps)
        feature_df = _build_features_from_series(arr, ts)

        # Take the last row (most recent step) as the feature vector
        last_row = feature_df.iloc[[-1]][feature_cols]

        # Override calendar features with the actual future timestamp
        future_ts = test_timestamps[step]
        last_row = last_row.copy()
        last_row["hour_of_day"] = future_ts.hour
        last_row["day_of_week"] = future_ts.dayofweek
        last_row["month"]       = future_ts.month
        last_row["is_weekend"]  = int(future_ts.dayofweek >= 5)

        pred_scaled = model.predict(last_row.values.astype(np.float32))[0]

        predictions.append(float(pred_scaled))
        running_series.append(float(pred_scaled))
        running_timestamps.append(future_ts)

    return np.array(predictions, dtype=np.float64)


# ===========================================================================
# Model training
# ===========================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    DISSERTATION NOTE: Train XGBRegressor with early stopping.
    XGBoost's early-stopping API changed across versions:
      - v2.x+: early_stopping_rounds is a constructor parameter
      - v1.6–1.x: passed via fit() or via EarlyStopping callback
    We try each in order and fall back to full training if all fail.
    """
    # Attempt 1: XGBoost >= 2.0 — constructor parameter (preferred)
    try:
        model = xgb.XGBRegressor(
            **XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        log.debug("Early stopping via constructor parameter (XGBoost 2.x+).")
        return model
    except Exception as exc:
        log.debug("Constructor early stopping failed (%s); trying callback.", exc)

    # Attempt 2: modern callbacks-based early stopping (XGBoost >= 1.6)
    try:
        from xgboost.callback import EarlyStopping as XGBEarlyStopping
        model = xgb.XGBRegressor(**XGB_PARAMS)
        cb = XGBEarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS,
            metric_name="rmse",
            save_best=True,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=[cb],
        )
        log.debug("Early stopping via EarlyStopping callback.")
        return model
    except Exception as exc:
        log.debug("Callback early stopping failed (%s); trying fit() param.", exc)

    # Attempt 3: legacy early_stopping_rounds in fit() (XGBoost < 1.6)
    try:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        log.debug("Early stopping via fit() parameter (legacy API).")
        return model
    except Exception as exc:
        log.warning(
            "All early stopping methods unavailable (%s). "
            "Training for full %d estimators.", exc, XGB_PARAMS["n_estimators"]
        )

    # Fallback: full training without early stopping
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": rmse}


# ===========================================================================
# Per-client pipeline
# ===========================================================================

def train_and_evaluate(client_id: str,
                        train_vals: np.ndarray,
                        val_vals: np.ndarray,
                        test_vals: np.ndarray,
                        train_ts: pd.DatetimeIndex,
                        val_ts: pd.DatetimeIndex,
                        test_ts: pd.DatetimeIndex,
                        client_idx: int,
                        total_clients: int) -> list:
    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    # DISSERTATION NOTE: Scaler fitted on training split only — never on val or
    # test data.  Inverse-transform is applied to both predictions and actuals
    # before metric computation so all errors are in original kWh units.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
    val_scaled   = scaler.transform(val_vals.reshape(-1, 1)).ravel()

    # Build supervised datasets
    X_train, y_train, feature_cols = build_supervised_dataset(train_scaled, train_ts)
    X_val,   y_val,   _            = build_supervised_dataset(
        np.concatenate([train_scaled, val_scaled]),
        train_ts.append(val_ts),
    )
    # Align val set: discard rows that belong to the training portion
    n_train_rows = len(X_train)
    X_val_only = X_val[n_train_rows:] if len(X_val) > n_train_rows else X_val
    y_val_only = y_val[n_train_rows:] if len(y_val) > n_train_rows else y_val

    if len(X_train) == 0 or len(X_val_only) == 0:
        log.warning("Client %s: insufficient data after feature engineering. Skipping.", client_id)
        return []

    # --- Training ---
    t0_train = time.time()
    try:
        model = train_xgboost(X_train, y_train, X_val_only, y_val_only)
    except Exception as exc:
        log.error("Training failed for client %s: %s", client_id, exc)
        return []
    train_time = time.time() - t0_train

    log.info(
        "Client %s trained | n_estimators_used=%s | train_time=%.1fs",
        client_id,
        getattr(model, "best_iteration", XGB_PARAMS["n_estimators"]),
        train_time,
    )

    # DISSERTATION NOTE: Full context passed to the recursive forecaster.
    # Using train+val history means lag features at test-step 1 always have
    # 168 prior observations available — identical reasoning to the ARIMA fix.
    history_scaled     = np.concatenate([train_scaled, val_scaled])
    history_timestamps = train_ts.append(val_ts)

    records = []
    for horizon in HORIZONS:
        n_test = len(test_vals)
        if n_test < horizon:
            log.warning(
                "Client %s: test set too short for horizon %d (%d steps). Skipping.",
                client_id, horizon, n_test,
            )
            continue

        t0_inf = time.time()
        try:
            forecast_scaled = recursive_forecast(
                model=model,
                history_scaled=history_scaled,
                history_timestamps=history_timestamps,
                test_timestamps=test_ts[:horizon],
                horizon=horizon,
                feature_cols=feature_cols,
            )
        except Exception as exc:
            log.error(
                "Inference failed for client %s horizon %d: %s",
                client_id, horizon, exc,
            )
            continue
        inference_time = time.time() - t0_inf

        # Inverse-transform to original kWh units before computing metrics
        forecast = scaler.inverse_transform(
            forecast_scaled.reshape(-1, 1)
        ).ravel()
        actual = test_vals[:horizon]

        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id":          client_id,
            "horizon":            horizon,
            "MSE":                round(metrics["MSE"],  4),
            "MAE":                round(metrics["MAE"],  4),
            "RMSE":               round(metrics["RMSE"], 4),
            "train_time_sec":     round(train_time,      2),
            "inference_time_sec": round(inference_time,  4),
            # Private fields used for Figure 1 only — removed before CSV write
            "_actual":    actual,
            "_forecast":  forecast,
        })
        print(
            f"  Horizon {horizon:3d}h → "
            f"MSE={metrics['MSE']:.3f} | "
            f"MAE={metrics['MAE']:.3f} | "
            f"RMSE={metrics['RMSE']:.3f} | "
            f"Inference={inference_time:.3f}s"
        )

    return records


# ===========================================================================
# Figures
# ===========================================================================

def save_sample_forecast_plot(actual, forecast, client_id, horizon, out_path):
    """
    DISSERTATION NOTE: Figure 1 — qualitative forecast quality check.
    Shows actual vs XGBoost-predicted consumption for one representative
    client at horizon 24h.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,   label="Actual",    linewidth=1.5)
    ax.plot(forecast, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(f"XGBoost Forecast — Client {client_id}, Horizon {horizon}h")
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Electricity consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sample forecast plot → %s", out_path)


def save_error_by_horizon_plot(results_df: pd.DataFrame, out_path: str):
    """
    DISSERTATION NOTE: Figure 2 — RMSE and MAE averaged across 20 clients.
    Both metrics share kWh units; MSE is deliberately excluded because it
    uses kWh² units and would be visually incomparable with MAE on the same
    axis.  This mirrors the figure design used in the ARIMA baseline.
    """
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
    ax.set_title("XGBoost — Average RMSE and MAE by Forecast Horizon")
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
# DISSERTATION NOTE: Main execution
# ===========================================================================
if __name__ == "__main__":
    # --- Load data ---
    log.info("Loading data from %s", DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)

    # DISSERTATION NOTE: Robust timestamp column handling.
    # The ECL CSV uses "date" as the first column name; we rename it
    # defensively in case the column appears with a different name.
    first_col = df_all.columns[0]
    if first_col != "date":
        df_all = df_all.rename(columns={first_col: "date"})
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

    # --- DISSERTATION NOTE: Client selection (identical to ARIMA baseline) ---
    # Top-10 by mean consumption in training split → high-load industrial clients
    # Random-10 from the remainder (seed=42) → representative population sample
    means      = train_df.mean(axis=0).sort_values(ascending=False)
    top10      = list(means.head(N_TOP).index)
    rng        = np.random.default_rng(RANDOM_SEED)
    remaining  = [c for c in df_all.columns if c not in top10]
    random10   = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    selected   = top10 + random10

    log.info("Top-10 clients (by mean consumption): %s", top10)
    log.info("Random-10 clients (seed=%d): %s", RANDOM_SEED, random10)
    log.info("Total clients to train: %d", len(selected))

    # --- Train and evaluate ---
    all_records   = []
    sample_record = None   # for Figure 1: first successful 24h client

    for idx, cid in enumerate(selected, start=1):
        # Skip clients with >50% missing values
        nan_ratio = df_all[cid].isna().mean()
        if nan_ratio > 0.50:
            log.warning(
                "Client %s has %.1f%% missing values — skipping.", cid, nan_ratio * 100
            )
            continue

        # Forward-fill minor gaps; convert to float64
        train_vals = pd.Series(train_df[cid].values).ffill().bfill().values.astype(float)
        val_vals   = pd.Series(val_df[cid].values).ffill().bfill().values.astype(float)
        test_vals  = pd.Series(test_df[cid].values).ffill().bfill().values.astype(float)

        train_ts = train_df.index
        val_ts   = val_df.index
        test_ts  = test_df.index

        try:
            records = train_and_evaluate(
                client_id=cid,
                train_vals=train_vals,
                val_vals=val_vals,
                test_vals=test_vals,
                train_ts=train_ts,
                val_ts=val_ts,
                test_ts=test_ts,
                client_idx=idx,
                total_clients=len(selected),
            )
        except Exception as exc:
            log.error("Unexpected error for client %s: %s", cid, exc)
            continue

        for r in records:
            if sample_record is None and r["horizon"] == 24:
                sample_record = r.copy()

            r.pop("_actual",   None)
            r.pop("_forecast", None)
            all_records.append(r)

    # --- Save CSV ---
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # --- Console summary table ---
    print("\n" + "=" * 78)
    print("XGBOOST RESULTS SUMMARY — Average across successfully evaluated clients")
    print("=" * 78)
    for h in HORIZONS:
        subset = results_df[results_df["horizon"] == h]
        if subset.empty:
            print(f"  Horizon {h:3d}h → no data")
            continue
        print(
            f"  Horizon {h:3d}h → "
            f"MSE: {subset['MSE'].mean():12.3f} | "
            f"MAE: {subset['MAE'].mean():8.3f} | "
            f"RMSE: {subset['RMSE'].mean():8.3f} | "
            f"Train: {subset['train_time_sec'].mean():.1f}s | "
            f"Inference: {subset['inference_time_sec'].mean():.3f}s"
        )
    print("=" * 78)

    # --- Figure 1: sample forecast ---
    if sample_record and "_actual" in sample_record:
        save_sample_forecast_plot(
            actual    = sample_record["_actual"],
            forecast  = sample_record["_forecast"],
            client_id = sample_record["client_id"],
            horizon   = 24,
            out_path  = os.path.join(FIGURES_DIR, "xgboost_forecast_sample.png"),
        )

    # --- Figure 2: error by horizon ---
    if not results_df.empty:
        save_error_by_horizon_plot(
            results_df,
            out_path=os.path.join(FIGURES_DIR, "xgboost_error_by_horizon.png"),
        )

    print("\nDone. All outputs saved to:", RESULTS_DIR)
