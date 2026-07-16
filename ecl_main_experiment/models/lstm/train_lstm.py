"""
================================================================================
DISSERTATION NOTE: LSTM Deep Learning Baseline — train_lstm.py
================================================================================
Research topic: "When Do Transformer-Based Models Outperform Traditional
Approaches? A Comparative Study of Multivariate Time Series Forecasting"

Model: LSTM (Long Short-Term Memory) via PyTorch torch.nn.LSTM.
       One model is trained per client — univariate, directly comparable with
       the ARIMA and XGBoost baselines.
Dataset: ECL (Electricity Consuming Load), 321 clients, hourly 2012–2014

Sampling strategy (identical to ARIMA and XGBoost for fair comparison):
  - 20 representative clients:
      * Top 10 by mean consumption in the training split (high-load segment)
      * 10 randomly selected from remaining clients (seed=42)
  - Using the same clients across all models is essential for a valid
    apples-to-apples comparison across forecast horizons.

Architecture:
  - 2-layer LSTM, hidden_size=64, dropout=0.2
  - Direct multi-output linear head: hidden state → 168 outputs
  - Input sequence length: 168 hours (one week of context)
  - Forecast horizon: 168 hours (one week ahead, direct prediction)
  - 24h and 48h metrics use the first 24 / first 48 predicted values
    from the single 168-step output — no re-training needed per horizon.

Train/Val/Test split: 70% / 10% / 20% (strict chronological order)
Normalisation: StandardScaler per client, fit on train only; all metrics
               reported in original kWh units after inverse-transform.
Optimiser: Adam, lr=0.001, max 50 epochs, early stopping patience=10.
Batch size: 32. Device priority: CUDA → MPS → CPU.

Fixed-origin forecasting at test time:
  - Input context: the final 168 hours before the test set starts
    (i.e. the tail of the validation split after scaling).
  - Prediction: the 168-step output is compared to the first 168 hours
    of the test set (first 24 for 24h, first 48 for 48h).
  - True test values are NEVER used as input — no data leakage.

Outputs:
  - dissertation/results/raw_metrics/lstm_results.csv
  - dissertation/results/figures/model_diagnostics/lstm_forecast_sample.png
  - dissertation/results/figures/model_diagnostics/lstm_error_by_horizon.png
================================================================================
"""

import os
import sys
import time
import logging
import warnings

# Force line-buffered stdout so progress prints appear immediately when running
# in background or piped to a file (Python defaults to block-buffering otherwise).
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures without display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Paths — all relative to the repo root for portability.
# Do not hard-code local machine-specific paths.
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_PATH   = os.path.join(ROOT, "data", "ECL", "electricity.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw_metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "model_diagnostics")
RESULTS_CSV = os.path.join(RAW_RESULTS_DIR, "lstm_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Hyperparameters — centralised here for reproducibility.
# These match the forecast horizons used in ARIMA and XGBoost baselines.
# ---------------------------------------------------------------------------
HORIZONS      = [24, 48, 168]
N_TOP         = 10          # top clients by mean consumption
N_RANDOM      = 10          # randomly sampled remaining clients
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10        # TEST_RATIO = 0.20 (implicit)

# LSTM architecture
SEQ_LEN       = 168         # input context window (one week of hourly data)
FORECAST_LEN  = 168         # direct multi-step output length
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2

# Training
LEARNING_RATE = 0.001
MAX_EPOCHS    = 50
PATIENCE      = 10          # early stopping patience (epochs with no val improvement)
BATCH_SIZE    = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Device selection — auto-detect in order: CUDA → MPS → CPU.
# CUDA is used on Azure GPU instances. MPS is Apple Silicon. CPU is the fallback.
# The script is portable across all three environments with no code changes.
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)
    return device


# ===========================================================================
# DISSERTATION NOTE: LSTM model definition.
# A 2-layer LSTM followed by a linear projection head that maps the final
# hidden state to FORECAST_LEN (168) outputs simultaneously.
# This is the "direct multi-output" (MIMO) strategy — avoids error accumulation
# from recursive forecasting and is trained end-to-end on the full horizon.
# ===========================================================================
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        forecast_len: int = FORECAST_LEN,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Linear head: maps from the final hidden state to all forecast steps
        self.fc = nn.Linear(hidden_size, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last time-step's hidden state for the output projection
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_size)
        return self.fc(last_hidden)         # (batch, forecast_len)


# ===========================================================================
# DISSERTATION NOTE: Sliding-window dataset for supervised sequence learning.
# Each sample is:
#   X: a window of SEQ_LEN normalised values  → shape (SEQ_LEN, 1)
#   y: the following FORECAST_LEN values       → shape (FORECAST_LEN,)
# Only constructed from within-split data to prevent leakage.
# ===========================================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, forecast_len: int):
        self.seq_len      = seq_len
        self.forecast_len = forecast_len
        self.series       = series.astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.series) - self.seq_len - self.forecast_len + 1)

    def __getitem__(self, idx: int):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.forecast_len]
        # Reshape x to (seq_len, 1) for LSTM input_size=1 (univariate)
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


# ===========================================================================
# DISSERTATION NOTE: Metric computation.
# Metrics are always computed on inverse-transformed values (original kWh
# units) so results are directly comparable with ARIMA and XGBoost baselines.
# ===========================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = float(mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


# ===========================================================================
# DISSERTATION NOTE: Per-client training and evaluation pipeline.
#
# Training data leakage prevention:
#   - Sliding-window samples built ONLY from the training split.
#   - Validation samples built from the validation split, with the last
#     SEQ_LEN rows of the training set prepended so the first validation
#     window always has a full context window available.
#   - Scaler fitted on the training split only; applied to val and test
#     with transform() (not fit_transform()) to prevent leakage.
#
# Fixed-origin test inference:
#   - The input context is the final SEQ_LEN (168) hours before the test
#     set boundary — taken from the end of the scaled train+val series.
#   - The model predicts all FORECAST_LEN=168 steps in one forward pass.
#   - For 24h and 48h horizons, we slice [:24] and [:48] from the output.
#   - True test values are NEVER part of the model input.
# ===========================================================================
def train_and_evaluate(
    client_id: str,
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    test_vals: np.ndarray,
    client_idx: int,
    total_clients: int,
    device: torch.device,
) -> list:
    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    # --- Normalisation ---
    # DISSERTATION NOTE: Scaler fit on training split only. val and test use
    # transform() so their statistics are never seen during fitting.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel().astype(np.float32)
    val_scaled   = scaler.transform(val_vals.reshape(-1, 1)).ravel().astype(np.float32)

    # --- Build datasets ---
    # Training: sliding windows purely within the training split.
    train_dataset = SlidingWindowDataset(train_scaled, SEQ_LEN, FORECAST_LEN)

    # Validation: prepend the last SEQ_LEN points of training data so the first
    # validation window has a complete context (no leakage: only training-split
    # values are used as context, not future val targets).
    val_context   = np.concatenate([train_scaled[-SEQ_LEN:], val_scaled])
    val_dataset   = SlidingWindowDataset(val_context, SEQ_LEN, FORECAST_LEN)

    if len(train_dataset) == 0:
        log.warning("Client %s: insufficient training data for sliding windows. Skipping.", client_id)
        return []

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # --- Model, loss, optimiser ---
    model     = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop with early stopping ---
    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_improve = 0

    t0_train = time.time()
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            # --- Train phase ---
            model.train()
            epoch_train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                epoch_train_loss += loss.item() * len(xb)
            epoch_train_loss /= max(len(train_dataset), 1)

            # --- Validation phase ---
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    epoch_val_loss += criterion(pred, yb).item() * len(xb)
            epoch_val_loss /= max(len(val_dataset), 1)

            # --- Early stopping ---
            if epoch_val_loss < best_val_loss - 1e-8:
                best_val_loss      = epoch_val_loss
                best_state         = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve  = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                log.info(
                    "Client %s: early stopping at epoch %d (best val loss=%.6f)",
                    client_id, epoch, best_val_loss,
                )
                break

    except Exception as exc:
        log.error("Training failed for client %s: %s", client_id, exc)
        return []

    train_time = time.time() - t0_train

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    log.info(
        "Client %s trained | epochs=%d | best_val_loss=%.6f | train_time=%.1fs",
        client_id, epoch, best_val_loss, train_time,
    )

    # --- Fixed-origin test inference ---
    # DISSERTATION NOTE: The input context is the final SEQ_LEN scaled values
    # from the combined train+val series. This places the forecast origin exactly
    # at the start of the test set — identical to the ARIMA and XGBoost approach.
    # True test values are NEVER included in the input tensor.
    train_val_scaled = np.concatenate([train_scaled, val_scaled])
    context = train_val_scaled[-SEQ_LEN:].astype(np.float32)
    context_tensor = torch.tensor(context).unsqueeze(0).unsqueeze(-1).to(device)
    # Shape: (1, SEQ_LEN, 1)

    t0_inf = time.time()
    try:
        with torch.no_grad():
            pred_scaled_tensor = model(context_tensor)   # (1, FORECAST_LEN)
        pred_scaled = pred_scaled_tensor.squeeze(0).cpu().numpy()  # (FORECAST_LEN,)
    except Exception as exc:
        log.error("Inference failed for client %s: %s", client_id, exc)
        return []
    inference_time = time.time() - t0_inf

    # Inverse-transform the full 168-step prediction once
    pred_full = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    records = []
    for horizon in HORIZONS:
        n_test = len(test_vals)
        if n_test < horizon:
            log.warning(
                "Client %s: test set too short for horizon %d (%d steps available). Skipping.",
                client_id, horizon, n_test,
            )
            continue

        # DISSERTATION NOTE: Slice the first `horizon` steps from the 168-step
        # direct output. This avoids re-training per horizon and is equivalent
        # to evaluating shorter horizons from a joint multi-horizon model.
        forecast = pred_full[:horizon]
        actual   = test_vals[:horizon]

        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id":          client_id,
            "horizon":            horizon,
            "MSE":                round(metrics["MSE"],  4),
            "MAE":                round(metrics["MAE"],  4),
            "RMSE":               round(metrics["RMSE"], 4),
            "train_time_sec":     round(train_time,      2),
            "inference_time_sec": round(inference_time,  6),
            # Private fields for Figure 1 only — removed before CSV write
            "_actual":   actual,
            "_forecast": forecast,
        })
        print(
            f"  Horizon {horizon:3d}h → "
            f"MSE={metrics['MSE']:.3f} | "
            f"MAE={metrics['MAE']:.3f} | "
            f"RMSE={metrics['RMSE']:.3f} | "
            f"Inference={inference_time:.4f}s"
        )

    return records


# ===========================================================================
# DISSERTATION NOTE: Figures — two required outputs matching the format of
# the ARIMA and XGBoost baseline figures.
# ===========================================================================

def save_sample_forecast_plot(actual, forecast, client_id, horizon, out_path: str):
    """Figure 1: actual vs LSTM-predicted for one representative client at 24h."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,   label="Actual",    linewidth=1.5)
    ax.plot(forecast, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(f"LSTM Forecast — Client {client_id}, Horizon {horizon}h")
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Electricity consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sample forecast plot → %s", out_path)


def save_error_by_horizon_plot(results_df: pd.DataFrame, out_path: str):
    """
    Figure 2: RMSE and MAE averaged across 20 clients per horizon.
    DISSERTATION NOTE: MSE is deliberately excluded because it is measured in
    kWh² while RMSE and MAE share kWh units — plotting them together would be
    misleading. This mirrors the figure design of the ARIMA and XGBoost scripts.
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
    ax.set_title("LSTM — Average RMSE and MAE by Forecast Horizon")
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
# DISSERTATION NOTE: Main execution — orchestrates the full pipeline.
# ===========================================================================
if __name__ == "__main__":
    device = get_device()

    # --- Load data ---
    log.info("Loading data from %s", DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)

    # DISSERTATION NOTE: Robust timestamp column handling.
    # The ECL CSV uses "date" as the first column; we rename defensively in case
    # the column appears with a different name (e.g. "Unnamed: 0", "timestamp").
    first_col = df_all.columns[0]
    if first_col != "date":
        log.info("Renaming first column '%s' → 'date'", first_col)
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

    # --- DISSERTATION NOTE: Client selection (identical to ARIMA and XGBoost) ---
    # Top-10 by mean consumption in the training split → high-load industrial clients.
    # Random-10 from the remaining 311 clients (seed=42) → representative sample.
    # Using the same 20 clients across all models is critical for a fair comparison.
    means     = train_df.mean(axis=0).sort_values(ascending=False)
    top10     = list(means.head(N_TOP).index)
    rng       = np.random.default_rng(RANDOM_SEED)
    remaining = [c for c in df_all.columns if c not in top10]
    random10  = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    selected  = top10 + random10

    log.info("Top-10 clients (by mean consumption): %s", top10)
    log.info("Random-10 clients (seed=%d): %s", RANDOM_SEED, random10)
    log.info("Total clients to train: %d", len(selected))

    # --- Train and evaluate ---
    all_records   = []
    sample_record = None   # for Figure 1: first successful client at horizon 24h

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

        try:
            records = train_and_evaluate(
                client_id=cid,
                train_vals=train_vals,
                val_vals=val_vals,
                test_vals=test_vals,
                client_idx=idx,
                total_clients=len(selected),
                device=device,
            )
        except Exception as exc:
            log.error("Unexpected error for client %s: %s", cid, exc)
            continue

        for r in records:
            if sample_record is None and r["horizon"] == 24:
                sample_record = r.copy()

            # Remove private plot fields before appending to CSV records
            r.pop("_actual",   None)
            r.pop("_forecast", None)
            all_records.append(r)

    # --- Save CSV ---
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # --- Console summary table ---
    print("\n" + "=" * 78)
    print("LSTM RESULTS SUMMARY — Average across successfully evaluated clients")
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
            f"Inference: {subset['inference_time_sec'].mean():.4f}s"
        )
    print("=" * 78)

    # --- Figure 1: sample forecast ---
    if sample_record and "_actual" in sample_record:
        save_sample_forecast_plot(
            actual    = sample_record["_actual"],
            forecast  = sample_record["_forecast"],
            client_id = sample_record["client_id"],
            horizon   = 24,
            out_path  = os.path.join(FIGURES_DIR, "lstm_forecast_sample.png"),
        )

    # --- Figure 2: error by horizon ---
    if not results_df.empty:
        save_error_by_horizon_plot(
            results_df,
            out_path=os.path.join(FIGURES_DIR, "lstm_error_by_horizon.png"),
        )

    print("\nDone. All outputs saved to:", RESULTS_DIR)
