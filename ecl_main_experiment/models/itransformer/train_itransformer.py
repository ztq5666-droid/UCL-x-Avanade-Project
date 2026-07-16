"""
================================================================================
DISSERTATION NOTE: iTransformer — train_itransformer.py
================================================================================
Research topic: "When Do Transformer-Based Models Outperform Traditional
Approaches? A Comparative Study of Multivariate Time Series Forecasting"

Model: iTransformer (Inverted Transformer) — the PRIMARY model of this
       dissertation study.
Dataset: ECL (Electricity Consuming Load), 321 clients, hourly 2012–2014

Core experimental design (critical difference from ARIMA, XGBoost, LSTM):
  - iTransformer is trained as a TRUE MULTIVARIATE model using ALL 321 client
    variables simultaneously in every forward pass.
  - The model's inverted attention mechanism attends ACROSS variables rather
    than across time, capturing cross-variable dependencies.
  - This is the central design property evaluated in this dissertation:
    whether capturing inter-variable correlations improves forecasting over
    per-client univariate approaches.

Evaluation on 20 representative clients (for fair comparison with baselines):
  - Metrics are reported only on the same 20 clients used by ARIMA, XGBoost,
    and LSTM (top-10 by mean consumption + random-10, seed=42).
  - The model still trains on all 321 variables; client subsetting only
    occurs at the metric-reporting stage.

Architecture:
  - Official THUML iTransformer (https://github.com/thuml/iTransformer)
  - seq_len=96, pred_len=168 (6 days ahead), d_model=512, n_heads=8,
    e_layers=3, d_ff=512, dropout=0.1, factor=1
  - Input/output: [batch, time, 321_variables]
  - Uses internal instance normalisation (RevIN-equivalent, use_norm=True)
    if available; falls back to per-variable StandardScaler otherwise.

Forecasting strategy:
  - One fixed-origin forward pass from the test-set boundary.
  - Context: final seq_len=96 hours before the test set (from train+val history).
  - 24h and 48h metrics slice the first 24 / first 48 steps of the 168-step
    output — no re-training per horizon.

Train/Val/Test split: 70% / 10% / 20% (strict chronological order, no shuffle)
Optimiser: Adam, lr=0.0001, max 10 epochs, early stopping patience=3.
Batch size: 32 (auto-reduces to 16 → 8 on GPU OOM).
Device priority: CUDA → MPS → CPU.

Outputs:
  - dissertation/results/raw_metrics/itransformer_results.csv
  - dissertation/results/figures/model_diagnostics/itransformer_forecast_sample.png
  - dissertation/results/figures/model_diagnostics/itransformer_error_by_horizon.png
================================================================================
"""

import os
import sys
import time
import logging
import warnings
import copy
import subprocess
import types

# Force line-buffered stdout so progress prints appear immediately.
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Paths — all relative to the repository root.
# ITRANSFORMER_REPO_DIR points to the cloned official THUML repo.
# The repo is cloned here automatically if it does not already exist.
# ---------------------------------------------------------------------------
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
ROOT                  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ITRANSFORMER_REPO_DIR = os.path.join(SCRIPT_DIR, "iTransformer")

DATA_PATH   = os.path.join(ROOT, "data", "ECL", "electricity.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw_metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "model_diagnostics")
RESULTS_CSV = os.path.join(RAW_RESULTS_DIR, "itransformer_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Hyperparameters — centralised for reproducibility.
# These values follow the iTransformer paper's ETTh1 / ECL configuration.
# seq_len=96: one week of 4-hourly or 4 days of hourly context.
# pred_len=168: exactly one week ahead (7 days × 24 hours).
# ---------------------------------------------------------------------------
SEQ_LEN       = 96
PRED_LEN      = 168
D_MODEL       = 512
N_HEADS       = 8
E_LAYERS      = 3
D_FF          = 512
DROPOUT       = 0.1
FACTOR        = 1

HORIZONS      = [24, 48, 168]
N_VARS        = 321         # all ECL client columns
N_TOP         = 10          # top clients by mean train consumption
N_RANDOM      = 10          # randomly sampled remaining clients
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10        # TEST_RATIO = 0.20 (implicit)

LEARNING_RATE = 0.0001
MAX_EPOCHS    = 10
PATIENCE      = 3           # early stopping: epochs with no val improvement
INITIAL_BATCH = 32          # auto-reduces on GPU OOM

OFFICIAL_REPO_URL = "https://github.com/thuml/iTransformer.git"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DISSERTATION NOTE: Device selection — CUDA → MPS → CPU, matching all other
# baseline scripts in this dissertation for fair wall-clock comparison.
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
# DISSERTATION NOTE: Official THUML iTransformer repo management.
# The repo is cloned into ./dissertation/models/itransformer/iTransformer/
# if it does not already exist. No other model folders are modified.
# The official model class is imported from models.iTransformer within the
# cloned repo after inserting the repo root into sys.path.
# ===========================================================================
def ensure_itransformer_repo() -> str:
    """Clone the official THUML iTransformer repo if not already present.

    Returns the path to the repo root (ITRANSFORMER_REPO_DIR).
    """
    model_file = os.path.join(ITRANSFORMER_REPO_DIR, "model", "iTransformer.py")

    if os.path.isfile(model_file):
        log.info("iTransformer repo found at %s", ITRANSFORMER_REPO_DIR)
        return ITRANSFORMER_REPO_DIR

    log.info(
        "iTransformer repo not found. Cloning from %s into %s ...",
        OFFICIAL_REPO_URL, ITRANSFORMER_REPO_DIR,
    )
    result = subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, ITRANSFORMER_REPO_DIR],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed (return code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    log.info("Clone successful.")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Expected model file not found after clone: {model_file}\n"
            "The repository structure may have changed. "
            "Inspect the repo and adjust the import path in this script."
        )
    return ITRANSFORMER_REPO_DIR


def import_itransformer_model(repo_dir: str):
    """Add the repo to sys.path and import the official Model class.

    Tries the canonical path model.iTransformer first, then falls back to
    alternative locations in case the repo structure differs.
    """
    # Insert repo root so `import model.iTransformer` resolves correctly.
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Attempt 1: canonical import path used in the official THUML repo.
    try:
        from model.iTransformer import Model  # noqa: PLC0415
        log.info("Imported Model from model.iTransformer (canonical path).")
        return Model
    except ImportError as e1:
        log.warning("Canonical import failed (%s); trying alternative paths.", e1)

    # Attempt 2: some repo versions use a flat structure.
    try:
        from iTransformer import Model  # noqa: PLC0415
        log.info("Imported Model from iTransformer (flat path).")
        return Model
    except ImportError as e2:
        log.warning("Flat import failed (%s); trying model subdirectory.", e2)

    # Attempt 3: import directly after adding the model/ directory to sys.path.
    try:
        if os.path.join(repo_dir, "model") not in sys.path:
            sys.path.insert(0, os.path.join(repo_dir, "model"))
        from iTransformer import Model  # noqa: PLC0415
        log.info("Imported Model from model/iTransformer (subdirectory path).")
        return Model
    except ImportError as e3:
        raise ImportError(
            "Could not import the iTransformer Model class. "
            "Inspect the cloned repo and update the import path manually.\n"
            f"Tried: model.iTransformer, iTransformer, model/iTransformer\n"
            f"Last error: {e3}"
        ) from e3


# ===========================================================================
# DISSERTATION NOTE: iTransformer configuration.
# The official THUML model expects a configs namespace with specific attributes.
# enc_in / dec_in / c_out are all set to N_VARS=321 for multivariate I/O.
# use_norm=True enables the model's internal instance normalisation (RevIN),
# which normalises each variable per-sample and denormalises the output.
# When RevIN is active, no external StandardScaler is applied.
# ===========================================================================
def make_configs(n_vars: int, use_norm: bool = True) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        seq_len          = SEQ_LEN,
        label_len        = 0,           # not used by iTransformer (encoder-only)
        pred_len         = PRED_LEN,
        enc_in           = n_vars,
        dec_in           = n_vars,
        c_out            = n_vars,
        d_model          = D_MODEL,
        n_heads          = N_HEADS,
        e_layers         = E_LAYERS,
        d_layers         = 1,           # decoder depth, unused by iTransformer
        d_ff             = D_FF,
        dropout          = DROPOUT,
        factor           = FACTOR,
        embed             = "timeF",    # time-feature embedding type
        freq              = "h",        # hourly data
        activation        = "gelu",
        output_attention  = False,
        use_norm          = use_norm,
        class_strategy    = "projection",
    )
    return cfg


def probe_revin(Model, n_vars: int) -> bool:
    """Return True if the model supports use_norm=True without error.

    DISSERTATION NOTE: RevIN (use_norm=True) normalises inputs per-sample
    per-variable and denormalises outputs inside the forward pass, so
    predictions are returned in original electricity-consumption units.
    When RevIN is not supported, we fall back to per-variable StandardScaler
    fitted on the training split (consistent with ARIMA, XGBoost, LSTM).
    """
    try:
        cfg = make_configs(n_vars, use_norm=True)
        model = Model(cfg)
        x = torch.zeros(1, SEQ_LEN, n_vars)
        mark = torch.zeros(1, SEQ_LEN, 4)
        with torch.no_grad():
            out = model(x, mark, None, None)
        if isinstance(out, tuple):
            out = out[0]
        # Output must be [1, pred_len, n_vars]
        assert out.shape == (1, PRED_LEN, n_vars), f"unexpected shape {out.shape}"
        del model
        log.info("RevIN (use_norm=True) is available and validated.")
        return True
    except Exception as exc:
        log.warning("RevIN probe failed (%s). Will use StandardScaler.", exc)
        return False


# ===========================================================================
# DISSERTATION NOTE: Multivariate sliding-window dataset.
#
# X shape: [batch_size, seq_len, N_VARS]  — all 321 variables simultaneously.
# y shape: [batch_size, pred_len, N_VARS] — predict all 321 variables ahead.
#
# This is fundamentally different from ARIMA / XGBoost / LSTM, where a
# separate model is trained per client on a single univariate series.
# iTransformer sees the full cross-variable context in every training sample.
#
# No data leakage: windows for validation use at most the final seq_len rows
# of training data as context; no test-split values are included.
# ===========================================================================
class MultivariateWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        # data: [T, N_VARS], already normalised (or raw if use_revin=True)
        self.data     = data.astype(np.float32)
        self.seq_len  = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]                             # [seq_len, N]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # [pred_len, N]
        return torch.tensor(x), torch.tensor(y)


# ===========================================================================
# DISSERTATION NOTE: Metric computation.
# Always applied to inverse-transformed (original kWh) values so results
# are directly comparable with ARIMA, XGBoost, and LSTM baselines.
# ===========================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = float(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    mae  = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


# ===========================================================================
# DISSERTATION NOTE: iTransformer forward-pass helper.
# The official model signature is:
#   forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
# x_mark_enc carries time features; we pass zeros because iTransformer's
# inverted attention operates on the variable dimension, not the time
# dimension, so temporal positional encoding is secondary.
# x_dec / x_mark_dec are not used by the encoder-only iTransformer.
# If the model returns a tuple (output, attentions), only output[0] is used.
# ===========================================================================
def model_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run one forward pass. x: [B, seq_len, N_vars]. Returns [B, pred_len, N_vars]."""
    B, T, _ = x.shape
    x_mark = torch.zeros(B, T, 4, dtype=torch.float32, device=x.device)
    out = model(x, x_mark, None, None)
    if isinstance(out, tuple):
        out = out[0]
    return out  # [B, pred_len, N_vars]


# ===========================================================================
# DISSERTATION NOTE: Training loop.
#
# Batch-size auto-reduction:
#   iTransformer with 321 variables and d_model=512 is memory-intensive.
#   If a CUDA OOM error occurs, the batch size is halved (32→16→8) and
#   training restarts from epoch 1 with the reduced batch. This is logged
#   clearly so the actual batch size used is visible in the console output.
#
# Early stopping monitors validation loss with patience=3.
# The best model state (lowest val loss) is restored after training.
# ===========================================================================
def run_training(
    Model_cls,
    train_data: np.ndarray,
    val_data: np.ndarray,
    n_vars: int,
    use_revin: bool,
    device: torch.device,
) -> tuple:
    """Train the iTransformer and return (model, train_time_sec, batch_used).

    DISSERTATION NOTE: train_data and val_data are 2-D arrays [T, N_VARS].
    If use_revin=True they contain raw electricity values (RevIN handles
    normalisation internally). If use_revin=False they are already scaled
    by StandardScaler (one scaler per variable, fit on training split only).
    """
    batch_size = INITIAL_BATCH
    configs    = make_configs(n_vars, use_norm=use_revin)

    # Validation context: prepend the last seq_len rows of training data so
    # the first validation window always has a complete context window.
    # This uses only training-split values as context — no leakage.
    val_context = np.concatenate([train_data[-SEQ_LEN:], val_data], axis=0)

    t0 = time.time()

    for attempt in range(3):  # up to 3 batch-size reductions
        try:
            model = Model_cls(configs).to(device)
            log.info(
                "Model built | params=%d | batch_size=%d",
                sum(p.numel() for p in model.parameters()),
                batch_size,
            )

            train_dataset = MultivariateWindowDataset(train_data, SEQ_LEN, PRED_LEN)
            val_dataset   = MultivariateWindowDataset(val_context, SEQ_LEN, PRED_LEN)

            if len(train_dataset) == 0:
                raise ValueError("Training dataset is empty — insufficient data for sliding windows.")

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=(device.type == "cuda"),
            )

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            best_val_loss     = float("inf")
            best_state        = None
            epochs_no_improve = 0
            last_epoch        = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                last_epoch = epoch

                # --- Train phase ---
                model.train()
                epoch_train_loss = 0.0
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    pred = model_forward(model, x_batch)  # [B, pred_len, N]
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * len(x_batch)
                epoch_train_loss /= max(len(train_dataset), 1)

                # --- Validation phase ---
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        pred = model_forward(model, x_batch)
                        epoch_val_loss += criterion(pred, y_batch).item() * len(x_batch)
                epoch_val_loss /= max(len(val_dataset), 1)

                print(
                    f"Epoch {epoch:2d}/{MAX_EPOCHS} — "
                    f"train loss: {epoch_train_loss:.4f} | "
                    f"val loss: {epoch_val_loss:.4f}"
                )

                # --- Early stopping ---
                if epoch_val_loss < best_val_loss - 1e-8:
                    best_val_loss     = epoch_val_loss
                    best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:
                    log.info(
                        "Early stopping at epoch %d (best val loss=%.6f).",
                        epoch, best_val_loss,
                    )
                    break

            # Restore best weights
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            train_time = time.time() - t0
            log.info(
                "Training complete | epochs=%d | best_val_loss=%.6f | "
                "train_time=%.1fs | batch_size=%d",
                last_epoch, best_val_loss, train_time, batch_size,
            )
            return model, train_time, batch_size

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 8:
                new_batch = batch_size // 2
                log.warning(
                    "GPU OOM with batch_size=%d. Reducing to %d and restarting.",
                    batch_size, new_batch,
                )
                batch_size = new_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t0 = time.time()  # reset timer after restart
            else:
                raise

    raise RuntimeError(
        f"Training failed even with minimum batch_size=8. "
        "Consider reducing d_model or e_layers."
    )


# ===========================================================================
# DISSERTATION NOTE: Figures
# ===========================================================================

def save_sample_forecast_plot(
    actual: np.ndarray,
    forecast: np.ndarray,
    client_id: str,
    horizon: int,
    out_path: str,
):
    """Figure 1: actual vs iTransformer-predicted for one representative client.

    DISSERTATION NOTE: Shows the qualitative forecast quality for the 24h
    horizon. This mirrors Figure 1 in the ARIMA, XGBoost, and LSTM scripts
    so the plots are directly visually comparable in the dissertation.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,   label="Actual",    linewidth=1.5)
    ax.plot(forecast, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title(f"iTransformer Forecast — Client {client_id}, Horizon {horizon}h")
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Electricity consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved sample forecast plot → %s", out_path)


def save_error_by_horizon_plot(results_df: pd.DataFrame, out_path: str):
    """Figure 2: average RMSE and MAE across 20 evaluation clients per horizon.

    DISSERTATION NOTE: MSE is deliberately excluded because it is measured
    in kWh² while RMSE and MAE share kWh units. Plotting them on the same
    axis would be visually misleading. This mirrors the figure design of all
    other baseline scripts.
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
    ax.set_title("iTransformer — Average RMSE and MAE by Forecast Horizon")
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
    device = get_device()

    # --- Step 1: Ensure the official THUML repo is available and import Model ---
    repo_dir   = ensure_itransformer_repo()
    Model_cls  = import_itransformer_model(repo_dir)

    # --- Step 2: Probe for RevIN support ---
    # DISSERTATION NOTE: If the official model supports use_norm=True (RevIN),
    # we pass raw electricity values and the model normalises internally.
    # The model's output is then already in original kWh units — no external
    # inverse-transform is needed.
    # If use_norm is not supported, we apply StandardScaler per variable
    # (fitted on training split only, identical to the other baselines).
    USE_REVIN = probe_revin(Model_cls, N_VARS)

    # --- Step 3: Load and split data ---
    log.info("Loading data from %s", DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)

    # DISSERTATION NOTE: Robust timestamp handling — rename the first column
    # to "date" regardless of how the ECL CSV names it.
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
        "Split: Train=%d | Val=%d | Test=%d rows | total_vars=%d",
        len(train_df), len(val_df), len(test_df), len(df_all.columns),
    )

    # --- Step 4: Select the 20 evaluation clients (identical to all baselines) ---
    # DISSERTATION NOTE: The evaluation clients are selected from the training
    # split so their identity is never influenced by validation or test data.
    # Using the SAME 20 clients across ARIMA, XGBoost, LSTM, TabPFN, and
    # iTransformer is essential for a fair apples-to-apples comparison.
    #
    # iTransformer is TRAINED on all 321 variables — client subsetting only
    # occurs at the metric-reporting stage after the single forward pass.
    means     = train_df.mean(axis=0).sort_values(ascending=False)
    top10     = list(means.head(N_TOP).index)
    rng       = np.random.default_rng(RANDOM_SEED)
    remaining = [c for c in df_all.columns if c not in top10]
    random10  = [str(c) for c in rng.choice(remaining, size=N_RANDOM, replace=False)]
    eval_clients = top10 + random10   # 20 clients for metric reporting

    log.info("Top-10 evaluation clients (by mean consumption): %s", top10)
    log.info("Random-10 evaluation clients (seed=%d): %s", RANDOM_SEED, random10)
    log.info(
        "Training: ALL %d variables. Evaluation reported on %d clients.",
        N_VARS, len(eval_clients),
    )

    # Map evaluation client names to column indices in the full dataset
    all_cols       = list(df_all.columns)
    eval_col_idx   = [all_cols.index(c) for c in eval_clients]

    # --- Step 5: Forward-fill minor gaps; convert to numpy arrays ---
    train_raw = df_all.iloc[:train_end].ffill().bfill().values.astype(np.float64)
    val_raw   = df_all.iloc[train_end:val_end].ffill().bfill().values.astype(np.float64)
    test_raw  = df_all.iloc[val_end:].ffill().bfill().values.astype(np.float64)
    # Shapes: [T_train, 321], [T_val, 321], [T_test, 321]

    # --- Step 6: Normalisation ---
    if USE_REVIN:
        # DISSERTATION NOTE: RevIN handles normalisation internally.
        # Raw values are passed to the model directly. Predictions and actuals
        # are both in original kWh units — no external scaling or inverse-
        # transform is required.
        log.info("Normalisation: using model-internal RevIN (use_norm=True).")
        train_data = train_raw.astype(np.float32)
        val_data   = val_raw.astype(np.float32)
        test_data  = test_raw.astype(np.float32)
        scalers    = None
    else:
        # DISSERTATION NOTE: Fallback — StandardScaler per variable, fitted
        # exclusively on the training split. This is identical to the
        # normalisation strategy used in ARIMA, XGBoost, and LSTM.
        log.info("Normalisation: using per-variable StandardScaler (no RevIN).")
        scalers = []
        train_scaled = np.empty_like(train_raw, dtype=np.float32)
        val_scaled   = np.empty_like(val_raw,   dtype=np.float32)
        for col in range(train_raw.shape[1]):
            sc = StandardScaler()
            train_scaled[:, col] = sc.fit_transform(train_raw[:, col].reshape(-1, 1)).ravel()
            val_scaled[:, col]   = sc.transform(val_raw[:, col].reshape(-1, 1)).ravel()
            scalers.append(sc)
        train_data = train_scaled
        val_data   = val_scaled
        # test_data is never passed to the model — we only use raw test values
        # for metric computation after inverse-transforming predictions.
        test_data  = test_raw.astype(np.float32)

    # --- Step 7: Train the single multivariate iTransformer ---
    # DISSERTATION NOTE: One model is trained on the full [T, 321] dataset.
    # This is the key experimental design: the model learns cross-variable
    # dependencies simultaneously, unlike the per-client baselines.
    print("\n" + "=" * 78)
    print("Training iTransformer on ALL 321 variables simultaneously ...")
    print("=" * 78)

    model, train_time_sec, batch_used = run_training(
        Model_cls  = Model_cls,
        train_data = train_data,
        val_data   = val_data,
        n_vars     = N_VARS,
        use_revin  = USE_REVIN,
        device     = device,
    )

    log.info("Total training time: %.1f s (batch_size used: %d)", train_time_sec, batch_used)

    # --- Step 8: Fixed-origin test inference ---
    # DISSERTATION NOTE: The forecast origin is the start of the test set.
    # Context: the final seq_len=96 hours immediately before the test set,
    # taken from the combined train+val history (never from test values).
    # One forward pass produces all 321 variable forecasts simultaneously.
    if USE_REVIN:
        # Raw history (RevIN normalises inside the model)
        train_val_raw = np.concatenate([train_raw, val_raw], axis=0).astype(np.float32)
        context = train_val_raw[-SEQ_LEN:]  # [seq_len, 321]
    else:
        # Scaled history
        train_val_scaled = np.concatenate([train_data, val_data], axis=0).astype(np.float32)
        context = train_val_scaled[-SEQ_LEN:]   # [seq_len, 321]

    context_tensor = torch.tensor(context).unsqueeze(0).to(device)  # [1, seq_len, 321]

    t0_inf = time.time()
    try:
        with torch.no_grad():
            pred_tensor = model_forward(model, context_tensor)   # [1, pred_len, 321]
        pred_all_scaled = pred_tensor.squeeze(0).cpu().numpy()   # [pred_len, 321]
    except Exception as exc:
        log.error("Inference failed: %s", exc)
        raise
    inference_time_sec = time.time() - t0_inf
    log.info("Inference complete in %.4f s", inference_time_sec)

    # Inverse-transform if using external StandardScaler
    if USE_REVIN:
        # RevIN already denormalised inside the model; output is in original units.
        pred_all = pred_all_scaled   # [pred_len, 321]
    else:
        pred_all = np.empty_like(pred_all_scaled)
        for col in range(N_VARS):
            pred_all[:, col] = scalers[col].inverse_transform(
                pred_all_scaled[:, col].reshape(-1, 1)
            ).ravel()
    # pred_all: [pred_len, 321] in original kWh units

    # --- Step 9: Evaluate on the 20 representative clients ---
    # DISSERTATION NOTE: We evaluate only on the same 20 clients used by all
    # other baselines so the comparison is fair. The model was trained on all
    # 321 variables; we simply slice the columns corresponding to the 20
    # evaluation clients from the full 321-variable prediction array.
    #
    # train_time_sec is identical for all client/horizon rows because there
    # is only one shared model (not one model per client).
    # inference_time_sec reflects the single fixed-origin forward pass.
    all_records   = []
    sample_record = None  # for Figure 1: first successful client at horizon 24h

    n_test = len(test_df)

    for client_rank, cid in enumerate(eval_clients):
        col_idx = eval_col_idx[client_rank]

        try:
            actual_full = test_raw[:, col_idx]   # original kWh, full test set

            for horizon in HORIZONS:
                if n_test < horizon:
                    log.warning(
                        "Client %s: test set too short for horizon %d (%d steps). Skipping.",
                        cid, horizon, n_test,
                    )
                    continue

                # Slice the first `horizon` steps from the 168-step output.
                forecast = pred_all[:horizon, col_idx]   # [horizon]
                actual   = actual_full[:horizon]         # [horizon]

                metrics = compute_metrics(actual, forecast)
                record  = {
                    "client_id":          cid,
                    "horizon":            horizon,
                    "MSE":                round(metrics["MSE"],  4),
                    "MAE":                round(metrics["MAE"],  4),
                    "RMSE":               round(metrics["RMSE"], 4),
                    # DISSERTATION NOTE: train_time_sec is the same for every
                    # row because iTransformer is a single shared model — unlike
                    # per-client baselines where each model has its own time.
                    # This directly supports RQ2 (wall-clock comparison).
                    "train_time_sec":     round(train_time_sec,    2),
                    "inference_time_sec": round(inference_time_sec, 6),
                    # Private fields for Figure 1 only
                    "_actual":   actual,
                    "_forecast": forecast,
                    "_client_id": cid,
                }
                all_records.append(record)

                if sample_record is None and horizon == 24:
                    sample_record = record.copy()

        except Exception as exc:
            log.error("Metric calculation failed for client %s: %s — skipping.", cid, exc)
            continue

    # --- Step 10: Save CSV ---
    csv_records = []
    for r in all_records:
        csv_records.append({
            "client_id":          r["client_id"],
            "horizon":            r["horizon"],
            "MSE":                r["MSE"],
            "MAE":                r["MAE"],
            "RMSE":               r["RMSE"],
            "train_time_sec":     r["train_time_sec"],
            "inference_time_sec": r["inference_time_sec"],
        })

    results_df = pd.DataFrame(csv_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # --- Console summary ---
    print("\n" + "=" * 78)
    print("iTransformer Results (average across 20 evaluation clients):")
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
    print(f"\nTotal training time:   {train_time_sec:.1f}s")
    print(f"Total inference time:  {inference_time_sec:.4f}s")

    # --- Figure 1: sample forecast ---
    if sample_record is not None and "_actual" in sample_record:
        save_sample_forecast_plot(
            actual    = sample_record["_actual"],
            forecast  = sample_record["_forecast"],
            client_id = sample_record["_client_id"],
            horizon   = 24,
            out_path  = os.path.join(FIGURES_DIR, "itransformer_forecast_sample.png"),
        )

    # --- Figure 2: error by horizon ---
    if not results_df.empty:
        save_error_by_horizon_plot(
            results_df,
            out_path=os.path.join(FIGURES_DIR, "itransformer_error_by_horizon.png"),
        )

    print("\nDone. All outputs saved to:", RESULTS_DIR)
