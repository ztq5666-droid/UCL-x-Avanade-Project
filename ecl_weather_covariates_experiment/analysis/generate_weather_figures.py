"""
Publication-quality figures for the Weather Extension experiment.

Figures generated
-----------------
W1  Grouped bar chart: model RMSE across 3 horizons (ARIMA as load-only baseline).
W2  Improvement % vs ARIMA baseline (horizontal bars, green / red).
W3  XGBoost feature importance — weather vs calendar/lag feature groups.
W4  Pearson correlation heatmap: exogenous features vs sampled load clients.

Usage
-----
    python ecl_weather_covariates_experiment/models/analysis/generate_weather_figures.py

All figures saved to:
    ecl_weather_covariates_experiment/outputs/figures/weather_extension/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent.parent   # analysis/ → ecl_weather_covariates_experiment/
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_RESULTS_DIR     = ROOT / "outputs" / "raw_metrics"
OUT_DIR             = ROOT / "outputs" / "figures" / "weather_extension"
CHECKPOINT_DIR      = ROOT / "outputs" / "model_checkpoints"
ATTENTION_PATH      = CHECKPOINT_DIR / "itransformer_exog_attention.npy"
COLUMNS_PATH        = CHECKPOINT_DIR / "itransformer_exog_columns.json"
IMPORTANCE_CSV      = RAW_RESULTS_DIR / "xgboost_exog_past_weather_feature_importance.csv"
SHAP_CSV            = RAW_RESULTS_DIR / "xgboost_shap_horizon24.csv"

HORIZONS = [24, 48, 168]

# Consistent colour palette across all 5 figures
COLOR_MAP = {
    "ARIMA":        "#1f77b4",
    "XGBoost":      "#ff7f0e",
    "LSTM":         "#2ca02c",
    "iTransformer": "#d62728",
    "TabPFN-TS":    "#9467bd",
}

# CSV model-name → display label
MODEL_LABELS = {
    "arima_load_only":            "ARIMA",
    "xgboost_exog_past_weather":  "XGBoost",
    "lstm_exog":                  "LSTM",
    "itransformer_style_exog":    "iTransformer",
    "tabpfn_ts_exog_past_weather":"TabPFN-TS",
}

RESULT_FILES = {
    "arima_load_only":            RAW_RESULTS_DIR / "arima_load_only_results.csv",
    "xgboost_exog_past_weather":  RAW_RESULTS_DIR / "xgboost_exog_past_weather_results.csv",
    "lstm_exog":                  RAW_RESULTS_DIR / "lstm_exog_results.csv",
    "itransformer_style_exog":    RAW_RESULTS_DIR / "itransformer_exog_results.csv",
    "tabpfn_ts_exog_past_weather":RAW_RESULTS_DIR / "tabpfn_exog_past_weather_results.csv",
}

DISPLAY_ORDER = ["ARIMA", "XGBoost", "LSTM", "iTransformer", "TabPFN-TS"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)


def load_summary() -> pd.DataFrame:
    """Return mean RMSE per (display_name, horizon)."""
    frames = []
    for key, path in RESULT_FILES.items():
        if not path.exists():
            print(f"  [skip] missing {path.name}")
            continue
        df = pd.read_csv(path)
        df["display_name"] = _display(key)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No result CSVs found.")
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["display_name", "horizon"])["RMSE"]
        .mean()
        .reset_index()
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# W1 — Grouped bar chart: RMSE by model × horizon
# ---------------------------------------------------------------------------

def figure_w1_grouped_bar(summary: pd.DataFrame) -> None:
    pivot = (
        summary.pivot(index="display_name", columns="horizon", values="RMSE")
        .reindex(DISPLAY_ORDER)
    )
    n_models = len(pivot)
    n_horizons = len(HORIZONS)
    x = np.arange(n_models)
    width = 0.22
    offsets = np.linspace(-(n_horizons - 1) / 2 * width, (n_horizons - 1) / 2 * width, n_horizons)
    horizon_colours = ["#4878d0", "#ee854a", "#6acc65"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (h, colour) in enumerate(zip(HORIZONS, horizon_colours)):
        if h not in pivot.columns:
            continue
        vals = pivot[h].values
        bars = ax.bar(x + offsets[i], vals, width, label=f"{h}h", color=colour, alpha=0.88)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("Mean RMSE (kWh)", fontsize=11)
    ax.set_title("W1 — Model RMSE by Forecast Horizon\n(ARIMA: load-only; others: past weather exogenous)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Horizon", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, "fig_01_rmse_by_model.png")


# ---------------------------------------------------------------------------
# W2 — Improvement % vs ARIMA baseline
# ---------------------------------------------------------------------------

def figure_w2_improvement(summary: pd.DataFrame) -> None:
    arima_rmse = (
        summary[summary["display_name"] == "ARIMA"]
        .set_index("horizon")["RMSE"]
    )
    if arima_rmse.empty:
        print("  [skip W2] ARIMA results missing.")
        return

    records = []
    for _, row in summary.iterrows():
        h = row["horizon"]
        if h not in arima_rmse.index:
            continue
        baseline = arima_rmse[h]
        pct = (baseline - row["RMSE"]) / baseline * 100   # positive = better than ARIMA
        records.append({"display_name": row["display_name"], "horizon": h, "improvement_pct": pct})

    df = pd.DataFrame(records)
    df = df[df["display_name"] != "ARIMA"]   # baseline is 0 %; skip it

    n_models = len(df["display_name"].unique())
    horizon_colours = ["#4878d0", "#ee854a", "#6acc65"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    for ax, (h, colour) in zip(axes, zip(HORIZONS, horizon_colours)):
        sub = df[df["horizon"] == h].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.set_index("display_name").reindex(
            [m for m in DISPLAY_ORDER if m != "ARIMA"]
        ).reset_index()
        vals = sub["improvement_pct"].values
        bar_colours = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        bars = ax.barh(sub["display_name"], vals, color=bar_colours, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            x_pos = val + (0.3 if val >= 0 else -0.3)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", va="center", ha=ha, fontsize=9)

        # Annotate iTransformer rank
        ranked = sub.sort_values("improvement_pct", ascending=False).reset_index(drop=True)
        itrans_rank = ranked[ranked["display_name"] == "iTransformer"].index
        if len(itrans_rank):
            rank = int(itrans_rank[0]) + 1
            ax.set_title(f"{h}h horizon\niTransformer rank: {rank} of {len(sub)}",
                         fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"{h}h horizon", fontsize=10, fontweight="bold")

        ax.set_xlabel("Improvement vs ARIMA (%)", fontsize=9)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.suptitle(
        "W2 — Improvement Over ARIMA Baseline\n"
        "(positive = lower RMSE than ARIMA load-only baseline; "
        "note: other models use past weather covariates while ARIMA does not)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_02_improvement_over_arima.png")


# ---------------------------------------------------------------------------
# W3 — XGBoost feature importance (weather vs calendar/lag)
# ---------------------------------------------------------------------------

def figure_w3_feature_importance() -> None:
    # Prefer SHAP if available; fall back to built-in XGBoost importance.
    if SHAP_CSV.exists():
        imp = pd.read_csv(SHAP_CSV)
        value_col = "mean_abs_shap"
        title_suffix = "(mean |SHAP|, 24h horizon)"
    elif IMPORTANCE_CSV.exists():
        imp = pd.read_csv(IMPORTANCE_CSV)
        value_col = "importance"
        title_suffix = "(XGBoost gain importance, all horizons)"
    else:
        print("  [skip W3] neither SHAP CSV nor importance CSV found.")
        return

    # Aggregate across clients / horizons
    summary = (
        imp.groupby("feature")[value_col]
        .mean()
        .sort_values(ascending=False)
        .head(25)
        .sort_values()
    )

    # Classify features
    import re
    WEATHER_KEYWORDS = {"temp", "humidity", "windspeed", "precipitation", "pressure", "dewpoint",
                        "cloud", "solar", "radiation", "wind", "rain", "snow"}
    CALENDAR_KEYWORDS = {"hour", "day_of_week", "month", "weekday", "weekend", "day_of_year",
                         "week_of_year", "is_holiday"}

    def _classify(feat: str) -> str:
        fl = feat.lower()
        # Lag features: e.g. "MT_001_lag_24"
        if re.search(r"_lag_\d+$", fl):
            return "Load lag"
        for kw in WEATHER_KEYWORDS:
            if kw in fl:
                return "Weather"
        for kw in CALENDAR_KEYWORDS:
            if kw in fl:
                return "Calendar"
        return "Other"

    group_colours = {
        "Weather":   "#d62728",
        "Calendar":  "#ff7f0e",
        "Load lag":  "#1f77b4",
        "Other":     "#7f7f7f",
    }

    colours = [group_colours[_classify(f)] for f in summary.index]

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(summary.index, summary.values, color=colours, alpha=0.85)
    ax.set_xlabel(f"Mean {value_col.replace('_', ' ').title()}", fontsize=11)
    ax.set_title(
        f"W3 — XGBoost Feature Importance {title_suffix}\n(top 25 features)",
        fontsize=12, fontweight="bold",
    )

    # Legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=g) for g, c in group_colours.items()
                      if g in [_classify(f) for f in summary.index]]
    ax.legend(handles=legend_handles, title="Feature group", fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, "fig_03_xgboost_feature_importance.png")


# ---------------------------------------------------------------------------
# W4 — Weather × Load cross-feature correlation heatmap
# ---------------------------------------------------------------------------

def figure_w4_correlation_heatmap() -> None:
    from common.common import WEATHER_COLUMNS, CALENDAR_COLUMNS, DATA_PATH, RANDOM_SEED

    if not DATA_PATH.exists():
        print(f"  [skip W4] data file not found: {DATA_PATH}")
        return

    print("  Loading data for correlation heatmap...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    exog_cols = WEATHER_COLUMNS + CALENDAR_COLUMNS
    load_cols = [c for c in df.columns if c not in exog_cols and c != "date"]

    # Sample 30 load clients for readability; use a fixed seed for reproducibility.
    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(30, len(load_cols))
    sampled_load = sorted(rng.choice(load_cols, size=n_sample, replace=False).tolist())

    # Pearson correlation: rows = exog features, columns = sampled load clients
    sub = df[exog_cols + sampled_load].dropna()
    corr = sub[exog_cols].corrwith(sub[sampled_load[0]]).copy()   # placeholder shape
    corr_matrix = np.zeros((len(exog_cols), len(sampled_load)))
    for j, lc in enumerate(sampled_load):
        for i, ec in enumerate(exog_cols):
            corr_matrix[i, j] = sub[ec].corr(sub[lc])

    # Mean absolute correlation per exog feature (for right-side summary bar)
    mean_abs_corr = np.abs(corr_matrix).mean(axis=1)

    # Sort exog features by mean |r| descending
    order = np.argsort(mean_abs_corr)[::-1]
    corr_matrix = corr_matrix[order, :]
    exog_labels = [exog_cols[i] for i in order]
    mean_abs_corr = mean_abs_corr[order]

    # Short display names for exog features
    def _short(name: str) -> str:
        replacements = {
            "temperature_2m": "Temp (2m)",
            "relative_humidity_2m": "Humidity",
            "wind_speed_10m": "Wind speed",
            "precipitation": "Precip.",
            "surface_pressure": "Pressure",
            "cloud_cover": "Cloud cover",
            "shortwave_radiation": "SW Radiation",
            "hour": "Hour",
            "day_of_week": "Day of week",
            "month": "Month",
            "is_weekend": "Is weekend",
            "day_of_year": "Day of year",
            "week_of_year": "Week of year",
        }
        return replacements.get(name, name)

    exog_display = [_short(e) for e in exog_labels]

    # Short client labels (strip common prefix)
    def _short_client(name: str) -> str:
        return name.replace("MT_", "")

    client_display = [_short_client(c) for c in sampled_load]

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [5, 1]},
    )

    # --- Heatmap ---
    im = ax_heat.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(sampled_load)))
    ax_heat.set_xticklabels(client_display, rotation=90, fontsize=7)
    ax_heat.set_yticks(range(len(exog_labels)))
    ax_heat.set_yticklabels(exog_display, fontsize=9)
    ax_heat.set_xlabel(f"Load client (n={n_sample} sampled)", fontsize=10)
    ax_heat.set_ylabel("Exogenous feature", fontsize=10)
    plt.colorbar(im, ax=ax_heat, label="Pearson r", shrink=0.85)

    # Divider line between weather and calendar features
    n_weather = len(WEATHER_COLUMNS)
    n_calendar = len(CALENDAR_COLUMNS)
    # Find where calendar block starts in sorted order
    weather_set = set(WEATHER_COLUMNS)
    first_calendar_idx = next(
        (i for i, e in enumerate(exog_labels) if e not in weather_set), None
    )
    if first_calendar_idx is not None and 0 < first_calendar_idx < len(exog_labels):
        ax_heat.axhline(first_calendar_idx - 0.5, color="black", linewidth=1.2, linestyle="--")
        ax_heat.text(
            len(sampled_load) - 0.3, first_calendar_idx - 0.7,
            "── Weather above / Calendar below ──",
            fontsize=7, color="black", ha="right", va="bottom",
        )

    # --- Summary bar: mean |r| per feature ---
    colours_bar = ["#d62728" if e in weather_set else "#ff7f0e" for e in exog_labels]
    ax_bar.barh(range(len(exog_labels)), mean_abs_corr, color=colours_bar, alpha=0.85)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean |r|", fontsize=9)
    ax_bar.set_xlim(0, max(mean_abs_corr) * 1.25)
    ax_bar.grid(axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(mean_abs_corr):
        ax_bar.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            Patch(color="#d62728", label="Weather"),
            Patch(color="#ff7f0e", label="Calendar"),
        ],
        fontsize=8, loc="lower right",
    )

    fig.suptitle(
        "W4 — Pearson Correlation: Exogenous Features vs Load Clients\n"
        "Red = positive, Blue = negative; sorted by mean |r| across sampled clients",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_04_feature_load_correlation.png")


# ---------------------------------------------------------------------------
# Helpers for A / B / C
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    """Per-client (non-aggregated) results from all model CSVs."""
    frames = []
    for key, path in RESULT_FILES.items():
        if not path.exists():
            print(f"  [skip] missing {path.name}")
            continue
        df = pd.read_csv(path)
        df["display_name"] = _display(key)
        df["client_id"] = df["client_id"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No result CSVs found.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# A — Per-client RMSE box plot
# ---------------------------------------------------------------------------

def figure_a_rmse_boxplot(all_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, h in zip(axes, HORIZONS):
        sub = all_results[all_results["horizon"] == h]
        data, labels, colours = [], [], []
        for m in DISPLAY_ORDER:
            vals = sub[sub["display_name"] == m]["RMSE"].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(m)
                colours.append(COLOR_MAP[m])

        bp = ax.boxplot(
            data, tick_labels=labels, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_title(f"{h}h horizon", fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("RMSE (kWh)", fontsize=10)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(
        "A — Per-Client RMSE Distribution by Model and Horizon\n"
        "(box = IQR; whiskers = 1.5×IQR; dots = outlier clients)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_05_rmse_distribution.png")


# ---------------------------------------------------------------------------
# B — Forecast degradation curve (normalised to 24 h baseline)
# ---------------------------------------------------------------------------

def figure_b_degradation_curve(all_results: pd.DataFrame) -> None:
    rmse_24 = (
        all_results[all_results["horizon"] == 24]
        .set_index(["display_name", "client_id"])["RMSE"]
        .rename("rmse_24")
    )
    merged = all_results.join(rmse_24, on=["display_name", "client_id"])
    merged["rmse_ratio"] = merged["RMSE"] / merged["rmse_24"]

    summary = (
        merged.groupby(["display_name", "horizon"])["rmse_ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for m in DISPLAY_ORDER:
        sub = summary[summary["display_name"] == m].set_index("horizon").reindex(HORIZONS)
        if sub.empty or sub["mean"].isna().all():
            continue
        means = sub["mean"].values
        stds  = sub["std"].fillna(0).values
        colour = COLOR_MAP[m]
        ax.plot(HORIZONS, means, marker="o", color=colour, linewidth=2.2, label=m)
        ax.fill_between(HORIZONS, means - stds, means + stds,
                        color=colour, alpha=0.12)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="24h baseline (ratio = 1)")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(["24h", "48h", "168h"])
    ax.set_xlabel("Forecast horizon", fontsize=11)
    ax.set_ylabel("RMSE ratio relative to 24h", fontsize=11)
    ax.set_title(
        "B — Forecast Degradation: RMSE Relative to 24h Baseline\n"
        "(shaded = ±1 std across clients; steeper rise = faster error accumulation)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, "fig_06_forecast_degradation.png")


# ---------------------------------------------------------------------------
# C — Actual vs predicted time series for one client at 168 h
# ---------------------------------------------------------------------------

def _predict_arima(history: np.ndarray, steps: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    res = StatsARIMA(history, order=(1, 1, 1)).fit()
    return res.forecast(steps=steps)


def _predict_xgboost(history: np.ndarray, steps: int, n_lags: int = 48) -> np.ndarray:
    X, y = [], []
    for i in range(n_lags, len(history)):
        X.append(history[i - n_lags:i])
        y.append(history[i])
    X_arr, y_arr = np.array(X), np.array(y)

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, random_state=42, verbosity=0)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                          learning_rate=0.05, random_state=42)

    model.fit(X_arr, y_arr)
    buf = list(history[-n_lags:])
    preds = []
    for _ in range(steps):
        p = float(model.predict(np.array(buf[-n_lags:]).reshape(1, -1))[0])
        preds.append(p)
        buf.append(p)
    return np.array(preds)


def _predict_lstm(history: np.ndarray, steps: int,
                  seq_len: int = 96, hidden: int = 64,
                  n_epochs: int = 5) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    mu, sigma = history.mean(), history.std() + 1e-8
    h = (history - mu) / sigma

    X_lst, y_lst = [], []
    for i in range(seq_len, len(h)):
        X_lst.append(h[i - seq_len:i])
        y_lst.append(h[i])
    X_t = torch.FloatTensor(X_lst).unsqueeze(-1)
    y_t = torch.FloatTensor(y_lst).unsqueeze(-1)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, 2, batch_first=True, dropout=0.1)
            self.fc   = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = _LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(n_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    buf = list(h[-seq_len:])
    preds_norm = []
    for _ in range(steps):
        x_in = torch.FloatTensor(buf[-seq_len:]).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            p = float(model(x_in).item())
        preds_norm.append(p)
        buf.append(p)
    return np.array(preds_norm) * sigma + mu


def _predict_tabpfn(split, client_id: str, steps: int,
                    context_rows: int = 500) -> np.ndarray:
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode, TimeSeriesDataFrame
    from common.common import series_values

    ctx_data = pd.concat([split.train, split.val]).tail(context_rows).reset_index(drop=True)
    ctx_df = pd.DataFrame({
        "item_id": client_id,
        "timestamp": ctx_data["date"].values,
        "target": series_values(ctx_data, client_id),
    })
    fut_df = pd.DataFrame({
        "item_id": client_id,
        "timestamp": split.test["date"].values[:steps],
        "target": np.nan,
    })
    ctx_tsdf = TimeSeriesDataFrame(ctx_df, id_column="item_id", timestamp_column="timestamp")
    fut_tsdf = TimeSeriesDataFrame(fut_df, id_column="item_id", timestamp_column="timestamp")

    pipe = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
    result = pipe.predict(context_tsdf=ctx_tsdf, future_tsdf=fut_tsdf)
    return result["target"].values[:steps]


def figure_c_actual_vs_predicted(all_results: pd.DataFrame) -> None:
    from common.common import load_and_split, series_values

    # Find a client present in all 5 models' results
    n_models = all_results["display_name"].nunique()
    counts = all_results.groupby("client_id")["display_name"].nunique()
    common_clients = counts[counts == n_models].index.tolist()
    if not common_clients:
        print("  [skip C] no client found in all 5 model result CSVs.")
        return
    client_id = str(common_clients[0])
    print(f"  Client selected: {client_id}")

    split = load_and_split()
    STEPS = 168
    history = np.concatenate([
        series_values(split.train, client_id),
        series_values(split.val,   client_id),
    ])
    actual     = series_values(split.test, client_id)[:STEPS]
    test_dates = pd.to_datetime(split.test["date"].values[:STEPS])

    predictions: dict[str, np.ndarray] = {}

    print("  ARIMA ...", end=" ", flush=True)
    try:
        predictions["ARIMA"] = _predict_arima(history, STEPS)
        print("done")
    except Exception as exc:
        print(f"failed ({exc})")

    print("  XGBoost ...", end=" ", flush=True)
    try:
        predictions["XGBoost"] = _predict_xgboost(history, STEPS)
        print("done")
    except Exception as exc:
        print(f"failed ({exc})")

    print("  LSTM (5 epochs, univariate) ...", end=" ", flush=True)
    try:
        predictions["LSTM"] = _predict_lstm(history, STEPS)
        print("done")
    except Exception as exc:
        print(f"failed ({exc})")

    print("  TabPFN-TS ...", end=" ", flush=True)
    try:
        predictions["TabPFN-TS"] = _predict_tabpfn(split, client_id, STEPS)
        print("done")
    except Exception as exc:
        print(f"failed ({exc})")

    # iTransformer: annotate RMSE from results CSV (multivariate model, no inline rerun)
    itrans_row = all_results[
        (all_results["display_name"] == "iTransformer") &
        (all_results["client_id"] == client_id) &
        (all_results["horizon"] == STEPS)
    ]
    itrans_rmse = float(itrans_row["RMSE"].iloc[0]) if not itrans_row.empty else None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_dates, actual, color="black", linewidth=2.0, label="Actual", zorder=10)
    for m, preds in predictions.items():
        ax.plot(test_dates, preds[:len(test_dates)],
                color=COLOR_MAP[m], linewidth=1.5, alpha=0.82, label=m)

    if itrans_rmse is not None:
        ax.text(
            0.01, 0.97,
            f"iTransformer 168h RMSE = {itrans_rmse:.0f} kWh\n"
            "(predictions not shown — multivariate model requires full re-training)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc=COLOR_MAP["iTransformer"], alpha=0.12,
                      ec=COLOR_MAP["iTransformer"]),
        )

    ax.set_xlabel("Date (first 168h of test period)", fontsize=11)
    ax.set_ylabel("Electricity load (kWh)", fontsize=11)
    ax.set_title(
        f"C — Actual vs Predicted Load: Client {client_id}, 168h Forecast\n"
        "(LSTM: simplified univariate approximation; "
        "XGBoost: autoregressive with lag-48 features)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.2, linestyle="--")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_07_actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results...")
    summary = load_summary()
    print(f"Loaded {len(summary)} (model, horizon) rows.\n")

    print("=== W1: Grouped bar chart ===")
    figure_w1_grouped_bar(summary)

    print("\n=== W2: Improvement % vs ARIMA ===")
    figure_w2_improvement(summary)

    print("\n=== W3: XGBoost feature importance ===")
    figure_w3_feature_importance()

    print("\n=== W4: Feature-load correlation heatmap ===")
    figure_w4_correlation_heatmap()

    print("\n=== A: Per-client RMSE box plot ===")
    all_results = load_all_results()
    figure_a_rmse_boxplot(all_results)

    print("\n=== B: Degradation curve ===")
    figure_b_degradation_curve(all_results)

    print("\n=== C: Actual vs predicted (client inference) ===")
    figure_c_actual_vs_predicted(all_results)

    print("\nDone. All figures saved to:", OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
