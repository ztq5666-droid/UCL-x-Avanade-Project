"""
Step 1b — Weekly seasonal-naive pointwise benchmark.

Forecast rule:  yhat  = y[origin + H - 1 - 168]
Actual target:  actual = y[origin + H - 1]

This is the single-point variant of the seasonal naive, matching how
TabPFN-TS is evaluated: one prediction at exactly step H from the forecast
origin (1-indexed, so step H-1 in 0-based offset). Because there is only
one error per client per horizon, RMSE == MAE == |error|.

The trajectory-based SeasonalNaive (01_seasonal_naive.py) averages error
over steps 1..H; this script computes error only at step H, making the
comparison against TabPFN-TS metric-equivalent.

No-leakage proof: for H in {24, 48, 168}, lookback index = origin+H-1-168.
  H=24:  origin + 23 - 168 = origin - 145  < origin  (safe)
  H=168: origin + 167 - 168 = origin - 1   < origin  (safe)
Assertions below verify this at runtime.

Usage (run from repository root):
    ecl_main_experiment/venv/bin/python3 ecl_main_experiment/analysis/01b_seasonal_naive_pointwise.py
Output:
    ecl_main_experiment/results/raw_metrics/seasonal_naive_pointwise_results.csv
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ECL_PATH    = "ecl_main_experiment/data/ECL/electricity.csv"
PANEL_SRC   = "ecl_main_experiment/results/raw_metrics/arima_results.csv"
TABPFN_PATH = "ecl_main_experiment/results/raw_metrics/tabpfn_results.csv"
OUT_PATH    = "ecl_main_experiment/results/raw_metrics/seasonal_naive_pointwise_results.csv"

CLIENT_COL   = "client_id"
HORIZONS     = [24, 48, 168]
SEASONAL_LAG = 168
TRAIN_FRAC   = 0.7
VAL_FRAC     = 0.1
MODEL_NAME   = "SeasonalNaive-Point"
# ----------------------------------------------------------------------


def load_panel(path, client_col):
    df = pd.read_csv(path)
    if client_col not in df.columns:
        raise KeyError(
            f"Column '{client_col}' not in {path}. Available: {list(df.columns)}"
        )
    return df[client_col].astype(str).unique().tolist()


def main():
    df = pd.read_csv(ECL_PATH)
    date_col = df.columns[0]
    df = df.drop(columns=[date_col])
    df.columns = [str(c) for c in df.columns]

    n = len(df)
    origin = int(n * (TRAIN_FRAC + VAL_FRAC))   # val/test boundary
    print(f"Rows: {n}   Forecast origin (val_end): {origin}")

    clients = load_panel(PANEL_SRC, CLIENT_COL)
    print(f"Evaluation panel: {len(clients)} clients -> {clients}")

    missing = [c for c in clients if c not in df.columns]
    if missing:
        raise KeyError(f"Panel clients absent from ECL columns: {missing}")

    rows = []
    for client in clients:
        series = df[client].to_numpy(dtype=float)

        for H in HORIZONS:
            # 1-indexed: step H from origin → 0-based index = origin + H - 1
            idx_actual   = origin + H - 1
            idx_lookback = idx_actual - SEASONAL_LAG

            assert idx_lookback >= 0, \
                f"lookback negative (client={client}, H={H})"
            assert idx_lookback < origin, \
                f"leakage: lookback at or past forecast origin (client={client}, H={H})"
            assert idx_actual < n, \
                f"horizon {H} exceeds dataset length (client={client})"

            actual = float(series[idx_actual])
            yhat   = float(series[idx_lookback])

            err  = yhat - actual
            rmse = float(np.abs(err))   # single-point: RMSE == MAE == |error|
            mae  = float(np.abs(err))

            rows.append({
                "model":              MODEL_NAME,
                "client_id":          client,
                "horizon":            H,
                "RMSE":               round(rmse, 4),
                "MAE":                round(mae,  4),
                "train_time_sec":     0.0,
                "inference_time_sec": 0.0,
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}   ({len(out)} rows)")

    print("\nSeasonalNaive-Point — mean and median RMSE per horizon:")
    sn_summary = (
        out.groupby("horizon")["RMSE"]
           .agg(mean="mean", median="median")
           .reindex(HORIZONS)
    )
    print(sn_summary.round(2).to_string())

    # --- Comparison against TabPFN-TS ---
    print("\n" + "=" * 65)
    print("TabPFN-TS vs SeasonalNaive-Point")
    print("=" * 65)

    if not os.path.exists(TABPFN_PATH):
        print(f"  TabPFN results not found at {TABPFN_PATH}")
        return

    tab = pd.read_csv(TABPFN_PATH)
    rmse_col = next((c for c in tab.columns if c.upper() == "RMSE"), None)
    if rmse_col is None:
        print(f"  No RMSE column in {TABPFN_PATH}. Columns: {list(tab.columns)}")
        return

    tab_sum = (
        tab.groupby("horizon")[rmse_col]
           .agg(tabpfn_mean="mean", tabpfn_median="median")
           .reindex(HORIZONS)
    )
    sn_sum = (
        out.groupby("horizon")["RMSE"]
           .agg(sn_mean="mean", sn_median="median")
           .reindex(HORIZONS)
    )

    cmp = tab_sum.join(sn_sum)
    cmp["mean_diff_pct"]   = 100.0 * (cmp["tabpfn_mean"]   - cmp["sn_mean"])   / cmp["sn_mean"]
    cmp["median_diff_pct"] = 100.0 * (cmp["tabpfn_median"] - cmp["sn_median"]) / cmp["sn_median"]

    print(f"\n{'H':>4}  {'TabPFN mean':>12}  {'SN-Point mean':>13}  {'diff%':>7}  "
          f"{'TabPFN med':>11}  {'SN-Point med':>12}  {'diff%':>7}")
    print("-" * 75)
    for H in HORIZONS:
        r = cmp.loc[H]
        print(f"{H:>4}  {r['tabpfn_mean']:>12.1f}  {r['sn_mean']:>13.1f}  "
              f"{r['mean_diff_pct']:>7.1f}%  "
              f"{r['tabpfn_median']:>11.1f}  {r['sn_median']:>12.1f}  "
              f"{r['median_diff_pct']:>7.1f}%")

    print("\ndiff% < 0  →  TabPFN-TS has lower RMSE than SeasonalNaive-Point")
    print("diff% > 0  →  SeasonalNaive-Point has lower RMSE than TabPFN-TS")


if __name__ == "__main__":
    main()
