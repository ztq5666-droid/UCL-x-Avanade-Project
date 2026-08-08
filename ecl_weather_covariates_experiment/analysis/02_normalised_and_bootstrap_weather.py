"""
Experiment 2 — Normalised error metrics + paired bootstrap CIs (weather covariates).

Mirrors files/02_normalised_and_bootstrap.py (Experiment 1) but points at the
Experiment 2 raw_metrics directory. SeasonalNaive is imported from Experiment 1:
the naive benchmark uses load only and is therefore unchanged by the covariate
extension — it acts as a fixed anchor across both experiments.

Scale denominators (mean absolute load per client per window) are always derived
from the ECL load series, not from the weather file, because normalisation is by
client load.

Usage (run from repository root):
    ecl_main_experiment/venv/bin/python3 \
        ecl_weather_covariates_experiment/analysis/02_normalised_and_bootstrap_weather.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ECL_PATH = "ecl_main_experiment/data/ECL/electricity.csv"

# Experiment 2 result files
EXP2_DIR = "ecl_weather_covariates_experiment/outputs/raw_metrics"
EXP2_FILES = {
    "ARIMA":        "arima_load_only_results.csv",
    "LSTM":         "lstm_exog_results.csv",
    "iTransformer": "itransformer_exog_results.csv",
    "TabPFN-TS":    "tabpfn_exog_past_weather_results.csv",
    "XGBoost":      "xgboost_exog_past_weather_results.csv",
}

# SeasonalNaive pulled from Experiment 1 (unchanged by covariate extension)
EXP1_NAIVE_PATH = "ecl_main_experiment/results/raw_metrics/seasonal_naive_results.csv"

HORIZONS   = [24, 48, 168]
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.1
N_BOOT     = 10_000
SEED       = 42

COMPARISONS = [
    ("iTransformer", "LSTM"),
    ("TabPFN-TS",    "iTransformer"),
    ("TabPFN-TS",    "LSTM"),
    ("iTransformer", "SeasonalNaive"),
    ("LSTM",         "SeasonalNaive"),
    ("XGBoost",      "SeasonalNaive"),
    ("ARIMA",        "SeasonalNaive"),
    ("TabPFN-TS",    "SeasonalNaive"),
    ("XGBoost",      "ARIMA"),
]
# ----------------------------------------------------------------------


def client_scale_table():
    """Mean absolute load per client over each evaluation window (ECL load series)."""
    df = pd.read_csv(ECL_PATH)
    df = df.drop(columns=[df.columns[0]])
    df.columns = [str(c) for c in df.columns]
    n      = len(df)
    origin = int(n * (TRAIN_FRAC + VAL_FRAC))

    rows = []
    for col in df.columns:
        series = df[col].to_numpy(dtype=float)
        for H in HORIZONS:
            window = series[origin: origin + H]
            rows.append({
                "client":      col,
                "horizon":     H,
                "mean_actual": float(np.mean(np.abs(window))),
            })
    return pd.DataFrame(rows)


def load_all_results():
    frames = []

    # --- Experiment 2 models ---
    for model, fname in EXP2_FILES.items():
        path = os.path.join(EXP2_DIR, fname)
        if not os.path.exists(path):
            print(f"  skipped (not found): {fname}")
            continue
        r = pd.read_csv(path)
        # Normalise to internal column names
        r = r.rename(columns={
            "client_id": "client",
            "RMSE":      "rmse",
            "MAE":       "mae",
        })
        r["client"] = r["client"].astype(str)
        r["model"]  = model
        keep = ["model", "client", "horizon", "rmse", "mae"]
        frames.append(r[[c for c in keep if c in r.columns]])
        print(f"  loaded (Exp2): {model:15s} rows={len(r)}")

    # --- SeasonalNaive from Experiment 1 ---
    if not os.path.exists(EXP1_NAIVE_PATH):
        print(f"  skipped (not found): {EXP1_NAIVE_PATH}")
    else:
        r = pd.read_csv(EXP1_NAIVE_PATH)
        # Exp1 naive uses lowercase column names; no rename needed beyond model
        if "client_id" in r.columns:
            r = r.rename(columns={"client_id": "client"})
        r["client"] = r["client"].astype(str)
        r["model"]  = "SeasonalNaive"
        keep = ["model", "client", "horizon", "rmse", "mae"]
        frames.append(r[[c for c in keep if c in r.columns]])
        print(f"  loaded (Exp1): {'SeasonalNaive':15s} rows={len(r)}")

    if not frames:
        raise SystemExit("No result files found.")
    return pd.concat(frames, ignore_index=True)


def paired_bootstrap(a_vals, b_vals, n_boot=N_BOOT, seed=SEED):
    rng  = np.random.default_rng(seed)
    diff = a_vals - b_vals
    k    = len(diff)
    idx  = rng.integers(0, k, size=(n_boot, k))
    boot_means = diff[idx].mean(axis=1)
    return {
        "mean_diff":           float(diff.mean()),
        "median_diff":         float(np.median(diff)),
        "ci_low":              float(np.percentile(boot_means, 2.5)),
        "ci_high":             float(np.percentile(boot_means, 97.5)),
        "pct_clients_a_better": float((diff < 0).mean() * 100),
    }


def main():
    print("Loading results...")
    res = load_all_results()

    print("\nComputing client-level scale denominators (ECL load)...")
    scale = client_scale_table()
    res   = res.merge(scale, on=["client", "horizon"], how="left")

    res["nmae"]  = res["mae"]  / res["mean_actual"]
    res["nrmse"] = res["rmse"] / res["mean_actual"]

    # ----------------------------------------------------------------
    # 1. Distribution summary
    # ----------------------------------------------------------------
    dist = (
        res.groupby(["model", "horizon"])["rmse"]
           .agg(mean="mean",
                median="median",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75))
           .reset_index()
    )
    dist["iqr"] = dist["q75"] - dist["q25"]
    dist_n = (
        res.groupby(["model", "horizon"])["nmae"]
           .agg(nmae_mean="mean", nmae_median="median")
           .reset_index()
    )
    dist = dist.merge(dist_n, on=["model", "horizon"])

    # ----------------------------------------------------------------
    # 2. Bootstrap comparisons
    # ----------------------------------------------------------------
    boot_rows = []
    for a, b in COMPARISONS:
        for H in HORIZONS:
            sa = res[(res.model == a) & (res.horizon == H)].set_index("client")["rmse"]
            sb = res[(res.model == b) & (res.horizon == H)].set_index("client")["rmse"]
            common = sa.index.intersection(sb.index)
            if len(common) < 5:
                continue
            av, bv = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()

            stats = paired_bootstrap(av, bv)
            try:
                _, pval = wilcoxon(av, bv)
            except ValueError:
                pval = np.nan

            rel           = 100.0 * stats["mean_diff"] / bv.mean()
            crosses_zero  = stats["ci_low"] < 0 < stats["ci_high"]

            boot_rows.append({
                "model_a":             a,
                "model_b":             b,
                "horizon":             H,
                "n_clients":           len(common),
                "mean_diff":           stats["mean_diff"],
                "median_diff":         stats["median_diff"],
                "rel_diff_pct":        rel,
                "ci_low":              stats["ci_low"],
                "ci_high":             stats["ci_high"],
                "ci_crosses_zero":     crosses_zero,
                "pct_clients_a_better": stats["pct_clients_a_better"],
                "wilcoxon_p":          pval,
            })

    boot = pd.DataFrame(boot_rows)

    # ================================================================
    # PRINT RESULTS
    # ================================================================

    # --- Table 1: distribution sorted by horizon then nmae_mean -----
    print("\n" + "=" * 90)
    print("TABLE 1 — DISTRIBUTION SUMMARY (Experiment 2, sorted by horizon → nmae_mean)")
    print("=" * 90)
    dist_sorted = dist.sort_values(["horizon", "nmae_mean"])
    print(dist_sorted.round(4).to_string(index=False))

    # --- Table 2: rank comparison -----------------------------------
    print("\n" + "=" * 90)
    print("TABLE 2 — RANK COMPARISON: nmae_mean rank vs mean RMSE rank (lower = better)")
    print("=" * 90)
    for H in HORIZONS:
        sub = dist[dist.horizon == H].copy()
        sub = sub.sort_values("nmae_mean").reset_index(drop=True)
        sub["nmae_rank"] = range(1, len(sub) + 1)
        sub = sub.sort_values("mean").reset_index(drop=True)
        sub["rmse_rank"] = range(1, len(sub) + 1)
        sub = sub.sort_values("nmae_rank")
        print(f"\n  Horizon {H}h")
        print(f"  {'model':<16} {'nmae_rank':>9} {'nmae_mean':>10} {'rmse_rank':>10} {'mean RMSE':>10}")
        print(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*10} {'-'*10}")
        for _, row in sub.iterrows():
            print(f"  {row['model']:<16} {int(row['nmae_rank']):>9} "
                  f"{row['nmae_mean']:>10.4f} {int(row['rmse_rank']):>10} "
                  f"{row['mean']:>10.1f}")

    # --- Table 3: bootstrap -----------------------------------------
    print("\n" + "=" * 110)
    print("TABLE 3 — PAIRED BOOTSTRAP (mean RMSE diff = a − b; 10 000 iterations; 95% CI)")
    print("=" * 110)
    print(f"  {'model_a':>14}  {'model_b':<14}  {'H':>4}  {'n':>3}  "
          f"{'mean_diff':>10}  {'[ci_low':>10}  {'ci_high]':>10}  "
          f"{'rel%':>7}  {'CI∩0':>6}  {'a_better%':>10}  {'wilcoxon_p':>11}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*4}  {'-'*3}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*7}  {'-'*6}  {'-'*10}  {'-'*11}")
    for _, r in boot.iterrows():
        flag = "YES" if r["ci_crosses_zero"] else "no"
        print(f"  {r['model_a']:>14}  {r['model_b']:<14}  {int(r['horizon']):>4}  "
              f"{int(r['n_clients']):>3}  "
              f"{r['mean_diff']:>10.1f}  {r['ci_low']:>10.1f}  {r['ci_high']:>10.1f}  "
              f"{r['rel_diff_pct']:>7.1f}%  {flag:>6}  "
              f"{r['pct_clients_a_better']:>10.1f}%  {r['wilcoxon_p']:>11.4f}")

    # --- iTransformer vs SeasonalNaive at 168h callout --------------
    print("\n" + "=" * 90)
    print("CALLOUT — iTransformer vs SeasonalNaive at 168h")
    print("=" * 90)
    row = boot[(boot.model_a == "iTransformer") &
               (boot.model_b == "SeasonalNaive") &
               (boot.horizon == 168)]
    if row.empty:
        print("  Not found in bootstrap results.")
    else:
        r = row.iloc[0]
        ci_verdict = "CI CROSSES ZERO — not distinguishable from zero" \
                     if r["ci_crosses_zero"] else "CI EXCLUDES ZERO — statistically significant"
        print(f"  mean_diff        : {r['mean_diff']:.1f} kWh  (iTransformer − SeasonalNaive)")
        print(f"  95% CI           : [{r['ci_low']:.1f}, {r['ci_high']:.1f}]")
        print(f"  rel_diff_pct     : {r['rel_diff_pct']:.1f}%")
        print(f"  CI verdict       : {ci_verdict}")
        print(f"  pct_clients_a_better: {r['pct_clients_a_better']:.1f}%  "
              f"(iTransformer lower RMSE on this fraction of clients)")
        print(f"  wilcoxon_p       : {r['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
