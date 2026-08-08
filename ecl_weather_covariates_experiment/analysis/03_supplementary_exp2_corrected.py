"""
Experiment 2 (CORRECTED, 2016-2019 aligned) — Supplementary analyses.

Tasks:
  1. Distribution table by (horizon, model): mean, median, q25, q75, iqr,
     nmae_mean, nmae_median. Six pipelines sorted by horizon → nmae_mean.
  2. Rank comparison per horizon: nmae_mean rank vs mean-RMSE rank.
  3. Paired bootstrap (10 000 iterations, seed 42, 2.5/97.5 pct).
     Special callout: iTransformer / LSTM / TabPFN each vs SeasonalNaive at 168h.
  4. Exp 1 vs Exp 2 (corrected) mean RMSE side-by-side for all six pipelines.

Outputs saved to:
  ecl_weather_covariates_experiment/outputs/raw_metrics/
      distribution_summary_exp2_corrected.csv
      bootstrap_comparisons_exp2_corrected.csv

Run from the repository root:
    /opt/anaconda3/bin/python3 \
        ecl_weather_covariates_experiment/analysis/03_supplementary_exp2_corrected.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ECL_PATH      = "ecl_main_experiment/data/ECL/electricity.csv"
EXP2_DIR      = "ecl_weather_covariates_experiment/outputs/raw_metrics"
EXP1_DIR      = "ecl_main_experiment/results/raw_metrics"
OUT_DIR       = EXP2_DIR

EXP2_FILES = {
    "ARIMA":        "arima_load_only_results.csv",
    "XGBoost":      "xgboost_exog_past_weather_results.csv",
    "LSTM":         "lstm_exog_results.csv",
    "iTransformer": "itransformer_exog_results.csv",
    "TabPFN-TS":    "tabpfn_exog_past_weather_results.csv",
}

EXP1_FILES = {
    "ARIMA":         "arima_results.csv",
    "XGBoost":       "xgboost_results.csv",
    "LSTM":          "lstm_results.csv",
    "iTransformer":  "itransformer_results.csv",
    "TabPFN-TS":     "tabpfn_results.csv",
    "SeasonalNaive": "seasonal_naive_results.csv",
}

EXP1_NAIVE_PATH = os.path.join(EXP1_DIR, "seasonal_naive_results.csv")

HORIZONS = [24, 48, 168]
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.1
N_BOOT     = 10_000
SEED       = 42

COMPARISONS = [
    ("iTransformer", "LSTM"),
    ("iTransformer", "SeasonalNaive"),
    ("LSTM",         "SeasonalNaive"),
    ("TabPFN-TS",    "SeasonalNaive"),
    ("TabPFN-TS",    "iTransformer"),
    ("XGBoost",      "SeasonalNaive"),
]

CALLOUT_168 = [
    ("iTransformer", "SeasonalNaive"),
    ("LSTM",         "SeasonalNaive"),
    ("TabPFN-TS",    "SeasonalNaive"),
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def client_scale_table():
    """Mean |load| per client over each evaluation window."""
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
            rows.append({"client": col, "horizon": H,
                         "mean_actual": float(np.mean(np.abs(window)))})
    return pd.DataFrame(rows)


def normalise_cols(df, client_col="client_id", rmse_col="RMSE", mae_col="MAE"):
    """Rename heterogeneous column names to internal standard."""
    rename = {}
    if client_col in df.columns:
        rename[client_col] = "client"
    if rmse_col in df.columns and rmse_col != "rmse":
        rename[rmse_col] = "rmse"
    if mae_col in df.columns and mae_col != "mae":
        rename[mae_col] = "mae"
    return df.rename(columns=rename)


def load_exp2():
    frames = []
    for model, fname in EXP2_FILES.items():
        path = os.path.join(EXP2_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {fname}")
            continue
        r = normalise_cols(pd.read_csv(path))
        r["client"] = r["client"].astype(str)
        r["model"]  = model
        frames.append(r[["model", "client", "horizon", "rmse", "mae"]])
        print(f"  loaded Exp2 {model:<15} rows={len(r)}")

    # SeasonalNaive from Exp1 (load-only, unchanged)
    r = pd.read_csv(EXP1_NAIVE_PATH)
    if "client_id" in r.columns:
        r = r.rename(columns={"client_id": "client"})
    r["client"] = r["client"].astype(str)
    r["model"]  = "SeasonalNaive"
    frames.append(r[["model", "client", "horizon", "rmse", "mae"]])
    print(f"  loaded Exp1 {'SeasonalNaive':<15} rows={len(r)}")

    return pd.concat(frames, ignore_index=True)


def load_exp1_means():
    """Return {model: {horizon: mean_rmse}} from Exp1 CSVs for Task 4."""
    result = {}
    for model, fname in EXP1_FILES.items():
        path = os.path.join(EXP1_DIR, fname)
        if not os.path.exists(path):
            continue
        r = normalise_cols(pd.read_csv(path))
        r["client"] = r["client"].astype(str)
        by_h = {}
        for H in HORIZONS:
            vals = r[r["horizon"] == H]["rmse"]
            by_h[H] = float(vals.mean()) if len(vals) > 0 else None
        result[model] = by_h
    return result


def paired_bootstrap(a_vals, b_vals):
    rng  = np.random.default_rng(SEED)
    diff = a_vals - b_vals
    k    = len(diff)
    idx  = rng.integers(0, k, size=(N_BOOT, k))
    boot_means = diff[idx].mean(axis=1)
    return {
        "mean_diff":            float(diff.mean()),
        "median_diff":          float(np.median(diff)),
        "ci_low":               float(np.percentile(boot_means, 2.5)),
        "ci_high":              float(np.percentile(boot_means, 97.5)),
        "pct_clients_a_better": float((diff < 0).mean() * 100),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading Exp2 corrected results...")
    res   = load_exp2()
    scale = client_scale_table()
    res   = res.merge(scale, on=["client", "horizon"], how="left")
    res["nmae"]  = res["mae"]  / res["mean_actual"]
    res["nrmse"] = res["rmse"] / res["mean_actual"]

    # ────────────────────────────────────────────
    # TASK 1 — Distribution summary
    # ────────────────────────────────────────────
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
    dist_sorted = dist.sort_values(["horizon", "nmae_mean"]).reset_index(drop=True)

    dist_sorted.round(4).to_csv(
        os.path.join(OUT_DIR, "distribution_summary_exp2_corrected.csv"), index=False)

    print("\n" + "=" * 100)
    print("TASK 1 — DISTRIBUTION SUMMARY (Exp2 corrected, sorted by horizon → nmae_mean)")
    print("=" * 100)
    print(dist_sorted.round(4).to_string(index=False))

    # ────────────────────────────────────────────
    # TASK 2 — Rank comparison
    # ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TASK 2 — RANK COMPARISON: nmae_mean rank vs mean-RMSE rank (lower = better)")
    print("=" * 100)
    for H in HORIZONS:
        sub = dist[dist.horizon == H].copy()
        sub["nmae_rank"] = sub["nmae_mean"].rank().astype(int)
        sub["rmse_rank"] = sub["mean"].rank().astype(int)
        sub = sub.sort_values("nmae_rank")
        print(f"\n  Horizon {H}h")
        print(f"  {'model':<16} {'nmae_rank':>9} {'nmae_mean':>10} {'rmse_rank':>10} {'mean RMSE':>10}")
        print(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*10} {'-'*10}")
        for _, row in sub.iterrows():
            print(f"  {row['model']:<16} {int(row['nmae_rank']):>9} "
                  f"{row['nmae_mean']:>10.4f} {int(row['rmse_rank']):>10} "
                  f"{row['mean']:>10.1f}")

    # ────────────────────────────────────────────
    # TASK 3 — Paired bootstrap
    # ────────────────────────────────────────────
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
            crosses = stats["ci_low"] < 0 < stats["ci_high"]
            boot_rows.append({
                "model_a":              a,
                "model_b":              b,
                "horizon":              H,
                "n_clients":            len(common),
                "mean_diff":            round(stats["mean_diff"], 2),
                "median_diff":          round(stats["median_diff"], 2),
                "rel_diff_pct":         round(100 * stats["mean_diff"] / bv.mean(), 2),
                "ci_low":               round(stats["ci_low"], 2),
                "ci_high":              round(stats["ci_high"], 2),
                "ci_crosses_zero":      crosses,
                "pct_clients_a_better": round(stats["pct_clients_a_better"], 1),
                "wilcoxon_p":           round(pval, 4) if not np.isnan(pval) else np.nan,
            })

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(os.path.join(OUT_DIR, "bootstrap_comparisons_exp2_corrected.csv"), index=False)

    print("\n" + "=" * 120)
    print("TASK 3 — PAIRED BOOTSTRAP (mean RMSE diff = a − b; 10 000 iter; 95% CI)")
    print("=" * 120)
    hdr = (f"  {'model_a':>14}  {'model_b':<14}  {'H':>4}  {'n':>3}  "
           f"{'mean_diff':>10}  {'[ci_low':>10}  {'ci_high]':>10}  "
           f"{'rel%':>7}  {'CI∩0':>6}  {'a_better%':>10}  {'wilcoxon_p':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in boot.iterrows():
        flag = "YES" if r["ci_crosses_zero"] else "no"
        print(f"  {r['model_a']:>14}  {r['model_b']:<14}  {int(r['horizon']):>4}  "
              f"{int(r['n_clients']):>3}  "
              f"{r['mean_diff']:>10.1f}  {r['ci_low']:>10.1f}  {r['ci_high']:>10.1f}  "
              f"{r['rel_diff_pct']:>7.1f}%  {flag:>6}  "
              f"{r['pct_clients_a_better']:>10.1f}%  {r['wilcoxon_p']:>11.4f}")

    # ── Callout: three models vs SeasonalNaive at 168h ──────────────────────
    print("\n" + "=" * 100)
    print("CALLOUT — iTransformer / LSTM / TabPFN-TS vs SeasonalNaive at 168h (corrected)")
    print("(test set = winter-spring 2018-11 to 2019-07; old weather was phase-inverted here)")
    print("=" * 100)

    sn_168 = res[(res.model == "SeasonalNaive") & (res.horizon == 168)].set_index("client")["rmse"]
    sn_mean = sn_168.mean()
    print(f"\n  SeasonalNaive 168h mean RMSE: {sn_mean:.1f}\n")

    for model_a, model_b in CALLOUT_168:
        row = boot[(boot.model_a == model_a) & (boot.model_b == model_b) & (boot.horizon == 168)]
        if row.empty:
            print(f"  {model_a} vs {model_b}: not in bootstrap results.")
            continue
        r = row.iloc[0]
        verdict = ("CI CROSSES ZERO — not robustly distinguishable from naive"
                   if r["ci_crosses_zero"]
                   else "CI EXCLUDES ZERO — model robustly beats naive")
        model_rmse = res[(res.model == model_a) & (res.horizon == 168)]["rmse"].mean()
        print(f"  {model_a} vs SeasonalNaive")
        print(f"    {model_a} mean RMSE : {model_rmse:.1f}")
        print(f"    mean_diff          : {r['mean_diff']:.1f}  ({model_a} − SeasonalNaive)")
        print(f"    95% CI             : [{r['ci_low']:.1f}, {r['ci_high']:.1f}]")
        print(f"    ci_crosses_zero    : {r['ci_crosses_zero']}  → {verdict}")
        print(f"    pct_clients_better : {r['pct_clients_a_better']:.1f}%  "
              f"({model_a} beats naive on this share of clients)")
        print(f"    wilcoxon_p         : {r['wilcoxon_p']:.4f}")
        print()

    # ────────────────────────────────────────────
    # TASK 4 — Exp1 vs Exp2 corrected side-by-side
    # ────────────────────────────────────────────
    print("=" * 100)
    print("TASK 4 — EXP1 (load-only) vs EXP2 CORRECTED (weather-augmented) mean RMSE")
    print("=" * 100)

    exp1_means = load_exp1_means()
    exp2_means = {}
    for model in list(EXP2_FILES.keys()) + ["SeasonalNaive"]:
        by_h = {}
        for H in HORIZONS:
            vals = res[(res.model == model) & (res.horizon == H)]["rmse"]
            by_h[H] = float(vals.mean()) if len(vals) > 0 else None
        exp2_means[model] = by_h

    for H in HORIZONS:
        print(f"\n  Horizon {H}h")
        print(f"  {'model':<16} {'Exp1 RMSE':>12} {'Exp2 RMSE':>12} {'Δ RMSE':>10} {'Δ%':>8}")
        print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
        all_models = list(EXP2_FILES.keys()) + ["SeasonalNaive"]
        for model in all_models:
            e1 = exp1_means.get(model, {}).get(H)
            e2 = exp2_means.get(model, {}).get(H)
            if e1 is None and e2 is None:
                continue
            e1_str = f"{e1:12.1f}" if e1 is not None else f"{'—':>12}"
            e2_str = f"{e2:12.1f}" if e2 is not None else f"{'—':>12}"
            if e1 is not None and e2 is not None:
                delta    = e2 - e1
                delta_pct = 100 * delta / e1
                d_str = f"{delta:+10.1f}"
                p_str = f"{delta_pct:+8.1f}%"
            else:
                d_str, p_str = f"{'—':>10}", f"{'—':>8}"
            print(f"  {model:<16} {e1_str} {e2_str} {d_str} {p_str}")

    print(f"\nSaved: {os.path.join(OUT_DIR, 'distribution_summary_exp2_corrected.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'bootstrap_comparisons_exp2_corrected.csv')}")


if __name__ == "__main__":
    main()
