"""
Create comparison, business, and explainability figures from saved model results.

This script is intentionally read-only with respect to model outputs: it loads the
existing *_results.csv files and writes derived figures/tables into classified
subfolders under results/figures/.

Output folders:
  - performance_comparison: model accuracy across forecast horizons
  - cost_efficiency: accuracy-cost trade-off figures
  - robustness_checks: sensitivity and robustness figures
  - business_decision_support: client-facing model selection figures
  - interpretability: RQ3/interpretability summary figures with English annotations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import textwrap

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt


RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw_metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

PERFORMANCE_DIR = FIGURES_DIR / "performance_comparison"
COST_EFFICIENCY_DIR = FIGURES_DIR / "cost_efficiency"
ROBUSTNESS_DIR = FIGURES_DIR / "robustness_checks"
BUSINESS_DIR = FIGURES_DIR / "business_decision_support"
INTERPRETABILITY_DIR = FIGURES_DIR / "interpretability"
SUMMARY_DIR = RESULTS_DIR / "summary_tables"

HORIZONS = [24, 48, 168]
CLIENT_313 = "313"


@dataclass(frozen=True)
class ModelStyle:
    label: str
    short_label: str
    color: str
    marker: str
    linestyle: str
    category: str
    complexity: int


MODEL_STYLES = {
    "arima": ModelStyle(
        "ARIMA",
        "ARIMA",
        "#4C78A8",
        "o",
        "-",
        "Statistical baseline",
        1,
    ),
    "xgboost": ModelStyle(
        "XGBoost",
        "XGB",
        "#F58518",
        "s",
        "-",
        "Traditional ML baseline",
        2,
    ),
    "lstm": ModelStyle(
        "LSTM",
        "LSTM",
        "#54A24B",
        "^",
        "-",
        "Deep learning baseline",
        4,
    ),
    "itransformer": ModelStyle(
        "iTransformer",
        "iTrans.",
        "#B279A2",
        "D",
        "-",
        "Transformer model",
        5,
    ),
    "tabpfn": ModelStyle(
        "TabPFN-TS",
        "TabPFN",
        "#E45756",
        "X",
        "--",
        "Exploratory zero-shot foundation baseline",
        3,
    ),
}


def setup() -> None:
    for path in [
        PERFORMANCE_DIR,
        COST_EFFICIENCY_DIR,
        ROBUSTNESS_DIR,
        BUSINESS_DIR,
        INTERPRETABILITY_DIR,
        SUMMARY_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def load_results() -> pd.DataFrame:
    frames = []
    for model in MODEL_STYLES:
        path = RAW_RESULTS_DIR / f"{model}_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        df = pd.read_csv(path)
        df["model"] = model
        df["model_label"] = MODEL_STYLES[model].label
        df["model_short_label"] = MODEL_STYLES[model].short_label
        df["model_category"] = MODEL_STYLES[model].category
        df["client_id"] = df["client_id"].astype(str)
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    results["horizon"] = results["horizon"].astype(int)
    return results


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["model", "model_label", "horizon"], as_index=False)
        .agg(
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            train_time_sec=("train_time_sec", "mean"),
            inference_time_sec=("inference_time_sec", "mean"),
        )
        .sort_values(["horizon", "RMSE"])
    )
    summary.to_csv(SUMMARY_DIR / "model_horizon_summary.csv", index=False)
    return summary


def total_training_time(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in results.groupby("model"):
        style = MODEL_STYLES[model]
        per_client_mean = group.groupby("client_id")["train_time_sec"].mean()
        if model in {"arima", "xgboost", "lstm"}:
            total_train_time = float(per_client_mean.sum())
            time_note = "sum of 20 independent client models"
        elif model == "itransformer":
            total_train_time = float(group["train_time_sec"].iloc[0])
            time_note = "single shared model trained on 321 series"
        elif model == "tabpfn":
            total_train_time = 0.0
            time_note = "zero-shot; no task-specific training"
        else:
            total_train_time = float(group["train_time_sec"].mean())
            time_note = "reported mean training time"

        rows.append(
            {
                "model": model,
                "model_label": style.label,
                "total_train_time_sec": total_train_time,
                "complexity_score": style.complexity,
                "time_note": time_note,
            }
        )

    training = pd.DataFrame(rows)
    training.to_csv(SUMMARY_DIR / "training_time_summary.csv", index=False)
    return training


def figure_rq1_rmse_by_horizon(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for model in MODEL_STYLES:
        style = MODEL_STYLES[model]
        sub = summary[summary["model"] == model].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["RMSE"],
            marker=style.marker,
            linestyle=style.linestyle,
            color=style.color,
            linewidth=2.2,
            markersize=7,
            label=style.label,
        )

    ax.set_title("Forecasting Error by Horizon")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Mean RMSE across 20 evaluation clients (kWh)")
    ax.set_xticks(HORIZONS)
    ax.set_xlim(18, 174)

    ax.annotate(
        "Ranking reversal at 168h:\nLSTM narrowly beats iTransformer",
        xy=(168, 3324.4),
        xytext=(91, 2500),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
        fontsize=9,
    )
    ax.annotate(
        "TabPFN-TS is treated as an\nexploratory zero-shot baseline",
        xy=(48, 1363.6),
        xytext=(62, 900),
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#DDDDDD"},
        fontsize=8.8,
    )

    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(PERFORMANCE_DIR / "figure_1_rq1_rmse_by_horizon.png")
    plt.close(fig)


def figure_rq2_accuracy_vs_training_time(
    summary: pd.DataFrame, training: pd.DataFrame
) -> None:
    merged = summary.merge(training, on=["model", "model_label"], how="left")
    plot_horizons = [24, 168]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, horizon in zip(axes, plot_horizons):
        sub = merged[merged["horizon"] == horizon].copy()
        for _, row in sub.iterrows():
            model = row["model"]
            style = MODEL_STYLES[model]
            x = max(row["total_train_time_sec"], 0.1)
            ax.scatter(
                x,
                row["RMSE"],
                s=120 + style.complexity * 55,
                color=style.color,
                edgecolor="white",
                linewidth=1.4,
                alpha=0.92,
            )
            ax.annotate(
                style.short_label,
                xy=(x, row["RMSE"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                weight="bold" if model in {"tabpfn", "itransformer", "lstm"} else "normal",
            )

        ax.set_xscale("log")
        ax.set_title(f"{horizon}h horizon")
        ax.set_xlabel("Total training time (seconds, log scale)")
        ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["0", "1", "10", "100", "1k", "10k"])
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Mean RMSE (kWh, lower is better)")
    fig.suptitle("Accuracy versus Training Cost", y=0.98, fontsize=14)
    fig.text(
        0.5,
        0.04,
        "Note: ARIMA, XGBoost and LSTM times are cumulative across 20 client-specific models; "
        "iTransformer is one shared 321-series model; TabPFN-TS is zero-shot.",
        ha="center",
        fontsize=8.8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(COST_EFFICIENCY_DIR / "figure_2_rq2_accuracy_vs_training_time.png")
    plt.close(fig)


def figure_client313_robustness(results: pd.DataFrame) -> None:
    rows = []
    for model in MODEL_STYLES:
        for horizon in HORIZONS:
            sub = results[(results["model"] == model) & (results["horizon"] == horizon)]
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_STYLES[model].short_label,
                    "horizon": horizon,
                    "With Client 313": sub["RMSE"].mean(),
                    "Without Client 313": sub[sub["client_id"] != CLIENT_313]["RMSE"].mean(),
                }
            )

    robustness = pd.DataFrame(rows)
    robustness.to_csv(SUMMARY_DIR / "client313_robustness_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.2), sharey=False)
    width = 0.36
    x = np.arange(len(MODEL_STYLES))
    labels = [MODEL_STYLES[m].short_label for m in MODEL_STYLES]

    for ax, horizon in zip(axes, HORIZONS):
        sub = robustness[robustness["horizon"] == horizon].set_index("model")
        with_values = [sub.loc[m, "With Client 313"] for m in MODEL_STYLES]
        without_values = [sub.loc[m, "Without Client 313"] for m in MODEL_STYLES]

        ax.bar(
            x - width / 2,
            with_values,
            width,
            label="With Client 313",
            color="#9ECAE1",
            edgecolor="#355C7D",
            linewidth=0.8,
        )
        ax.bar(
            x + width / 2,
            without_values,
            width,
            label="Without Client 313",
            color="#FDD0A2",
            edgecolor="#A65E2E",
            linewidth=0.8,
        )
        ax.set_title(f"{horizon}h horizon")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Mean RMSE (kWh)")
        ax.text(
            0.02,
            0.96,
            "Model ranking unchanged",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.6,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#CCCCCC"},
        )

    axes[0].legend(loc="upper right", frameon=False, fontsize=8.8)
    fig.suptitle("Robustness Check: Effect of Removing Client 313", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(ROBUSTNESS_DIR / "figure_3_client313_robustness.png")
    plt.close(fig)


def figure_model_selection_matrix() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axvline(5, color="#BBBBBB", linewidth=1.2)
    ax.axhline(5, color="#BBBBBB", linewidth=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Accuracy requirement: lower \u2192 higher")
    ax.set_ylabel("Training resources / time available: lower \u2192 higher")
    ax.set_title("Model Selection Decision Matrix for Forecasting Projects")

    quadrant_labels = [
        (2.5, 8.9, "High resources\nModerate accuracy"),
        (7.5, 8.9, "High resources\nHigh accuracy"),
        (2.5, 1.1, "Low resources\nModerate accuracy"),
        (7.5, 1.1, "Low resources\nHigh accuracy"),
    ]
    for x, y, text in quadrant_labels:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F7F7", "ec": "#DDDDDD"},
        )

    placements = {
        "arima": (2.0, 3.2, "Transparent baseline\nfor regulated settings"),
        "xgboost": (4.0, 4.1, "Fast traditional ML\nwhen features are acceptable"),
        "tabpfn": (7.2, 3.0, "Zero-shot short-horizon\nprototype or rapid deployment"),
        "lstm": (7.0, 7.0, "Trained deep model\nfor longer horizons"),
        "itransformer": (8.4, 8.1, "Shared multivariate model\nfor many related series"),
    }

    for model, (x, y, note) in placements.items():
        style = MODEL_STYLES[model]
        ax.scatter(x, y, s=280, color=style.color, edgecolor="white", linewidth=1.4)
        ax.text(x, y + 0.38, style.label, ha="center", fontsize=10, weight="bold")
        ax.text(x, y - 0.58, note, ha="center", va="top", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(BUSINESS_DIR / "figure_5_model_selection_decision_matrix.png")
    plt.close(fig)


def figure_accuracy_cost_complexity_bubble(
    summary: pd.DataFrame, training: pd.DataFrame
) -> None:
    avg_rmse = summary.groupby(["model", "model_label"], as_index=False)["RMSE"].mean()
    merged = avg_rmse.merge(training, on=["model", "model_label"], how="left")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for _, row in merged.iterrows():
        model = row["model"]
        style = MODEL_STYLES[model]
        x = max(row["total_train_time_sec"], 0.1)
        size = 230 + style.complexity * 140
        ax.scatter(
            x,
            row["RMSE"],
            s=size,
            color=style.color,
            alpha=0.82,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            style.label,
            xy=(x, row["RMSE"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9.5,
            weight="bold",
        )

    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0", "1", "10", "100", "1k", "10k"])
    ax.set_xlabel("Total training time (seconds, log scale)")
    ax.set_ylabel("Average RMSE across 24h/48h/168h (kWh)")
    ax.set_title("Business View: Accuracy, Training Cost and Deployment Complexity")
    ax.text(
        0.02,
        0.04,
        "Bubble size represents qualitative deployment complexity.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#DDDDDD"},
    )
    fig.tight_layout()
    fig.savefig(BUSINESS_DIR / "figure_6_accuracy_cost_complexity_bubble.png")
    plt.close(fig)


def figure_business_scenario_table() -> None:
    rows = [
        [
            "Rapid prototype / limited training resource",
            "TabPFN-TS",
            "Zero task-specific training; strongest 24h/48h RMSE in this experiment.",
        ],
        [
            "Longer-horizon planning (one week ahead)",
            "LSTM / iTransformer",
            "Trained models regain advantage at 168h over the zero-shot baseline.",
        ],
        [
            "Many correlated customer or asset series",
            "iTransformer",
            "One shared multivariate model can forecast all 321 series together.",
        ],
        [
            "Need strong business explainability",
            "XGBoost",
            "SHAP can translate lag, rolling and calendar features into drivers.",
        ],
        [
            "Highly regulated or audit-first setting",
            "ARIMA",
            "Most transparent baseline, but lower accuracy in this dataset.",
        ],
    ]
    columns = ["Business situation", "Recommended model", "Why it fits"]
    table_figure(
        rows,
        columns,
        title="Business Scenario Recommendations",
        out_path=BUSINESS_DIR / "figure_7_business_scenario_recommendations.png",
        figsize=(14.8, 5.2),
        font_size=8.8,
        col_widths=[0.34, 0.18, 0.48],
        wrap_widths=[42, 18, 58],
    )


def figure_explainability_matrix() -> None:
    rows = [
        [
            "ARIMA",
            "AR / MA coefficients and residual diagnostics",
            "High",
            "Clear statistical baseline; useful when transparency is more important than peak accuracy.",
        ],
        [
            "XGBoost",
            "SHAP feature attribution on lag, rolling and calendar features",
            "Medium-High",
            "Best practical explainability option for client-facing business narratives.",
        ],
        [
            "LSTM",
            "Permutation or input-window sensitivity analysis",
            "Low",
            "Accuracy can be strong, but explanations are indirect and harder to defend.",
        ],
        [
            "iTransformer",
            "Cross-series ablation or attention diagnostics",
            "Medium",
            "Useful for diagnosing multivariate dependencies; attention should not be treated as causal proof.",
        ],
        [
            "TabPFN-TS",
            "Context-length sensitivity analysis",
            "Medium",
            "Shows how much historical context is needed; not a traditional feature-importance method.",
        ],
    ]
    columns = ["Model", "Recommended explanation method", "Business readability", "Mentor-facing caveat"]
    table_figure(
        rows,
        columns,
        title="Explainability Plan for Mentor Deliverable",
        out_path=INTERPRETABILITY_DIR / "rq3_interpretability_method_matrix.png",
        figsize=(15.5, 5.7),
        font_size=8.3,
        col_widths=[0.11, 0.32, 0.14, 0.43],
        wrap_widths=[12, 42, 16, 56],
    )


def figure_explainability_readiness() -> None:
    models = ["ARIMA", "XGBoost", "TabPFN-TS", "iTransformer", "LSTM"]
    readability = [5, 4, 3, 3, 1]
    accuracy_24h_rank = [5, 4, 1, 2, 3]
    colors = [
        MODEL_STYLES["arima"].color,
        MODEL_STYLES["xgboost"].color,
        MODEL_STYLES["tabpfn"].color,
        MODEL_STYLES["itransformer"].color,
        MODEL_STYLES["lstm"].color,
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(models))
    ax.barh(y, readability, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 5.4)
    ax.set_xlabel("Business explainability score (1 = low, 5 = high)")
    ax.set_title("Explainability Readiness for Client Discussions")
    ax.invert_yaxis()

    notes = [
        "Coefficients are directly inspectable",
        "SHAP gives defensible feature-level explanations",
        "Explain through context sensitivity, not feature importance",
        "Attention diagnostics require careful technical framing",
        "Treat as black-box; explanations are indirect",
    ]
    for i, (score, note, rank) in enumerate(zip(readability, notes, accuracy_24h_rank)):
        note_x = max(score + 0.08, 1.25)
        ax.text(note_x, i, note, va="center", fontsize=9)
        ax.text(
            0.05,
            i,
            f"24h accuracy rank: {rank}",
            va="center",
            ha="left",
            fontsize=8,
            color="white" if score > 2 else "#333333",
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(INTERPRETABILITY_DIR / "rq3_interpretability_readiness.png")
    plt.close(fig)


def table_figure(
    rows: list[list[str]],
    columns: list[str],
    title: str,
    out_path: Path,
    figsize: tuple[float, float],
    font_size: float,
    col_widths: list[float] | None = None,
    wrap_widths: list[int] | None = None,
) -> None:
    if wrap_widths:
        wrapped_rows = [
            [textwrap.fill(str(value), width=wrap_widths[i]) for i, value in enumerate(row)]
            for row in rows
        ]
        wrapped_columns = [
            textwrap.fill(str(value), width=wrap_widths[i]) for i, value in enumerate(columns)
        ]
    else:
        wrapped_rows = rows
        wrapped_columns = columns

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=14)

    table = ax.table(
        cellText=wrapped_rows,
        colLabels=wrapped_columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=col_widths,
        colColours=["#F0F0F0"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.72)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.12)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FAFAFA")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    setup()
    results = load_results()
    summary = aggregate_metrics(results)
    training = total_training_time(results)

    figure_rq1_rmse_by_horizon(summary)
    figure_rq2_accuracy_vs_training_time(summary, training)
    figure_client313_robustness(results)

    figure_model_selection_matrix()
    figure_accuracy_cost_complexity_bubble(summary, training)
    figure_business_scenario_table()

    figure_explainability_matrix()
    figure_explainability_readiness()

    print("Created classified figure outputs:")
    print(f"  Performance comparison: {PERFORMANCE_DIR}")
    print(f"  Cost efficiency:        {COST_EFFICIENCY_DIR}")
    print(f"  Robustness checks:      {ROBUSTNESS_DIR}")
    print(f"  Business decisions:     {BUSINESS_DIR}")
    print(f"  Interpretability:       {INTERPRETABILITY_DIR}")
    print(f"  Summary tables:         {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
