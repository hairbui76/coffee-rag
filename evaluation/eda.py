"""EDA charts and statistics for Ragas evaluation results.

Usage:
    python -m evaluation.eda
    python -m evaluation.eda --csv evaluation/results/ragas_results_100.csv
    python -m evaluation.eda --csv evaluation/results/ragas_results_100.csv --out evaluation/charts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

METRIC_COLS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
PALETTE = {"context_precision": "#4C72B0", "context_recall": "#DD8452",
           "faithfulness": "#55A868", "answer_relevancy": "#C44E52"}

sns.set_theme(style="whitegrid", font_scale=1.1)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    return [c for c in METRIC_COLS if c in df.columns and df[c].notna().sum() > 0]


# ── 1. Score distributions ───────────────────────────────────

def plot_distributions(df: pd.DataFrame, metrics: list[str], out: Path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for i, col in enumerate(metrics):
        ax = axes[0, i]
        vals = df[col].dropna()
        sns.histplot(vals, bins=15, kde=True, color=PALETTE.get(col, "#4C72B0"), ax=ax, edgecolor="white")
        ax.axvline(vals.mean(), color="red", ls="--", lw=1.5, label=f"mean={vals.mean():.3f}")
        ax.axvline(vals.median(), color="orange", ls=":", lw=1.5, label=f"median={vals.median():.3f}")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=9)
    fig.suptitle("Score Distributions", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "1_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Boxplot comparison ────────────────────────────────────

def plot_boxplot(df: pd.DataFrame, metrics: list[str], out: Path):
    melted = df[metrics].melt(var_name="metric", value_name="score").dropna()
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(metrics)), 5))
    sns.boxplot(data=melted, x="metric", y="score", palette=PALETTE, ax=ax, width=0.5)
    sns.stripplot(data=melted, x="metric", y="score", color="black", alpha=0.3, size=3, ax=ax, jitter=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Metric Score Comparison", fontweight="bold")
    ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()])
    fig.tight_layout()
    fig.savefig(out / "2_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Scores by intent ──────────────────────────────────────

def plot_by_intent(df: pd.DataFrame, metrics: list[str], out: Path):
    if "intent" not in df.columns or df["intent"].nunique() < 2:
        return
    intent_order = df.groupby("intent")[metrics[0]].mean().sort_values(ascending=False).index.tolist()
    melted = df.melt(id_vars=["intent"], value_vars=metrics, var_name="metric", value_name="score").dropna()
    fig, ax = plt.subplots(figsize=(max(8, len(intent_order) * 1.5), 5))
    sns.barplot(data=melted, x="intent", y="score", hue="metric", palette=PALETTE,
                order=intent_order, ax=ax, errorbar="sd", capsize=0.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score (mean ± sd)")
    ax.set_title("Scores by Intent", fontweight="bold")
    ax.legend(title="", loc="lower right", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out / "3_by_intent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 4. Scores by difficulty ──────────────────────────────────

def plot_by_difficulty(df: pd.DataFrame, metrics: list[str], out: Path):
    if "difficulty" not in df.columns or df["difficulty"].nunique() < 2:
        return
    order = [d for d in ["easy", "medium", "hard"] if d in df["difficulty"].values]
    melted = df.melt(id_vars=["difficulty"], value_vars=metrics, var_name="metric", value_name="score").dropna()
    fig, ax = plt.subplots(figsize=(max(6, len(order) * 2.5), 5))
    sns.barplot(data=melted, x="difficulty", y="score", hue="metric", palette=PALETTE,
                order=order, ax=ax, errorbar="sd", capsize=0.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score (mean ± sd)")
    ax.set_title("Scores by Difficulty", fontweight="bold")
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "4_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 5. Scores by language ────────────────────────────────────

def plot_by_language(df: pd.DataFrame, metrics: list[str], out: Path):
    if "language" not in df.columns or df["language"].nunique() < 2:
        return
    melted = df.melt(id_vars=["language"], value_vars=metrics, var_name="metric", value_name="score").dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=melted, x="language", y="score", hue="metric", palette=PALETTE,
                ax=ax, errorbar="sd", capsize=0.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score (mean ± sd)")
    ax.set_title("Scores by Language", fontweight="bold")
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "5_by_language.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 6. Precision vs Recall scatter ───────────────────────────

def plot_precision_recall(df: pd.DataFrame, out: Path) -> int:
    """One scatter plot per intent. Returns number of charts generated."""
    if "context_precision" not in df.columns or "context_recall" not in df.columns:
        return 0
    sub = df[["context_precision", "context_recall", "intent"]].dropna()
    if sub.empty:
        return 0

    intents = sorted(sub["intent"].unique())
    colors = sns.color_palette("husl", len(intents))
    intent_color = dict(zip(intents, colors))

    for intent in intents:
        idf = sub[sub["intent"] == intent]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(idf["context_precision"], idf["context_recall"],
                   s=80, alpha=0.7, color=intent_color[intent], edgecolor="white", linewidth=0.5)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5, lw=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Context Precision")
        ax.set_ylabel("Context Recall")
        mean_p = idf["context_precision"].mean()
        mean_r = idf["context_recall"].mean()
        ax.set_title(f"Precision vs Recall — {intent}\n"
                     f"(n={len(idf)}, mean P={mean_p:.2f}, R={mean_r:.2f})",
                     fontweight="bold", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"6_precision_recall_{intent}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Also save a combined overview grid
    ncols = min(3, len(intents))
    nrows = (len(intents) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for idx, intent in enumerate(intents):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        idf = sub[sub["intent"] == intent]
        ax.scatter(idf["context_precision"], idf["context_recall"],
                   s=60, alpha=0.7, color=intent_color[intent], edgecolor="white", linewidth=0.5)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.4, lw=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Precision", fontsize=9)
        ax.set_ylabel("Recall", fontsize=9)
        mean_p = idf["context_precision"].mean()
        mean_r = idf["context_recall"].mean()
        ax.set_title(f"{intent} (n={len(idf)})\nP={mean_p:.2f}  R={mean_r:.2f}",
                     fontweight="bold", fontsize=10)
    for idx in range(len(intents), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Context Precision vs Recall by Intent", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "6_precision_recall_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return len(intents)


# ── 7. Failure analysis ──────────────────────────────────────

def plot_failure_rate(df: pd.DataFrame, metrics: list[str], out: Path, threshold: float = 0.3):
    if "intent" not in df.columns:
        return
    intents = sorted(df["intent"].dropna().unique())
    data = []
    for intent in intents:
        subset = df[df["intent"] == intent]
        for m in metrics:
            vals = subset[m].dropna()
            if len(vals) == 0:
                continue
            fail_pct = (vals < threshold).sum() / len(vals) * 100
            data.append({"intent": intent, "metric": m, "fail_pct": fail_pct})
    if not data:
        return
    fdf = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(max(8, len(intents) * 1.5), 5))
    sns.barplot(data=fdf, x="intent", y="fail_pct", hue="metric", palette=PALETTE, ax=ax)
    ax.set_ylabel(f"% cases with score < {threshold}")
    ax.set_xlabel("")
    ax.set_title(f"Failure Rate (score < {threshold}) by Intent", fontweight="bold")
    ax.legend(title="", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out / "7_failure_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 8. Timing distribution ───────────────────────────────────

def plot_timing(df: pd.DataFrame, out: Path):
    if "elapsed_s" not in df.columns:
        return
    vals = df["elapsed_s"].dropna()
    if vals.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(vals, bins=20, kde=True, color="#6A5ACD", ax=axes[0], edgecolor="white")
    axes[0].axvline(vals.mean(), color="red", ls="--", lw=1.5, label=f"mean={vals.mean():.1f}s")
    axes[0].set_title("Elapsed Time Distribution", fontweight="bold")
    axes[0].set_xlabel("Seconds")
    axes[0].legend(fontsize=9)

    if "intent" in df.columns and df["intent"].nunique() > 1:
        sns.boxplot(data=df, x="intent", y="elapsed_s", color="#6A5ACD", ax=axes[1], width=0.5)
        axes[1].set_title("Elapsed Time by Intent", fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Seconds")
        plt.sca(axes[1])
        plt.xticks(rotation=25, ha="right")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "8_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 9. Heatmap: intent × metric ──────────────────────────────

def plot_heatmap(df: pd.DataFrame, metrics: list[str], out: Path):
    if "intent" not in df.columns or df["intent"].nunique() < 2:
        return
    pivot = df.groupby("intent")[metrics].mean()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 2), max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "avg score"})
    ax.set_title("Average Score Heatmap (intent × metric)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()])
    fig.tight_layout()
    fig.savefig(out / "9_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 10. Score distribution by intent (1 chart per metric) ───

INTENT_LABELS = {
    "product_search": "PS (Product Search)",
    "similar_search": "SM (Similar Search)",
    "news_search": "NS (News Search)",
}

INTENT_COLORS = {
    "product_search": "#2196F3",   # blue
    "similar_search": "#FF9800",   # orange
    "news_search": "#4CAF50",     # green
}

METRIC_DISPLAY = {
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_relevancy": "Answer Relevancy",
}


def plot_dist_by_intent(df: pd.DataFrame, metrics: list[str], out: Path) -> int:
    """Generate one score-distribution chart per metric, split by intent (PS / SM / NS).

    Each chart is an overlaid histogram with 3 colours.
    Returns the number of charts saved.
    """
    if "intent" not in df.columns:
        return 0

    saved = 0
    bins = np.linspace(0, 1, 21)  # 0.00, 0.05, … 1.00

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        has_data = False

        for intent_key in ("product_search", "similar_search", "news_search"):
            vals = df.loc[
                (df["intent"] == intent_key) & df[metric].notna(), metric
            ]
            if vals.empty:
                continue
            has_data = True
            label = (
                f"{INTENT_LABELS.get(intent_key, intent_key)}  "
                f"(n={len(vals)}, avg={vals.mean():.3f})"
            )
            ax.hist(
                vals, bins=bins, alpha=0.55, label=label,
                color=INTENT_COLORS.get(intent_key, "#999999"),
                edgecolor="white", linewidth=0.5,
            )

        if not has_data:
            plt.close(fig)
            continue

        display = METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())
        ax.set_title(f"Score Distribution — {display}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"10_dist_by_intent_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


# ── Stats table ───────────────────────────────────────────────

def print_stats(df: pd.DataFrame, metrics: list[str]):
    print(f"\n{'=' * 70}")
    print(f"  EVALUATION STATISTICS  ({len(df)} cases)")
    print(f"{'=' * 70}")

    print(f"\n  Overall:")
    for m in metrics:
        vals = df[m].dropna()
        if vals.empty:
            continue
        print(f"    {m:<24s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"median={vals.median():.3f}  min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")

    if "intent" in df.columns and df["intent"].nunique() > 1:
        print(f"\n  By intent:")
        for intent in sorted(df["intent"].dropna().unique()):
            sub = df[df["intent"] == intent]
            parts = []
            for m in metrics:
                vals = sub[m].dropna()
                if not vals.empty:
                    parts.append(f"{m}={vals.mean():.3f}")
            print(f"    {intent:<22s}  n={len(sub):>3d}  {', '.join(parts)}")

    if "difficulty" in df.columns and df["difficulty"].nunique() > 1:
        print(f"\n  By difficulty:")
        for diff in ["easy", "medium", "hard"]:
            sub = df[df["difficulty"] == diff]
            if sub.empty:
                continue
            parts = []
            for m in metrics:
                vals = sub[m].dropna()
                if not vals.empty:
                    parts.append(f"{m}={vals.mean():.3f}")
            print(f"    {diff:<22s}  n={len(sub):>3d}  {', '.join(parts)}")

    if "language" in df.columns and df["language"].nunique() > 1:
        print(f"\n  By language:")
        for lang in sorted(df["language"].dropna().unique()):
            sub = df[df["language"] == lang]
            parts = []
            for m in metrics:
                vals = sub[m].dropna()
                if not vals.empty:
                    parts.append(f"{m}={vals.mean():.3f}")
            print(f"    {lang:<22s}  n={len(sub):>3d}  {', '.join(parts)}")

    # Worst cases
    for m in metrics:
        vals = df[m].dropna()
        if vals.empty:
            continue
        bottom = df.nsmallest(5, m)[["id", "intent", "difficulty", "language", m, "question"]]
        print(f"\n  Bottom 5 by {m}:")
        for _, row in bottom.iterrows():
            q = str(row.get("question", ""))[:60]
            print(f"    {row.get('id',''):<10s} {row[m]:.3f}  ({row.get('intent','')}, {row.get('difficulty','')})  {q}")

    if "elapsed_s" in df.columns:
        vals = df["elapsed_s"].dropna()
        if not vals.empty:
            print(f"\n  Timing:  mean={vals.mean():.1f}s  std={vals.std():.1f}s  "
                  f"min={vals.min():.1f}s  max={vals.max():.1f}s")

    print(f"\n{'=' * 70}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA for Ragas evaluation results.")
    parser.add_argument("--csv", type=Path, default=ROOT / "evaluation" / "results" / "ragas_results.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for charts. Defaults to same dir as CSV.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    out = args.out or args.csv.parent / "charts"
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    metrics = available_metrics(df)
    if not metrics:
        print("No metric columns with data found.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Metrics: {metrics}")
    print(f"Charts → {out}/\n")

    plot_distributions(df, metrics, out)
    print("  [1/10] distributions")

    plot_boxplot(df, metrics, out)
    print("  [2/10] boxplot")

    plot_by_intent(df, metrics, out)
    print("  [3/10] by intent")

    plot_by_difficulty(df, metrics, out)
    print("  [4/10] by difficulty")

    plot_by_language(df, metrics, out)
    print("  [5/10] by language")

    n_pr = plot_precision_recall(df, out)
    if n_pr:
        print(f"  [6/10] precision vs recall ({n_pr} intent charts + 1 overview)")
    else:
        print("  [6/10] precision vs recall (skipped)")

    plot_failure_rate(df, metrics, out)
    print("  [7/10] failure rate")

    plot_timing(df, out)
    print("  [8/10] timing")

    plot_heatmap(df, metrics, out)
    print("  [9/10] heatmap")

    n_dist = plot_dist_by_intent(df, metrics, out)
    if n_dist:
        print(f"  [10/10] score dist by intent ({n_dist} charts)")
    else:
        print("  [10/10] score dist by intent (skipped — need intent column)")

    print_stats(df, metrics)
    print(f"\nAll charts saved to {out}/")


if __name__ == "__main__":
    main()
