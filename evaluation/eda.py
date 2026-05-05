"""EDA charts and statistics for Ragas evaluation results.

Usage:
    python -m evaluation.eda
    python -m evaluation.eda --csv evaluation/results/ragas_results_100.csv
    python -m evaluation.eda --csv evaluation/results/ragas_results_100.csv --out evaluation/charts

    # Compare two runs (e.g. no-RRF vs RRF):
    python -m evaluation.eda --compare baseline.csv experiment.csv --labels "No RRF" "With RRF"
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
    """Generate score-distribution charts per metric, split by intent (PS / SM / NS).

    Two styles per metric:
      - 10a: KDE density curves (smooth lines, easy shape comparison)
      - 10b: Grouped bars (side-by-side, easy count reading)

    Returns the number of chart files saved.
    """
    if "intent" not in df.columns:
        return 0

    intents = [k for k in ("product_search", "similar_search", "news_search")
               if k in df["intent"].values]
    if not intents:
        return 0

    saved = 0
    bins = np.linspace(0, 1, 11)  # 0.0, 0.1, … 1.0

    for metric in metrics:
        # ── 10a: KDE lines ──
        fig, ax = plt.subplots(figsize=(8, 5))
        has_data = False
        for intent_key in intents:
            vals = df.loc[(df["intent"] == intent_key) & df[metric].notna(), metric]
            if vals.empty:
                continue
            has_data = True
            label = (f"{INTENT_LABELS.get(intent_key, intent_key)}  "
                     f"(n={len(vals)}, avg={vals.mean():.3f})")
            sns.kdeplot(vals, ax=ax, color=INTENT_COLORS.get(intent_key, "#999"),
                        linewidth=2.5, label=label, clip=(0, 1), fill=True, alpha=0.15)
        if has_data:
            display = METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())
            ax.set_title(f"Score Distribution — {display}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Score", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_xlim(-0.02, 1.02)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"10a_kde_{metric}.png", dpi=150, bbox_inches="tight")
            saved += 1
        plt.close(fig)

        # ── 10b: Grouped bars ──
        fig, ax = plt.subplots(figsize=(9, 5))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = 0.8 / len(intents)  # bar width per intent
        has_data = False
        for i, intent_key in enumerate(intents):
            vals = df.loc[(df["intent"] == intent_key) & df[metric].notna(), metric]
            if vals.empty:
                continue
            has_data = True
            counts, _ = np.histogram(vals, bins=bins)
            offset = (i - len(intents) / 2 + 0.5) * width
            label = (f"{INTENT_LABELS.get(intent_key, intent_key)}  "
                     f"(n={len(vals)}, avg={vals.mean():.3f})")
            ax.bar(bin_centers + offset, counts, width=width, label=label,
                   color=INTENT_COLORS.get(intent_key, "#999"), edgecolor="white", linewidth=0.5)
        if has_data:
            display = METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())
            ax.set_title(f"Score Distribution — {display}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Score", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(bin_centers)
            ax.set_xticklabels([f"{v:.1f}" for v in bin_centers], fontsize=9)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"10b_grouped_{metric}.png", dpi=150, bbox_inches="tight")
            saved += 1
        plt.close(fig)

    return saved


# ── 11. A/B comparison (two CSV files) ────────────────────────

def compare_runs(
    csv_a: Path, csv_b: Path, out: Path,
    label_a: str = "Baseline", label_b: str = "Experiment",
) -> None:
    """Compare two evaluation CSV files side-by-side.

    Generates:
      - 11a_compare_heatmap.png  — delta heatmap (intent × metric)
      - 11b_compare_bars.png     — grouped bars (baseline vs experiment per metric per intent)
      - Prints markdown table to stdout
    """
    df_a = load(csv_a)
    df_b = load(csv_b)
    metrics = [m for m in available_metrics(df_a) if m in available_metrics(df_b)]
    intents = sorted(set(df_a["intent"].dropna().unique()) & set(df_b["intent"].dropna().unique()))
    if not metrics or not intents:
        print("  [compare] No overlapping metrics or intents — skipped.")
        return

    out.mkdir(parents=True, exist_ok=True)

    # Build mean tables
    rows_md = []
    data_a, data_b, data_delta = [], [], []
    for intent in intents:
        row_a, row_b, row_d = {}, {}, {}
        for m in metrics:
            va = df_a.loc[df_a["intent"] == intent, m].dropna()
            vb = df_b.loc[df_b["intent"] == intent, m].dropna()
            ma = va.mean() if not va.empty else float("nan")
            mb = vb.mean() if not vb.empty else float("nan")
            delta = mb - ma
            row_a[m] = ma
            row_b[m] = mb
            row_d[m] = delta
            sign = "+" if delta > 0 else ""
            rows_md.append({
                "Intent": intent, "Metric": m,
                label_a: f"{ma:.3f}", label_b: f"{mb:.3f}",
                "Δ": f"{sign}{delta:.3f}",
            })
        data_a.append(row_a)
        data_b.append(row_b)
        data_delta.append(row_d)

    tbl_a = pd.DataFrame(data_a, index=intents)
    tbl_b = pd.DataFrame(data_b, index=intents)
    tbl_delta = pd.DataFrame(data_delta, index=intents)

    # ── Print markdown table ──
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"  A = {csv_a.name} ({len(df_a)} cases)")
    print(f"  B = {csv_b.name} ({len(df_b)} cases)")
    print(f"{'=' * 80}\n")

    header = f"| {'Intent':<16s} | {'Metric':<20s} | {label_a:>10s} | {label_b:>10s} | {'Δ':>8s} |"
    sep = f"|{'-'*18}|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*10}|"
    print(header)
    print(sep)
    for r in rows_md:
        d = float(r['Δ'])
        arrow = '↑' if d > 0.01 else ('↓' if d < -0.01 else '≈')
        print(f"| {r['Intent']:<16s} | {r['Metric']:<20s} | {r[label_a]:>10s} | {r[label_b]:>10s} | {r['Δ']:>6s} {arrow} |")
    print()

    # ── 11a: Delta heatmap ──
    fig, ax = plt.subplots(figsize=(8, max(3, len(intents) * 1.2)))
    display_cols = [METRIC_DISPLAY.get(m, m) for m in metrics]
    plot_delta = tbl_delta.copy()
    plot_delta.columns = display_cols
    sns.heatmap(
        plot_delta, annot=True, fmt="+.3f", center=0,
        cmap="RdYlGn", linewidths=1, ax=ax,
        cbar_kws={"label": "Δ score"},
        vmin=-0.5, vmax=0.5,
    )
    ax.set_title(f"Score Delta: {label_b} − {label_a}", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / "11a_compare_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 11b: Grouped bars (A vs B per intent per metric) ──
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(intents))
    w = 0.35
    for ax, m in zip(axes, metrics):
        vals_a = [tbl_a.loc[i, m] for i in intents]
        vals_b = [tbl_b.loc[i, m] for i in intents]
        ax.bar(x - w / 2, vals_a, w, label=label_a, color="#FF8A65", edgecolor="white")
        ax.bar(x + w / 2, vals_b, w, label=label_b, color="#26A69A", edgecolor="white")
        ax.set_title(METRIC_DISPLAY.get(m, m), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(intents, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "11b_compare_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Comparison charts saved to {out}/")


# ── 12. Multi-run comparison table ─────────────────────────────

def compare_multi(
    csv_paths: list[Path],
    labels: list[str],
    out: Path,
) -> None:
    """Compare N evaluation CSV files in a single markdown table + CSV.

    Generates:
      - Markdown table to stdout (copy-paste ready)
      - 12_compare_table.csv — same data as CSV
    """
    dfs = {}
    for p, label in zip(csv_paths, labels):
        dfs[label] = load(p)

    all_metrics = None
    for df in dfs.values():
        m = set(available_metrics(df))
        all_metrics = m if all_metrics is None else all_metrics & m
    metrics = [m for m in ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
               if m in (all_metrics or set())]

    all_intents = None
    for df in dfs.values():
        i = set(df["intent"].dropna().unique())
        all_intents = i if all_intents is None else all_intents & i
    intents = sorted(all_intents or set())

    if not metrics or not intents:
        print("  [compare-multi] No overlapping metrics or intents — skipped.")
        return

    out.mkdir(parents=True, exist_ok=True)

    def _best(r, keys):
        scores = {l: r[l] for l in keys if r.get(l) is not None}
        return max(scores, key=scores.get) if scores else None

    col_w = max(10, max(len(l) for l in labels) + 1)

    def _save_table(rows, col_names, title, prefix, caption, tab_label):
        """Shared helper: print markdown, save CSV + LaTeX + PNG."""
        # ── Markdown ──
        print(f"\n  ── {title} ──\n")
        hdr = "| " + " | ".join(f"{c:<{20 if i < len(col_names) - len(labels) else col_w}s}"
                                  if i < len(col_names) - len(labels)
                                  else f"{c:>{col_w}s}"
                                  for i, c in enumerate(col_names)) + " |"
        sep = "|" + "|".join("-" * (22 if i < len(col_names) - len(labels) else col_w + 2)
                              for i in range(len(col_names))) + "|"
        print(hdr)
        print(sep)
        prev_first = None
        for r in rows:
            first_val = list(r.values())[0]
            if first_val == "OVERALL" and prev_first != "OVERALL":
                print(sep)
            best = _best(r, labels)
            parts = []
            for i, c in enumerate(col_names):
                v = r.get(c)
                if i < len(col_names) - len(labels):
                    parts.append(f"{str(v):<20s}" if v is not None else f"{'':20s}")
                else:
                    s = f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—"
                    if c == best and v is not None:
                        s = f"**{s}**"
                    parts.append(f"{s:>{col_w}s}")
            print("| " + " | ".join(parts) + " |")
            prev_first = first_val
        print()

        # ── CSV ──
        csv_path = out / f"{prefix}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  CSV   → {csv_path}")

        # ── LaTeX ──
        n_text_cols = len(col_names) - len(labels)
        tex_col_spec = "l" * n_text_cols + "r" * len(labels)
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{tab_label}}}",
            r"\small" if len(rows) > 6 else "",
            f"\\begin{{tabular}}{{{tex_col_spec}}}",
            r"\toprule",
            " & ".join(col_names) + r" \\",
            r"\midrule",
        ]
        prev_first = None
        for r in rows:
            first_val = list(r.values())[0]
            if first_val == "OVERALL" and prev_first != "OVERALL":
                tex_lines.append(r"\midrule")
            best = _best(r, labels)
            cells = []
            for i, c in enumerate(col_names):
                v = r.get(c)
                if i < n_text_cols:
                    show = str(v) if v is not None and v != prev_first else ""
                    if first_val == "OVERALL" and i == 0:
                        show = r"\textbf{OVERALL}" if v == first_val else ""
                    cells.append(show)
                else:
                    s = f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—"
                    if c == best and v is not None:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
            tex_lines.append(" & ".join(cells) + r" \\")
            prev_first = first_val
        tex_lines = [l for l in tex_lines if l]  # remove empty
        tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tex_path = out / f"{prefix}.tex"
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        print(f"  LaTeX → {tex_path}")

        # ── PNG ──
        n_rows = len(rows) + 1
        fig_h = max(2.5, 0.40 * n_rows + 0.8)
        fig_w = max(8, 1.8 * n_text_cols + col_w * 0.13 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        cell_text, cell_colors = [], []
        hdr_color = "#37474F"
        row_a, row_b = "#FAFAFA", "#F0F0F0"
        overall_bg = "#E3F2FD"
        best_clr = "#C8E6C9"

        for idx, r in enumerate(rows):
            best = _best(r, labels)
            is_overall = list(r.values())[0] == "OVERALL"
            base = overall_bg if is_overall else (row_a if idx % 2 == 0 else row_b)
            rc, cc = [], []
            for i, c in enumerate(col_names):
                v = r.get(c)
                if i < n_text_cols:
                    rc.append(str(v) if v is not None else "")
                    cc.append(base)
                else:
                    rc.append(f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—")
                    cc.append(best_clr if c == best and v is not None else base)
            cell_text.append(rc)
            cell_colors.append(cc)

        table = ax.table(
            cellText=cell_text, colLabels=col_names,
            cellColours=cell_colors,
            colColours=[hdr_color] * len(col_names),
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#BDBDBD")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            elif list(rows[row - 1].values())[0] == "OVERALL" if row - 1 < len(rows) else False:
                cell.set_text_props(fontweight="bold")
        fig.tight_layout()
        png_path = out / f"{prefix}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  PNG   → {png_path}")

    # ── Header ──
    print(f"\n{'=' * 80}")
    print(f"  MULTI-RUN COMPARISON  ({len(labels)} configs × {len(intents)} intents × {len(metrics)} metrics)")
    for label, p in zip(labels, csv_paths):
        print(f"    {label:<25s} ← {p.name} ({len(dfs[label])} cases)")
    print(f"{'=' * 80}")

    # ── Table 12a: By intent ──
    rows_intent = []
    for intent in intents:
        for m in metrics:
            row = {"Intent": intent, "Metric": METRIC_DISPLAY.get(m, m)}
            for label in labels:
                vals = dfs[label].loc[dfs[label]["intent"] == intent, m].dropna()
                row[label] = round(vals.mean(), 3) if not vals.empty else None
            rows_intent.append(row)
    _save_table(
        rows_intent,
        col_names=["Intent", "Metric"] + labels,
        title="Table 12a: By Intent",
        prefix="12a_compare_by_intent",
        caption="Retrieval quality by intent across embedding models and fusion strategies.",
        tab_label="tab:compare-intent",
    )

    # ── Table 12b: Overall ──
    rows_overall = []
    for m in metrics:
        row = {"Metric": METRIC_DISPLAY.get(m, m)}
        for label in labels:
            vals = dfs[label][m].dropna()
            row[label] = round(vals.mean(), 3) if not vals.empty else None
        rows_overall.append(row)
    _save_table(
        rows_overall,
        col_names=["Metric"] + labels,
        title="Table 12b: Overall",
        prefix="12b_compare_overall",
        caption=f"Overall retrieval quality across embedding models and fusion strategies (mean over {len(next(iter(dfs.values())))} evaluation cases).",
        tab_label="tab:compare-overall",
    )


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
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "EXPERIMENT"),
                        help="Compare two CSV files (e.g. --compare no_rrf.csv rrf.csv)")
    parser.add_argument("--compare-multi", nargs="+", type=Path, metavar="CSV",
                        help="Compare N CSV files (e.g. --compare-multi a.csv b.csv c.csv d.csv)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for runs (must match number of CSVs)")
    args = parser.parse_args()

    # ── Multi-compare mode ──
    if args.compare_multi:
        csvs = args.compare_multi
        for f in csvs:
            if not f.exists():
                print(f"CSV not found: {f}")
                sys.exit(1)
        labels = args.labels or [p.stem for p in csvs]
        if len(labels) != len(csvs):
            print(f"Error: {len(csvs)} CSVs but {len(labels)} labels")
            sys.exit(1)
        out = args.out or Path("evaluation/results/charts_compare")
        compare_multi(csvs, labels, out)
        return

    # ── Two-way compare mode ──
    if args.compare:
        csv_a, csv_b = args.compare
        for f in (csv_a, csv_b):
            if not f.exists():
                print(f"CSV not found: {f}")
                sys.exit(1)
        lab = args.labels or ["Baseline", "Experiment"]
        out = args.out or Path("evaluation/results/charts_compare")
        compare_runs(csv_a, csv_b, out, label_a=lab[0], label_b=lab[1])
        return

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
