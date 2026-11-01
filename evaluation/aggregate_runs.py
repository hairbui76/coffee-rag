"""Aggregate repeated evaluation runs into publication tables.

Two tables, kept apart on purpose:

* **System table** (``--run``): one row per system configuration, every row
  scored with the same metrics. Cells are mean ± SD across runs (SD of the
  per-run means). Paired tests compare each configuration with ``--baseline``
  on per-case scores averaged across runs: Wilcoxon signed-rank p-value and a
  case-level bootstrap 95% CI of the mean difference.

* **Metric table** (``--metric-table LABEL``): one system, several ways of
  measuring recall side by side (Ragas context_recall, retrieval_recall_soft,
  recall_at_k, hit_at_k), by intent, plus the Spearman correlation between the
  metrics across cases. Changing the metric is not a system improvement, so
  these numbers never share a table with the system comparison.

Usage:
    python -m evaluation.aggregate_runs \
        --run "MiniLM"          results_L12_v2_no_rrf.csv \
        --run "MiniLM+RRF"      results_L12_v2.csv \
        --run "BGE-M3"          results_norrf.csv \
        --run "BGE-M3+RRF"      results.csv \
        --soft-file results_soft.csv --soft-label "BGE-M3+RRF" \
        --baseline "BGE-M3+RRF" --metric-table "BGE-M3+RRF" \
        --out evaluation/results/tables

    # v6, three runs per configuration
    python -m evaluation.aggregate_runs \
        --run "BGE-M3+RRF" evaluation/results/v6/bge_rrf_r1.csv evaluation/results/v6/bge_rrf_r2.csv evaluation/results/v6/bge_rrf_r3.csv \
        --run "BGE-M3"     evaluation/results/v6/bge_norrf_r1.csv ... \
        --soft-file evaluation/results/v6/bge_rrf_soft_r1.csv ... --soft-label "BGE-M3+RRF" \
        --baseline "BGE-M3+RRF" --metric-table "BGE-M3+RRF" --out evaluation/results/v6/tables
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_METRICS = [
    "context_precision", "context_recall", "faithfulness", "answer_relevancy",
    "hit_at_k", "recall_at_k", "precision_at_k", "mrr",
]
RECALL_METRICS = ["context_recall", "retrieval_recall_soft", "recall_at_k", "hit_at_k", "context_precision"]
PRETTY = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall (Ragas)",
    "retrieval_recall_soft": "Soft Recall@k",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "hit_at_k": "Hit@k",
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "mrr": "MRR",
}
INTENTS = ["product_search", "similar_search", "news_search"]


# ── Loading ────────────────────────────────────────────────────


def load_run(path: Path, soft: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if soft and "context_recall" in df.columns and "retrieval_recall_soft" not in df.columns:
        # A --soft run stores the soft metric under context_recall.
        df = df.rename(columns={"context_recall": "retrieval_recall_soft"})
    for col in SYSTEM_METRICS + ["retrieval_recall_soft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_run"] = path.stem
    return df


def per_case_mean(runs: list[pd.DataFrame], metric: str) -> pd.Series:
    """Mean score per case id across runs (cases missing in a run are skipped)."""
    frames = [r[["id", metric]].dropna() for r in runs if metric in r.columns]
    if not frames:
        return pd.Series(dtype=float)
    allr = pd.concat(frames)
    return allr.groupby("id")[metric].mean()


# ── Statistics ─────────────────────────────────────────────────


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wilcoxon_p(diff: np.ndarray) -> float:
    if _stats is None or len(diff) < 10 or np.allclose(diff, 0):
        return float("nan")
    try:
        return float(_stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def fmt_p(p: float) -> str:
    if p != p:
        return "–"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ── Tables ─────────────────────────────────────────────────────


def system_table(runs_by_label: dict[str, list[pd.DataFrame]], metrics: list[str],
                 by_intent: bool) -> pd.DataFrame:
    rows = []
    groups = INTENTS if by_intent else ["all"]
    for label, runs in runs_by_label.items():
        for grp in groups:
            row = {"System": label, "Intent": grp, "Runs": len(runs)}
            for m in metrics:
                per_run = []
                for r in runs:
                    if m not in r.columns:
                        continue
                    sub = r if grp == "all" else r[r["intent"] == grp]
                    vals = sub[m].dropna()
                    if len(vals):
                        per_run.append(vals.mean())
                if not per_run:
                    continue
                row[m] = float(np.mean(per_run))
                row[m + "_sd"] = float(np.std(per_run, ddof=1)) if len(per_run) > 1 else float("nan")
                row[m + "_n"] = int(min(len(r[r["intent"] == grp]) if grp != "all" else len(r) for r in runs))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(runs_by_label: dict[str, list[pd.DataFrame]], baseline: str,
                 metrics: list[str]) -> pd.DataFrame:
    rows = []
    base = runs_by_label[baseline]
    for label, runs in runs_by_label.items():
        if label == baseline:
            continue
        for m in metrics:
            a = per_case_mean(base, m)
            b = per_case_mean(runs, m)
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue
            diff = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff)
            rows.append({
                "System": label, "vs": baseline, "Metric": m, "n_cases": int(len(common)),
                "mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                "wilcoxon_p": wilcoxon_p(diff),
                "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
            })
    return pd.DataFrame(rows)


def metric_table(runs: list[pd.DataFrame], soft_runs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall-type metrics side by side for one system, by intent, plus Spearman matrix."""
    merged = runs + soft_runs
    rows = []
    for grp in INTENTS + ["all"]:
        row = {"Intent": grp}
        for m in RECALL_METRICS:
            per_run = []
            for r in merged:
                if m not in r.columns:
                    continue
                sub = r if grp == "all" else r[r["intent"] == grp]
                vals = sub[m].dropna()
                if len(vals):
                    per_run.append(vals.mean())
            if per_run:
                row[m] = float(np.mean(per_run))
                row[m + "_sd"] = float(np.std(per_run, ddof=1)) if len(per_run) > 1 else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)

    # Spearman between metrics across cases (per-case means).
    series = {m: per_case_mean(merged, m) for m in RECALL_METRICS}
    series = {m: s for m, s in series.items() if len(s)}
    corr_rows = []
    for m1, m2 in itertools.combinations(series.keys(), 2):
        common = series[m1].index.intersection(series[m2].index)
        if len(common) < 10 or _stats is None:
            continue
        rho, p = _stats.spearmanr(series[m1].loc[common], series[m2].loc[common])
        corr_rows.append({"metric_a": m1, "metric_b": m2, "n": int(len(common)),
                          "spearman_rho": float(rho), "p": float(p)})
    return table, pd.DataFrame(corr_rows)


# ── Rendering ──────────────────────────────────────────────────


def cell(mean: float, sd: float, digits: int = 3) -> str:
    if mean != mean:
        return "–"
    if sd != sd:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def render_system_markdown(tab: pd.DataFrame, metrics: list[str]) -> str:
    cols = [m for m in metrics if m in tab.columns]
    head = "| System | Intent | Runs | " + " | ".join(PRETTY.get(m, m) for m in cols) + " |"
    sep = "|" + "---|" * (3 + len(cols))
    lines = [head, sep]
    for _, r in tab.iterrows():
        lines.append(f"| {r['System']} | {r['Intent']} | {r['Runs']} | "
                     + " | ".join(cell(r.get(m, float('nan')), r.get(m + '_sd', float('nan'))) for m in cols) + " |")
    return "\n".join(lines)


def render_system_latex(tab: pd.DataFrame, metrics: list[str], caption: str, label: str) -> str:
    cols = [m for m in metrics if m in tab.columns]
    out = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{ll" + "c" * len(cols) + "}", "\\toprule",
           "System & Intent & " + " & ".join(PRETTY.get(m, m) for m in cols) + " \\\\", "\\midrule"]
    for _, r in tab.iterrows():
        out.append(f"{r['System']} & {r['Intent'].replace('_', ' ')} & "
                   + " & ".join(cell(r.get(m, float('nan')), r.get(m + '_sd', float('nan'))).replace("±", "$\\pm$")
                                for m in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out)


def render_tests_markdown(tests: pd.DataFrame) -> str:
    if tests.empty:
        return "(no paired tests: need a baseline with overlapping case ids)"
    lines = ["| System | Metric | n | Δ mean | 95% CI | Wilcoxon p | W/L/T |", "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        lines.append(f"| {r['System']} vs {r['vs']} | {PRETTY.get(r['Metric'], r['Metric'])} | {r['n_cases']} | "
                     f"{r['mean_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {fmt_p(r['wilcoxon_p'])} | "
                     f"{r['wins']}/{r['losses']}/{r['ties']} |")
    return "\n".join(lines)


def render_metric_markdown(tab: pd.DataFrame, corr: pd.DataFrame, label: str) -> str:
    cols = [m for m in RECALL_METRICS if m in tab.columns]
    lines = [f"System: {label}", "",
             "| Intent | " + " | ".join(PRETTY.get(m, m) for m in cols) + " |", "|" + "---|" * (1 + len(cols))]
    for _, r in tab.iterrows():
        lines.append(f"| {r['Intent']} | " + " | ".join(cell(r.get(m, float('nan')), r.get(m + '_sd', float('nan'))) for m in cols) + " |")
    if not corr.empty:
        lines += ["", "Spearman ρ between metrics across cases:", "",
                  "| A | B | n | ρ | p |", "|---|---|---|---|---|"]
        for _, r in corr.iterrows():
            lines.append(f"| {PRETTY.get(r['metric_a'], r['metric_a'])} | {PRETTY.get(r['metric_b'], r['metric_b'])} | "
                         f"{r['n']} | {r['spearman_rho']:.3f} | {fmt_p(r['p'])} |")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate repeated evaluation runs into tables.")
    p.add_argument("--run", nargs="+", action="append", metavar=("LABEL", "CSV"), required=True,
                   help="System label followed by one or more result CSVs (one per run).")
    p.add_argument("--soft-file", nargs="+", type=Path, default=[],
                   help="CSV(s) produced with --soft (context_recall column holds retrieval_recall_soft).")
    p.add_argument("--soft-label", default=None, help="Which --run label the soft files belong to.")
    p.add_argument("--baseline", default=None, help="Label for paired tests. Default: first --run.")
    p.add_argument("--metric-table", default=None, help="Label whose recall metrics are compared side by side.")
    p.add_argument("--metrics", nargs="+", default=SYSTEM_METRICS)
    p.add_argument("--out", type=Path, default=Path("evaluation/results/tables"))
    args = p.parse_args()

    runs_by_label: dict[str, list[pd.DataFrame]] = {}
    for spec in args.run:
        label, *files = spec
        if not files:
            raise SystemExit(f"--run {label}: no CSV given")
        runs_by_label[label] = [load_run(Path(f)) for f in files]
    soft_runs = [load_run(Path(f), soft=True) for f in args.soft_file]
    soft_label = args.soft_label or (args.metric_table or next(iter(runs_by_label)))

    baseline = args.baseline or next(iter(runs_by_label))
    metrics = [m for m in args.metrics
               if any(m in r.columns and r[m].notna().any() for runs in runs_by_label.values() for r in runs)]

    args.out.mkdir(parents=True, exist_ok=True)

    # Table S: systems
    overall = system_table(runs_by_label, metrics, by_intent=False)
    by_intent = system_table(runs_by_label, metrics, by_intent=True)
    overall.to_csv(args.out / "table_systems_overall.csv", index=False)
    by_intent.to_csv(args.out / "table_systems_by_intent.csv", index=False)
    (args.out / "table_systems_overall.tex").write_text(
        render_system_latex(overall, metrics, "System comparison, mean $\\pm$ SD across runs (same metrics for every row).",
                            "tab:systems-overall"), encoding="utf-8")
    (args.out / "table_systems_by_intent.tex").write_text(
        render_system_latex(by_intent, metrics, "System comparison by intent, mean $\\pm$ SD across runs.",
                            "tab:systems-intent"), encoding="utf-8")

    tests = paired_tests(runs_by_label, baseline, metrics) if len(runs_by_label) > 1 else pd.DataFrame()
    if not tests.empty:
        tests.to_csv(args.out / "paired_tests.csv", index=False)

    md = ["# System table (overall)", "", render_system_markdown(overall, metrics), "",
          "# System table (by intent)", "", render_system_markdown(by_intent, metrics), "",
          f"# Paired tests vs {baseline} (per-case means across runs, bootstrap 95% CI, Wilcoxon signed-rank)", "",
          render_tests_markdown(tests), ""]

    # Table M: metrics on one system
    if args.metric_table:
        if args.metric_table not in runs_by_label:
            raise SystemExit(f"--metric-table {args.metric_table}: unknown label")
        mt, corr = metric_table(runs_by_label[args.metric_table],
                                soft_runs if soft_label == args.metric_table else [])
        mt.to_csv(args.out / "table_metrics.csv", index=False)
        corr.to_csv(args.out / "metric_correlations.csv", index=False)
        md += ["# Metric table (one system, several recall definitions)", "",
               render_metric_markdown(mt, corr, args.metric_table), ""]
        tex_cols = [m for m in RECALL_METRICS if m in mt.columns]
        tex = ["\\begin{table}[t]", "\\centering",
               f"\\caption{{Recall definitions on the same system ({args.metric_table}). A higher number here is a different ruler, not a better system.}}",
               "\\label{tab:metrics}", "\\begin{tabular}{l" + "c" * len(tex_cols) + "}", "\\toprule",
               "Intent & " + " & ".join(PRETTY.get(m, m) for m in tex_cols) + " \\\\", "\\midrule"]
        for _, r in mt.iterrows():
            tex.append(f"{r['Intent'].replace('_', ' ')} & " + " & ".join(
                cell(r.get(m, float('nan')), r.get(m + '_sd', float('nan'))).replace("±", "$\\pm$") for m in tex_cols) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (args.out / "table_metrics.tex").write_text("\n".join(tex), encoding="utf-8")

    report = "\n".join(md)
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote tables to {args.out}/")


if __name__ == "__main__":
    main()
