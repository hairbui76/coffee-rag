"""Inter-annotator agreement and metric-vs-human correlation.

Inputs: two or more ``labels_<annotator>.json`` files exported from
``annotate.html``, the result CSV of the run the sample was drawn from (Ragas
context_precision, context_recall, hit_at_k, recall_at_k, precision_at_k, mrr) and,
optionally, the matching ``--soft`` run CSV (retrieval_recall_soft).

Outputs (``--out``):
  agreement.csv      Cohen's κ (unweighted and quadratic-weighted) for the
                     per-context relevance label and the case-level coverage label
  correlations.csv   Spearman ρ and Kendall τ of every automatic metric against
                     the mean human coverage (recall-type) and the mean human
                     per-context relevance (precision-type), overall and by intent
  REPORT.md          both tables in markdown

Usage:
    python -m evaluation.annotation.analyze_annotations \
        --labels evaluation/annotation/labels_*.json \
        --results evaluation/results/v6/bge_rrf_r1.csv \
        --soft-results evaluation/results/v6/bge_rrf_soft_r1.csv \
        --out evaluation/annotation/report
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    warnings.filterwarnings("ignore", category=_stats.ConstantInputWarning)
except ImportError:  # pragma: no cover
    _stats = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

METRICS = ["context_recall", "retrieval_recall_soft", "recall_at_k", "hit_at_k", "mrr",
           "context_precision", "precision_at_k"]
PRETTY = {
    "context_recall": "Context Recall (Ragas)", "retrieval_recall_soft": "Soft Recall@k",
    "recall_at_k": "Recall@k", "hit_at_k": "Hit@k", "mrr": "MRR",
    "context_precision": "Context Precision (Ragas)", "precision_at_k": "Precision@k",
}


# ── Agreement ──────────────────────────────────────────────────


def cohen_kappa(a: np.ndarray, b: np.ndarray, categories: list[int], weighted: bool) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[int(x)], idx[int(y)]] += 1
    obs /= n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weighted:
        w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    po = (w * obs).sum()
    pe = (w * exp).sum()
    return float(1.0 - po / pe) if pe > 0 else float("nan")


def load_labels(patterns: list[str]) -> dict[str, dict]:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        out[d.get("annotator") or Path(f).stem] = d
    return out


def agreement_table(labels: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(labels.keys(), 2):
        la, lb = labels[a]["labels"], labels[b]["labels"]
        ids = [i for i in la if i in lb and la[i].get("coverage") is not None and lb[i].get("coverage") is not None]
        cov_a = np.array([la[i]["coverage"] for i in ids])
        cov_b = np.array([lb[i]["coverage"] for i in ids])
        ctx_a, ctx_b = [], []
        for i in ids:
            for x, y in zip(la[i].get("contexts", []), lb[i].get("contexts", [])):
                if x is not None and y is not None:
                    ctx_a.append(x)
                    ctx_b.append(y)
        ctx_a, ctx_b = np.array(ctx_a), np.array(ctx_b)
        rows.append({
            "annotator_a": a, "annotator_b": b, "n_cases": len(ids), "n_contexts": len(ctx_a),
            "kappa_coverage": cohen_kappa(cov_a, cov_b, [0, 1, 2], False),
            "kappa_w_coverage": cohen_kappa(cov_a, cov_b, [0, 1, 2], True),
            "kappa_context": cohen_kappa(ctx_a, ctx_b, [0, 1, 2], False),
            "kappa_w_context": cohen_kappa(ctx_a, ctx_b, [0, 1, 2], True),
            "exact_agree_coverage": float((cov_a == cov_b).mean()) if len(ids) else float("nan"),
            "exact_agree_context": float((ctx_a == ctx_b).mean()) if len(ctx_a) else float("nan"),
        })
    return pd.DataFrame(rows)


def human_scores(labels: dict[str, dict]) -> pd.DataFrame:
    """Mean coverage (0..2) and mean per-context relevance (0..2) per case across annotators."""
    rows = []
    for ann, d in labels.items():
        for cid, lab in d["labels"].items():
            ctx = [v for v in lab.get("contexts", []) if v is not None]
            rows.append({"id": cid, "annotator": ann, "coverage": lab.get("coverage"),
                         "relevance_mean": float(np.mean(ctx)) if ctx else np.nan,
                         "relevance_top1": ctx[0] if ctx else np.nan})
    df = pd.DataFrame(rows).dropna(subset=["coverage"])
    return df.groupby("id")[["coverage", "relevance_mean", "relevance_top1"]].mean()


# ── Correlation ────────────────────────────────────────────────


def corr_table(human: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = human.join(auto, how="inner")
    groups = [("all", merged)] + [(it, g) for it, g in merged.groupby("intent")] if "intent" in merged else [("all", merged)]
    for grp, g in groups:
        for m in METRICS:
            if m not in g.columns:
                continue
            for target, tname in [("coverage", "human coverage"), ("relevance_mean", "human precision")]:
                sub = g[[m, target]].dropna()
                if len(sub) < 10 or _stats is None:
                    continue
                rho, p_rho = _stats.spearmanr(sub[m], sub[target])
                tau, p_tau = _stats.kendalltau(sub[m], sub[target])
                rows.append({"intent": grp, "metric": m, "target": tname, "n": int(len(sub)),
                             "spearman_rho": float(rho), "p_rho": float(p_rho),
                             "kendall_tau": float(tau), "p_tau": float(p_tau)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", required=True, help="Glob(s) of labels_*.json files.")
    p.add_argument("--results", type=Path, required=True, help="Result CSV of the sampled run.")
    p.add_argument("--soft-results", type=Path, default=None, help="Matching --soft run CSV.")
    p.add_argument("--out", type=Path, default=Path("evaluation/annotation/report"))
    args = p.parse_args()

    labels = load_labels(args.labels)
    print(f"Annotators: {list(labels)}")
    agree = agreement_table(labels)
    human = human_scores(labels)

    res = pd.read_csv(args.results)
    keep = [c for c in ["id", "intent", "difficulty", "language"] + METRICS if c in res.columns]
    auto = res[keep].copy()
    if args.soft_results and args.soft_results.exists():
        soft = pd.read_csv(args.soft_results)
        col = "retrieval_recall_soft" if "retrieval_recall_soft" in soft.columns else "context_recall"
        auto = auto.merge(soft[["id", col]].rename(columns={col: "retrieval_recall_soft"}), on="id", how="left")
    for m in METRICS:
        if m in auto.columns:
            auto[m] = pd.to_numeric(auto[m], errors="coerce")
    auto = auto.set_index("id")
    corr = corr_table(human, auto)

    args.out.mkdir(parents=True, exist_ok=True)
    agree.to_csv(args.out / "agreement.csv", index=False)
    corr.to_csv(args.out / "correlations.csv", index=False)
    human.to_csv(args.out / "human_scores.csv")

    md = [f"# Annotation report ({len(human)} cases, annotators: {', '.join(labels)})", "",
          "## Inter-annotator agreement (Cohen's κ; _w = quadratic weights)", ""]
    md.append(agree.round(3).to_markdown(index=False) if not agree.empty else "(one annotator only: no κ)")
    md += ["", "## Metric vs human (Spearman ρ / Kendall τ)", ""]
    if not corr.empty:
        for target in ["human coverage", "human precision"]:
            sub = corr[corr.target == target]
            if sub.empty:
                continue
            piv = sub.pivot(index="metric", columns="intent", values="spearman_rho").round(3)
            piv.index = [PRETTY.get(m, m) for m in piv.index]
            md += [f"### Spearman ρ vs {target}", "", piv.to_markdown(), ""]
        md += ["Full table with p-values and Kendall τ: correlations.csv", ""]
    report = "\n".join(md)
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
