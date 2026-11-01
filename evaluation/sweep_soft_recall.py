"""Sensitivity sweep for the soft recall thresholds, offline after one embedding pass.

Reads the ``*_contexts.jsonl`` dump that ``ragas_eval.py`` writes, embeds every
retrieved and reference context once (cached on disk, so the sweep itself costs
no API calls), then recomputes Soft Recall@k for a grid of
(sim_low, sim_high, partial_cap, token_overlap_min). When human annotations are
given, it also reports the Spearman correlation of every setting with the
human coverage label, next to the same correlation for Recall@k and Ragas
context_recall, which is the comparison a reviewer will ask for.

Usage:
    python -m evaluation.sweep_soft_recall \
        --contexts evaluation/results/v6/bge_rrf_r1_contexts.jsonl \
        --results  evaluation/results/v6/bge_rrf_r1.csv \
        --cache    evaluation/results/v6/embedding_cache.json \
        --annotations evaluation/annotation/labels_*.json \
        --out evaluation/results/v6/sweep
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from evaluation.ragas_eval import (  # noqa: E402
    _l2_normalize_rows,
    _partial_credit,
    _token_overlap,
    _truncate_for_embed,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None

DEFAULT_GRID = {
    "sim_low": [0.78, 0.82, 0.84, 0.86, 0.90],
    "sim_high": [0.94, 0.96, 0.98],
    "partial_cap": [0.0, 0.25, 0.5, 0.75, 1.0],
    "token_overlap_min": [0.0, 0.08, 0.12, 0.2],
}
CURRENT = {"sim_low": 0.84, "sim_high": 0.96, "partial_cap": 0.75, "token_overlap_min": 0.12}


# ── Embedding cache ────────────────────────────────────────────


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path, model: str):
        self.path = path
        self.model = model
        self.vectors: dict[str, list[float]] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") == model:
                self.vectors = data.get("vectors", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model, "vectors": self.vectors}), encoding="utf-8")

    async def embed(self, texts: list[str], batch: int = 64) -> None:
        missing = sorted({t for t in texts if _key(t) not in self.vectors})
        if not missing:
            return
        print(f"  embedding {len(missing)} new texts with {self.model} ...")
        if "/" in self.model:
            # HuggingFace model id (e.g. BAAI/bge-m3): run locally, no API key needed.
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(self.model)
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                vecs = st.encode([_truncate_for_embed(t) for t in chunk], normalize_embeddings=True,
                                 batch_size=min(batch, 32), show_progress_bar=False)
                for t, v in zip(chunk, vecs):
                    self.vectors[_key(t)] = [float(x) for x in v]
                if (i // batch) % 10 == 0:
                    self.save()
            self.save()
            return
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        for i in range(0, len(missing), batch):
            chunk = missing[i:i + batch]
            resp = await client.embeddings.create(model=self.model, input=[_truncate_for_embed(t) for t in chunk])
            for t, d in zip(chunk, resp.data):
                self.vectors[_key(t)] = d.embedding
            if (i // batch) % 10 == 0:
                self.save()
        self.save()

    def matrix(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize_rows(np.asarray([self.vectors[_key(t)] for t in texts], dtype=np.float64))


# ── Per-case features (computed once) ──────────────────────────


def case_features(entry: dict[str, Any], cache: EmbeddingCache, k: int) -> dict[str, Any] | None:
    refs = [str(c) for c in entry.get("reference_contexts", []) if str(c).strip()]
    ctxs = [str(c) for c in entry.get("retrieved_contexts", []) if str(c).strip()]
    ids = [str(i) for i in entry.get("retrieved_ids", [])]
    relevant = {str(r) for r in entry.get("relevant_ids", []) if r}
    excluded = {str(e) for e in entry.get("exclude_ids", []) if e}
    if not refs or not relevant:
        return None
    pairs = list(zip(ctxs, ids)) if len(ids) >= len(ctxs) else [(c, "") for c in ctxs]
    seen: set[str] = set()
    ranked = []
    for c, rid in pairs:
        if rid and (rid in excluded or rid in seen):
            continue
        if rid:
            seen.add(rid)
        ranked.append((c, rid))
    ranked = ranked[:k]
    hit = np.array([1.0 if rid in relevant else 0.0 for _, rid in ranked])
    max_sim = np.zeros(len(ranked))
    overlap = np.zeros(len(ranked))
    misses = [i for i, h in enumerate(hit) if h == 0.0]
    if misses:
        a = cache.matrix([ranked[i][0] for i in misses])
        b = cache.matrix(refs)
        sims = a @ b.T
        for row, i in enumerate(misses):
            best = int(np.argmax(sims[row]))
            max_sim[i] = float(sims[row, best])
            overlap[i] = _token_overlap(ranked[i][0], refs[best])
    return {
        "id": entry["id"], "intent": entry.get("intent", ""), "hit": hit, "max_sim": max_sim,
        "overlap": overlap, "denom": max(1, min(len(relevant), k)),
        "recall_at_k": float(hit.sum() / max(1, min(len(relevant), k))),
    }


def soft_recall(feat: dict[str, Any], sim_low: float, sim_high: float, partial_cap: float,
                token_overlap_min: float) -> float:
    credits = feat["hit"].copy()
    for i in np.where(feat["hit"] == 0.0)[0]:
        credits[i] = _partial_credit(feat["max_sim"][i], feat["overlap"][i], sim_low, sim_high,
                                     partial_cap, token_overlap_min)
    return float(min(1.0, credits.sum() / feat["denom"]))


# ── Annotations ────────────────────────────────────────────────


def load_coverage(paths: list[str]) -> pd.Series:
    """Mean human coverage label (0..2) per case across annotators."""
    frames = []
    for pattern in paths:
        for f in glob.glob(pattern):
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            for cid, lab in data.get("labels", {}).items():
                if lab.get("coverage") is not None:
                    frames.append({"id": cid, "coverage": float(lab["coverage"]), "annotator": data.get("annotator", f)})
    if not frames:
        return pd.Series(dtype=float)
    df = pd.DataFrame(frames)
    return df.groupby("id")["coverage"].mean()


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    common = a.index.intersection(b.index)
    if len(common) < 10 or _stats is None:
        return float("nan"), int(len(common))
    rho, _ = _stats.spearmanr(a.loc[common], b.loc[common])
    return float(rho), int(len(common))


# ── Main ───────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> None:
    entries = [json.loads(l) for l in Path(args.contexts).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(entries)} cases from {args.contexts}")
    cache = EmbeddingCache(args.cache, args.embedding_model)
    texts = []
    for e in entries:
        texts += [str(c) for c in e.get("retrieved_contexts", [])]
        texts += [str(c) for c in e.get("reference_contexts", [])]
    await cache.embed(texts)

    feats = [f for f in (case_features(e, cache, args.k) for e in entries) if f is not None]
    print(f"Features for {len(feats)} cases with relevant_ids")
    intents = sorted({f["intent"] for f in feats})

    coverage = load_coverage(args.annotations) if args.annotations else pd.Series(dtype=float)
    ragas_cr = pd.Series(dtype=float)
    if args.results and Path(args.results).exists():
        res = pd.read_csv(args.results)
        if "context_recall" in res.columns:
            ragas_cr = pd.to_numeric(res.set_index("id")["context_recall"], errors="coerce").dropna()

    recall_k = pd.Series({f["id"]: f["recall_at_k"] for f in feats})
    rows = []
    grid = itertools.product(args.sim_low, args.sim_high, args.partial_cap, args.token_overlap_min)
    for lo, hi, cap, gate in grid:
        if hi <= lo:
            continue
        vals = pd.Series({f["id"]: soft_recall(f, lo, hi, cap, gate) for f in feats})
        row = {"sim_low": lo, "sim_high": hi, "partial_cap": cap, "token_overlap_min": gate,
               "mean": float(vals.mean())}
        for it in intents:
            ids = [f["id"] for f in feats if f["intent"] == it]
            row[f"mean_{it}"] = float(vals.loc[ids].mean())
        row["rho_vs_recall_at_k"], _ = spearman(vals, recall_k)
        if len(coverage):
            row["rho_vs_human"], row["n_human"] = spearman(vals, coverage)
        rows.append(row)
    sweep = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out / "sweep_grid.csv", index=False)

    cur = sweep[(sweep.sim_low == CURRENT["sim_low"]) & (sweep.sim_high == CURRENT["sim_high"])
                & (sweep.partial_cap == CURRENT["partial_cap"]) & (sweep.token_overlap_min == CURRENT["token_overlap_min"])]
    print("\nCurrent setting:", cur.to_dict("records")[0] if len(cur) else "(not in grid)")
    print(f"Soft recall mean range over grid: {sweep['mean'].min():.3f} .. {sweep['mean'].max():.3f}")
    print(f"Recall@k mean: {recall_k.mean():.3f}")

    lines = ["# Soft recall threshold sweep", "",
             f"Cases: {len(feats)}  |  k = {args.k}  |  Recall@k mean = {recall_k.mean():.3f}", ""]
    if len(coverage):
        rho_rk, n = spearman(recall_k, coverage)
        lines.append(f"Spearman vs human coverage (n={n}): Recall@k ρ = {rho_rk:.3f}")
        if len(ragas_cr):
            rho_cr, n2 = spearman(ragas_cr, coverage)
            lines.append(f"Spearman vs human coverage (n={n2}): Ragas context_recall ρ = {rho_cr:.3f}")
        best = sweep.sort_values("rho_vs_human", ascending=False).head(10)
        lines += ["", "Top 10 settings by ρ vs human:", "", best.to_markdown(index=False, floatfmt=".3f")]
        if len(cur):
            lines += ["", "Current setting:", "", cur.to_markdown(index=False, floatfmt=".3f")]
    else:
        lines += ["No annotations given: only the sensitivity of the mean is reported.", "",
                  "Mean soft recall by partial_cap and token gate (sim_low=0.84, sim_high=0.96):", ""]
        piv = sweep[(sweep.sim_low == 0.84) & (sweep.sim_high == 0.96)].pivot(
            index="partial_cap", columns="token_overlap_min", values="mean")
        lines.append(piv.round(3).to_markdown())
    (args.out / "SWEEP.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        sub = sweep[(sweep.sim_low == 0.84) & (sweep.sim_high == 0.96)]
        for gate, g in sub.groupby("token_overlap_min"):
            ax.plot(g["partial_cap"], g["mean"], marker="o", label=f"token gate {gate}")
        ax.axhline(recall_k.mean(), ls="--", color="gray", label="Recall@k (cap = 0)")
        ax.set_xlabel("partial_cap")
        ax.set_ylabel("mean soft recall")
        ax.set_title("Sensitivity of soft recall to its thresholds")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "sweep_sensitivity.png", dpi=150)
        if len(coverage):
            fig, ax = plt.subplots(figsize=(6, 4))
            for gate, g in sub.groupby("token_overlap_min"):
                ax.plot(g["partial_cap"], g["rho_vs_human"], marker="o", label=f"token gate {gate}")
            ax.set_xlabel("partial_cap")
            ax.set_ylabel("Spearman ρ vs human coverage")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out / "sweep_vs_human.png", dpi=150)
        print(f"Wrote plots to {args.out}/")
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep soft recall thresholds offline.")
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--results", type=Path, default=None, help="Result CSV of the same run (for Ragas context_recall).")
    p.add_argument("--cache", type=Path, default=ROOT / "evaluation" / "results" / "embedding_cache.json")
    p.add_argument("--embedding-model", default=os.getenv("RAGAS_AR_EMBEDDING_MODEL", "BAAI/bge-m3"),
                   help="Same model ragas_eval used for soft recall. A HuggingFace id runs locally; "
                        "an OpenAI id (text-embedding-3-small) needs OPENAI_API_KEY.")
    p.add_argument("--annotations", nargs="*", default=None, help="Glob(s) of annotation JSON files.")
    p.add_argument("--k", type=int, default=int(os.getenv("TOP_K_BEANS", "5")))
    p.add_argument("--sim-low", nargs="+", type=float, default=DEFAULT_GRID["sim_low"])
    p.add_argument("--sim-high", nargs="+", type=float, default=DEFAULT_GRID["sim_high"])
    p.add_argument("--partial-cap", nargs="+", type=float, default=DEFAULT_GRID["partial_cap"])
    p.add_argument("--token-overlap-min", nargs="+", type=float, default=DEFAULT_GRID["token_overlap_min"])
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results" / "sweep")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
