"""Evaluate the Coffee RAG pipeline with Ragas.

Example:
    python -m evaluation.ragas_eval --limit 5
    python -m evaluation.ragas_eval --mode retrieval --limit 20 --intent product_search
    python -m evaluation.ragas_eval --mode retrieval --limit 50 --workers 8 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.generation.llm_client import generate_structured  # noqa: E402
from src.generation.prompt_templates import build_prompt  # noqa: E402
from src.generation.schemas import CoffeeResponse  # noqa: E402
from src.pipeline import CoffeeRAG  # noqa: E402


DEFAULT_METRICS = {
    "full": ["faithfulness", "context_precision", "context_recall", "answer_relevancy"],
    "retrieval": ["context_precision", "context_recall"],
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _bean_contexts(beans_df) -> list[str]:
    if beans_df is None or beans_df.empty:
        return []

    contexts: list[str] = []
    for _, row in beans_df.iterrows():
        contexts.append(
            "Bean: {name}. Roaster: {roaster}. Origin: {origin}. Country: {country}. "
            "Roast: {roast}. Flavor: {flavor}. Processing: {processing}. "
            "Species: {species}. Description: {description}. URL: {url}".format(
                name=_as_str(row.get("product_name")),
                roaster=_as_str(row.get("roaster_name")),
                origin=_as_str(row.get("origin")),
                country=_as_str(row.get("country")),
                roast=_as_str(row.get("roast_level_clean") or row.get("roast_level")),
                flavor=", ".join(map(str, _as_list(row.get("flavor_notes_clean")))),
                processing=", ".join(map(str, _as_list(row.get("processing_clean")))),
                species=", ".join(map(str, _as_list(row.get("species")))),
                description=_as_str(row.get("about_description")),
                url=_as_str(row.get("product_url")),
            )
        )
    return contexts


def _news_contexts(news_df) -> list[str]:
    if news_df is None or news_df.empty:
        return []

    contexts: list[str] = []
    for _, row in news_df.iterrows():
        contexts.append(
            "Article: {title}. Source: {source}. Date: {date}. Content: {content}. URL: {url}".format(
                title=_as_str(row.get("title")),
                source=_as_str(row.get("source")),
                date=_as_str(row.get("publish_datetime")),
                content=_as_str(row.get("text") or row.get("summary") or row.get("content_text")),
                url=_as_str(row.get("article_url")),
            )
        )
    return contexts


def build_retrieved_contexts(ctx: dict[str, Any]) -> list[str]:
    return _bean_contexts(ctx.get("beans")) + _news_contexts(ctx.get("news"))


def response_to_text(response: CoffeeResponse | None) -> str:
    if response is None:
        return ""
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), ensure_ascii=False)
    return str(response)


def load_existing_ids(csv_path: Path) -> set[str]:
    """Return the set of case IDs already present in the results CSV."""
    if not csv_path.exists():
        return set()
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["id"] for row in reader if row.get("id")}
    except Exception:
        return set()


def load_cases(path: Path, limit: int | None, offset: int, intent: str | None, language: str | None,
               skip_ids: set[str] | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        cases = json.load(f)

    if intent:
        cases = [case for case in cases if case.get("intent") == intent]
    if language:
        cases = [case for case in cases if case.get("language") == language]
    if skip_ids:
        before = len(cases)
        cases = [case for case in cases if case.get("id", "") not in skip_ids]
        skipped = before - len(cases)
        if skipped:
            print(f"Skipping {skipped} already-evaluated cases (found in results CSV).")
    if offset:
        cases = cases[offset:]
    if limit is not None and limit > 0:
        cases = cases[:limit]
    return cases


def build_metrics(names: list[str], evaluator_model: str, embedding_model: str):
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = llm_factory(evaluator_model, client=client)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)

    available = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }
    return {name: available[name] for name in names}


# ── Async metric scoring ─────────────────────────────────────

async def async_score_metric(metric_name: str, metric: Any, sample: dict[str, Any]) -> float | None:
    if metric_name == "faithfulness":
        result = await metric.ascore(
            user_input=sample["question"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    elif metric_name == "context_precision":
        result = await metric.ascore(
            user_input=sample["question"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    elif metric_name == "context_recall":
        result = await metric.ascore(
            user_input=sample["question"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    elif metric_name == "answer_relevancy":
        result = await metric.ascore(
            user_input=sample["question"],
            response=sample["response"],
        )
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    value = getattr(result, "value", result)
    return None if value is None else float(value)


async def score_all_metrics(metrics: dict[str, Any], sample: dict[str, Any]) -> dict[str, float | str]:
    """Score all metrics for one sample concurrently."""
    results: dict[str, float | str] = {}
    errors: list[str] = []

    async def _run(name: str, metric: Any):
        try:
            results[name] = await async_score_metric(name, metric, sample)
        except Exception as exc:
            results[name] = ""
            errors.append(f"{name}: {exc}")

    await asyncio.gather(*[_run(name, m) for name, m in metrics.items()])
    results["error"] = " | ".join(errors) if errors else ""
    return results


# ── Per-sample processing ─────────────────────────────────────

MAX_EVAL_CONTEXTS = 20


def retrieve_one(rag: CoffeeRAG, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run retrieval + optional generation for a single case. Returns a sample dict."""
    question = case["question"]
    ctx = rag.retrieve(question, top_k_beans=args.top_k_beans, top_k_news=args.top_k_news)

    beans = ctx.get("beans")
    news = ctx.get("news")
    bean_count = len(beans) if beans is not None and not getattr(beans, "empty", True) else 0
    news_count = len(news) if news is not None and not getattr(news, "empty", True) else 0

    eval_ctx = dict(ctx)
    if beans is not None and len(beans) > MAX_EVAL_CONTEXTS:
        eval_ctx["beans"] = beans.head(MAX_EVAL_CONTEXTS)
    if news is not None and len(news) > MAX_EVAL_CONTEXTS:
        eval_ctx["news"] = news.head(MAX_EVAL_CONTEXTS)
    retrieved_contexts = build_retrieved_contexts(eval_ctx)

    response = None
    if args.mode == "full":
        messages = build_prompt(question, ctx["beans"], ctx["news"])
        response = generate_structured(messages, CoffeeResponse, client=rag.llm_client)

    return {
        "question": question,
        "reference": case["ground_truth"],
        "response": response_to_text(response),
        "retrieved_contexts": retrieved_contexts,
        "intent": ctx.get("intent", ""),
        "entities": ctx.get("entities", {}),
        "bean_count": bean_count,
        "news_count": news_count,
    }


# ── Logging helpers ───────────────────────────────────────────

class RunningStats:
    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, scores: dict[str, float | str]):
        for name, value in scores.items():
            if name == "error" or value == "" or value is None:
                continue
            self.sums[name] += float(value)
            self.counts[name] += 1

    def averages(self) -> dict[str, float]:
        return {name: self.sums[name] / self.counts[name] for name in sorted(self.sums) if self.counts[name]}


def log_sample(
    index: int,
    total: int,
    case: dict[str, Any],
    sample: dict[str, Any],
    scores: dict[str, float | str],
    elapsed: float,
    verbose: bool,
    stats: RunningStats,
):
    score_parts = []
    for name in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
        val = scores.get(name)
        if val not in ("", None):
            score_parts.append(f"{name}={val:.3f}")

    score_str = "  ".join(score_parts) if score_parts else "no scores"
    tag = f"[{index}/{total}]"
    id_str = case.get("id", "")
    intent = case.get("intent", "")
    question = case["question"][:80]

    print(f"{tag} {id_str} ({intent}) {elapsed:.1f}s  {score_str}")

    if verbose:
        print(f"     question : {case['question'][:120]}")
        print(f"     beans={sample['bean_count']}  news={sample['news_count']}  contexts={len(sample['retrieved_contexts'])}")
        entities = sample.get("entities", {})
        if entities:
            ent_parts = [f"{k}={v}" for k, v in entities.items() if v]
            if ent_parts:
                print(f"     entities : {', '.join(ent_parts)}")
        if scores.get("error"):
            print(f"     ERROR    : {scores['error']}")

        avgs = stats.averages()
        if avgs:
            avg_parts = [f"{name}={v:.3f}" for name, v in avgs.items()]
            print(f"     running  : {', '.join(avg_parts)}")
        print()


# ── Main eval loop ────────────────────────────────────────────

async def run_eval_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for Ragas evaluator LLM and embeddings.")

    metric_names = args.metrics or DEFAULT_METRICS[args.mode]
    unsupported = sorted(set(metric_names) - {"faithfulness", "context_precision", "context_recall", "answer_relevancy"})
    if unsupported:
        raise ValueError(f"Unsupported metrics: {', '.join(unsupported)}")
    if args.mode == "retrieval":
        metric_names = [name for name in metric_names if name in {"context_precision", "context_recall"}]

    skip_ids = load_existing_ids(args.out) if args.limit == 0 else None
    cases = load_cases(args.dataset, args.limit, args.offset, args.intent, args.language, skip_ids=skip_ids)
    if not cases:
        print("No cases matched filters.")
        return []

    metrics = build_metrics(metric_names, args.evaluator_model, args.embedding_model)
    rag = CoffeeRAG()
    total = len(cases)
    stats = RunningStats()
    semaphore = asyncio.Semaphore(args.workers)
    rows: list[dict[str, Any]] = [None] * total  # type: ignore[list-item]
    t_start = time.perf_counter()

    ZERO_PRECISION_STOP_RATIO = 0.2
    stop_event = asyncio.Event()
    completed_count = 0
    zero_precision_count = 0

    print(f"\n{'=' * 60}")
    print(f"  Ragas evaluation  |  mode={args.mode}  metrics={metric_names}")
    print(f"  cases={total}  workers={args.workers}  evaluator={args.evaluator_model}")
    print(f"  early stop: context_precision=0 >= {ZERO_PRECISION_STOP_RATIO:.0%} of cases")
    print(f"{'=' * 60}\n")

    async def process_one(index: int, case: dict[str, Any]):
        nonlocal completed_count, zero_precision_count

        if stop_event.is_set():
            return

        async with semaphore:
            if stop_event.is_set():
                return

            t0 = time.perf_counter()

            sample = await asyncio.to_thread(retrieve_one, rag, case, args)
            scores = await score_all_metrics(metrics, sample)

            elapsed = time.perf_counter() - t0
            stats.update(scores)

            row = {
                "id": case.get("id", ""),
                "intent": case.get("intent", ""),
                "difficulty": case.get("difficulty", ""),
                "language": case.get("language", ""),
                "question": case["question"],
                "retrieved_context_count": len(sample["retrieved_contexts"]),
                "bean_count": sample["bean_count"],
                "news_count": sample["news_count"],
                "response": sample["response"],
                "reference": sample["reference"],
                **{name: scores.get(name, "") for name in metric_names},
                "error": scores.get("error", ""),
                "elapsed_s": round(elapsed, 1),
            }
            rows[index] = row

            log_sample(index + 1, total, case, sample, scores, elapsed, args.verbose, stats)

            completed_count += 1
            cp = scores.get("context_precision")
            if cp is not None and cp != "" and float(cp) == 0.0:
                zero_precision_count += 1
            if completed_count >= 3 and zero_precision_count / completed_count >= ZERO_PRECISION_STOP_RATIO:
                print(f"\n{'!' * 60}")
                print(f"  EARLY STOP: {zero_precision_count}/{completed_count} cases "
                      f"({zero_precision_count/completed_count:.0%}) have context_precision=0")
                print(f"  Threshold: {ZERO_PRECISION_STOP_RATIO:.0%} — stopping evaluation.")
                print(f"{'!' * 60}\n")
                stop_event.set()

    await asyncio.gather(*[process_one(i, c) for i, c in enumerate(cases)])

    total_time = time.perf_counter() - t_start
    completed = [r for r in rows if r is not None]
    if stop_event.is_set():
        print(f"\nStopped early after {len(completed)}/{total} cases. "
              f"Total time: {total_time:.1f}s")
    else:
        print(f"\nTotal time: {total_time:.1f}s  ({total_time / max(total, 1):.1f}s/case avg)")

    return completed


def run_eval(args: argparse.Namespace) -> list[dict[str, Any]]:
    return asyncio.run(run_eval_async(args))


# ── Output ────────────────────────────────────────────────────

def load_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load all rows from an existing results CSV."""
    if not csv_path.exists():
        return []
    try:
        with csv_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_responses(rows: list[dict[str, Any]], path: Path) -> None:
    """Save a JSON file with question-response pairs for each evaluated case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        response_raw = row.get("response", "")
        try:
            response_parsed = json.loads(response_raw) if response_raw else None
        except (json.JSONDecodeError, TypeError):
            response_parsed = response_raw or None

        entries.append({
            "id": row.get("id", ""),
            "intent": row.get("intent", ""),
            "difficulty": row.get("difficulty", ""),
            "language": row.get("language", ""),
            "question": row.get("question", ""),
            "response": response_parsed,
            "reference": row.get("reference", ""),
            "bean_count": row.get("bean_count", 0),
            "news_count": row.get("news_count", 0),
        })

    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def write_csv(rows: list[dict[str, Any]], path: Path, merge_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if merge_existing:
        existing = load_existing_rows(path)
        new_ids = {row["id"] for row in rows if row.get("id")}
        merged = [row for row in existing if row.get("id") not in new_ids] + rows
    else:
        merged = rows

    preferred = ["id", "intent", "difficulty", "language", "question", "retrieved_context_count",
                 "bean_count", "news_count", "elapsed_s"]
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    tail = ["response", "reference", "error"]
    used = set(preferred + metric_cols + tail)
    extra = sorted(k for row in merged for k in row if k not in used)
    fieldnames = [c for c in preferred + metric_cols + extra + tail if any(c in row for row in merged)]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


def print_summary(rows: list[dict[str, Any]]) -> None:
    metric_names = [
        name
        for name in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
        if any(row.get(name) not in ("", None) for row in rows)
    ]
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY  ({len(rows)} cases)")
    print(f"{'=' * 60}")

    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows if row.get(metric_name) not in ("", None)]
        if values:
            avg = sum(values) / len(values)
            lo = min(values)
            hi = max(values)
            print(f"  {metric_name:<22s}  avg={avg:.3f}  min={lo:.3f}  max={hi:.3f}  n={len(values)}")

    # Breakdown by intent
    intents = sorted({row.get("intent", "") for row in rows} - {""})
    if len(intents) > 1:
        print(f"\n  By intent:")
        for intent in intents:
            subset = [row for row in rows if row.get("intent") == intent]
            parts = []
            for mn in metric_names:
                vals = [float(r[mn]) for r in subset if r.get(mn) not in ("", None)]
                if vals:
                    parts.append(f"{mn}={sum(vals)/len(vals):.3f}")
            print(f"    {intent:<20s}  n={len(subset):>3d}  {', '.join(parts)}")

    errors = [row for row in rows if row.get("error")]
    if errors:
        print(f"\n  errors: {len(errors)} rows (see CSV error column)")

    elapsed_vals = [float(row["elapsed_s"]) for row in rows if row.get("elapsed_s") not in ("", None)]
    if elapsed_vals:
        print(f"\n  timing: avg={sum(elapsed_vals)/len(elapsed_vals):.1f}s  "
              f"min={min(elapsed_vals):.1f}s  max={max(elapsed_vals):.1f}s")

    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation for the Coffee RAG system.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "ragas_eval_dataset.json")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results" / "ragas_results.csv")
    parser.add_argument("--mode", choices=["full", "retrieval"], default="full")
    parser.add_argument("--limit", type=int, default=5, help="Number of cases to run. Use 0 for all cases.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--intent", help="Filter by dataset intent, e.g. product_search")
    parser.add_argument("--language", choices=["vi", "en"], help="Filter by language.")
    parser.add_argument("--metrics", nargs="+", choices=["faithfulness", "context_precision", "context_recall", "answer_relevancy"])
    parser.add_argument("--top-k-beans", type=int, default=10)
    parser.add_argument("--top-k-news", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent samples scored in parallel.")
    parser.add_argument("--responses-out", type=Path, default=None,
                        help="Output JSON file for question-response pairs. Defaults to <out>_responses.json.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show entities, context counts, running averages per sample.")
    parser.add_argument("--evaluator-model", default=os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.responses_out is None:
        args.responses_out = args.out.with_name(args.out.stem + "_responses.json")
    resuming = args.limit == 0
    rows = run_eval(args)
    if rows:
        write_csv(rows, args.out, merge_existing=resuming)
        all_rows = load_existing_rows(args.out) if resuming else rows
        write_responses(all_rows, args.responses_out)
        print_summary(all_rows)
        print(f"\nWrote {args.out} ({len(all_rows)} total rows)")
        print(f"Wrote {args.responses_out} ({len(all_rows)} responses)")


if __name__ == "__main__":
    main()
