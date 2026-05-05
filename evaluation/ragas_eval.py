"""Evaluate the Coffee RAG pipeline with Ragas.

Example:
    python -m evaluation.ragas_eval --limit 5
    python -m evaluation.ragas_eval --mode retrieval --limit 20 --intent product_search
    python -m evaluation.ragas_eval --mode retrieval --metrics context_precision retrieval_recall_soft --limit 10
    python -m evaluation.ragas_eval --mode retrieval --limit 50 --workers 8 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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

import numpy as np  # noqa: E402
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
    # Ragas ContextRecall is still available as --metrics context_recall, but it
    # is a strict answer-support metric, not a document-retrieval recall metric.
    "full": [
        "faithfulness",
        "context_precision",
        "retrieval_recall_soft",
        "context_recall_soft",
        "answer_relevancy",
    ],
    "retrieval": ["context_precision", "retrieval_recall_soft", "context_recall_soft"],
}

RETRIEVAL_INTENTS = {"product_search", "similar_search", "news_search"}
CONTEXT_METRICS = {"context_precision", "context_recall", "context_recall_soft", "retrieval_recall_soft"}
OPENAI_LLM_METRICS = {"faithfulness", "context_precision", "context_recall", "answer_relevancy"}
SOFT_EMBEDDING_METRICS = {"context_recall_soft", "retrieval_recall_soft"}


def _metrics_for_intent(intent: str, all_metrics: dict[str, Any]) -> dict[str, Any]:
    """Return only metrics applicable to the given intent.

    Context metrics are skipped for non-retrieval intents because their
    ground_truth contains conceptual knowledge (knowledge_qa, comparison),
    aggregate statistics (exploration), or out-of-scope responses (edge_case)
    that cannot be found in individual bean/news contexts.
    """
    if intent in RETRIEVAL_INTENTS:
        return all_metrics
    return {k: v for k, v in all_metrics.items() if k not in CONTEXT_METRICS}


def _uses_openai(metric_names: list[str], embedding_model: str, ar_embedding_model: str | None) -> bool:
    if any(name in OPENAI_LLM_METRICS for name in metric_names):
        return True

    if any(name in SOFT_EMBEDDING_METRICS for name in metric_names):
        model = ar_embedding_model or embedding_model
        return "/" not in model

    return False


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
                date=str(row.get("publish_datetime", ""))[:10],
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


def response_summary_text(response: CoffeeResponse | None) -> str:
    """Extract only the summary field for Answer Relevancy scoring.

    The full JSON dump (products, articles, URLs) causes the AR metric to
    generate reverse-questions dominated by product details rather than the
    user's original intent, tanking cosine similarity.  Passing just the
    summary keeps generated questions aligned with the original query.
    """
    if response is None:
        return ""
    if hasattr(response, "summary"):
        return response.summary or ""
    return str(response)


# ── Soft context recall (embedding max-sim per reference unit) ─────────────

_SOFT_EMBED_MAX_CHARS = 8000
_SOFT_REF_MIN_UNIT_CHARS = 12
_RETRIEVAL_SOFT_SIM_LOW = float(os.getenv("RETRIEVAL_RECALL_SOFT_LOW", "0.78"))
_RETRIEVAL_SOFT_SIM_HIGH = float(os.getenv("RETRIEVAL_RECALL_SOFT_HIGH", "0.92"))


def _truncate_for_embed(text: str, max_chars: int = _SOFT_EMBED_MAX_CHARS) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _reference_units(reference: str, min_chars: int = _SOFT_REF_MIN_UNIT_CHARS) -> list[str]:
    """Split ground truth into sentence-like units for max-sim pooling."""
    text = (reference or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    units = [p.strip() for p in parts if p.strip() and len(p.strip()) >= min_chars]
    if not units:
        return [text] if text else []
    return units


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _scale_similarity(sim: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if sim >= high else 0.0
    return max(0.0, min(1.0, (sim - low) / (high - low)))


def _normalize_key(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,/\\")


def _context_keys(context: str) -> set[str]:
    """Stable identifiers for exact retrieval overlap before semantic scoring."""
    text = context or ""
    keys: set[str] = set()

    url_match = re.search(r"\bURL:\s*(\S+)", text)
    if url_match:
        keys.add("url:" + _normalize_key(url_match.group(1)))

    bean_match = re.match(r"Bean:\s*(.*?)\.\s*Roaster:\s*(.*?)\.\s", text, flags=re.S)
    if bean_match:
        name = _normalize_key(bean_match.group(1))
        roaster = _normalize_key(bean_match.group(2))
        if name and roaster:
            keys.add("bean_roaster:" + name + "|" + roaster)
        elif name:
            keys.add("bean:" + name)

    article_match = re.match(r"Article:\s*(.*?)\.\s*Source:\s*(.*?)\.\s", text, flags=re.S)
    if article_match:
        title = _normalize_key(article_match.group(1))
        source = _normalize_key(article_match.group(2))
        if title and source:
            keys.add("article_source:" + title + "|" + source)
        elif title:
            keys.add("article:" + title)

    return keys


class SoftContextRecall:
    """Mean over reference units of max cosine similarity to any retrieved context.

    Ragas ``ContextRecall`` uses an LLM to judge whether each reference claim
    is supported by a context span — strict on wording and structure. This metric
    aligns units of ``reference`` (sentences) with retrieved passages in
    embedding space instead.
    """

    __slots__ = ("embeddings",)

    def __init__(self, embeddings: Any):
        self.embeddings = embeddings

    async def ascore(
        self,
        *,
        user_input: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> Any:
        del user_input
        units = _reference_units(reference)
        if not units:
            return SimpleNamespace(value=float("nan"))
        contexts = [c for c in retrieved_contexts if c and str(c).strip()]
        if not contexts:
            return SimpleNamespace(value=0.0)

        ref_texts = [_truncate_for_embed(u) for u in units]
        ctx_texts = [_truncate_for_embed(c) for c in contexts]

        ref_emb = await self.embeddings.aembed_texts(ref_texts)
        ctx_emb = await self.embeddings.aembed_texts(ctx_texts)
        a = np.atleast_2d(np.asarray(ref_emb, dtype=np.float64))
        b = np.atleast_2d(np.asarray(ctx_emb, dtype=np.float64))
        a = _l2_normalize_rows(a)
        b = _l2_normalize_rows(b)
        sims = a @ b.T
        recall = float(np.mean(np.max(sims, axis=1)))
        recall = max(0.0, min(1.0, recall))
        return SimpleNamespace(value=recall)


class SoftRetrievalRecall:
    """Recall against dataset ground_truth_contexts.

    This is closer to IR recall than Ragas ContextRecall: each reference context
    is a target document/passage. Exact URL/name overlap receives full credit;
    otherwise the metric gives partial credit from embedding similarity.
    """

    __slots__ = ("embeddings", "sim_low", "sim_high")

    def __init__(
        self,
        embeddings: Any,
        sim_low: float = _RETRIEVAL_SOFT_SIM_LOW,
        sim_high: float = _RETRIEVAL_SOFT_SIM_HIGH,
    ):
        self.embeddings = embeddings
        self.sim_low = sim_low
        self.sim_high = sim_high

    async def ascore(
        self,
        *,
        user_input: str,
        reference_contexts: list[str],
        retrieved_contexts: list[str],
    ) -> Any:
        del user_input
        refs = [str(c).strip() for c in reference_contexts if c and str(c).strip()]
        if not refs:
            return SimpleNamespace(value=float("nan"))

        contexts = [str(c).strip() for c in retrieved_contexts if c and str(c).strip()]
        if not contexts:
            return SimpleNamespace(value=0.0)

        retrieved_keys = set().union(*(_context_keys(c) for c in contexts))
        scores: list[float | None] = []
        unresolved_refs: list[str] = []
        unresolved_positions: list[int] = []

        for ref in refs:
            ref_keys = _context_keys(ref)
            if ref_keys and ref_keys & retrieved_keys:
                scores.append(1.0)
            else:
                scores.append(None)
                unresolved_refs.append(ref)
                unresolved_positions.append(len(scores) - 1)

        if unresolved_refs:
            ref_texts = [_truncate_for_embed(c) for c in unresolved_refs]
            ctx_texts = [_truncate_for_embed(c) for c in contexts]
            ref_emb = await self.embeddings.aembed_texts(ref_texts)
            ctx_emb = await self.embeddings.aembed_texts(ctx_texts)
            a = _l2_normalize_rows(np.atleast_2d(np.asarray(ref_emb, dtype=np.float64)))
            b = _l2_normalize_rows(np.atleast_2d(np.asarray(ctx_emb, dtype=np.float64)))
            sims = a @ b.T

            for i, pos in enumerate(unresolved_positions):
                max_sim = float(np.max(sims[i]))
                scores[pos] = _scale_similarity(max_sim, self.sim_low, self.sim_high)

        recall = float(np.mean([s for s in scores if s is not None]))
        recall = max(0.0, min(1.0, recall))
        return SimpleNamespace(value=recall)


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


def build_metrics(names: list[str], evaluator_model: str, embedding_model: str,
                  ar_embedding_model: str | None = None):
    client = None
    llm = None
    embeddings = None

    def get_client():
        nonlocal client
        if client is None:
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client

    def get_llm():
        nonlocal llm
        if llm is None:
            llm = llm_factory(evaluator_model, client=get_client())
        return llm

    def get_embeddings():
        nonlocal embeddings
        if embeddings is None:
            embeddings = embedding_factory("openai", model=embedding_model, client=get_client())
        return embeddings

    need_ar_model = (
        "answer_relevancy" in names
        or "context_recall_soft" in names
        or "retrieval_recall_soft" in names
    )
    if need_ar_model:
        if ar_embedding_model and ar_embedding_model != embedding_model:
            if "/" in ar_embedding_model:
                ar_embeddings = embedding_factory(
                    "huggingface", model=ar_embedding_model, interface="modern",
                )
            else:
                ar_embeddings = embedding_factory(
                    "openai", model=ar_embedding_model, client=get_client(),
                )
        else:
            ar_embeddings = get_embeddings()
    else:
        ar_embeddings = None

    available: dict[str, Any] = {}
    if "faithfulness" in names:
        available["faithfulness"] = Faithfulness(llm=get_llm())
    if "context_precision" in names:
        available["context_precision"] = ContextPrecision(llm=get_llm())
    if "context_recall" in names:
        available["context_recall"] = ContextRecall(llm=get_llm())
    if "answer_relevancy" in names:
        ar_metric = AnswerRelevancy(llm=get_llm(), embeddings=ar_embeddings)
        # Patch prompt to force same-language reverse-question generation.
        ar_metric.prompt.instruction += (
            "\nIMPORTANT: Generate the question in the SAME LANGUAGE as the response. "
            "If the response is in Vietnamese, generate a Vietnamese question. "
            "If the response is in English, generate an English question."
        )
        available["answer_relevancy"] = ar_metric
    if "context_recall_soft" in names:
        available["context_recall_soft"] = SoftContextRecall(ar_embeddings)
    if "retrieval_recall_soft" in names:
        available["retrieval_recall_soft"] = SoftRetrievalRecall(ar_embeddings)

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
    elif metric_name == "context_recall_soft":
        result = await metric.ascore(
            user_input=sample["question"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    elif metric_name == "retrieval_recall_soft":
        result = await metric.ascore(
            user_input=sample["question"],
            reference_contexts=sample["reference_contexts"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
    elif metric_name == "answer_relevancy":
        result = await metric.ascore(
            user_input=sample["question"],
            response=sample.get("response_summary") or sample["response"],
        )
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    value = getattr(result, "value", result)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


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


PRODUCT_INTENTS = {"product_search", "similar_search"}
NEWS_INTENTS = {"news_search"}


def filter_reference_contexts_for_intent(intent: str, contexts: list[str]) -> list[str]:
    """Mirror retrieved-context pruning so recall is not penalized by ignored types."""
    if intent in PRODUCT_INTENTS:
        return [c for c in contexts if str(c).startswith("Bean: ")]
    if intent in NEWS_INTENTS:
        return [c for c in contexts if str(c).startswith("Article: ")]
    return contexts


def retrieve_one(rag: CoffeeRAG, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run retrieval + optional generation for a single case. Returns a sample dict."""
    question = case["question"]
    ctx = rag.retrieve(question, top_k_beans=args.top_k_beans, top_k_news=args.top_k_news,
                       use_rrf=args.use_rrf)

    beans = ctx.get("beans")
    news = ctx.get("news")
    pipeline_intent = ctx.get("intent", "")
    case_intent = case.get("intent", "") or pipeline_intent
    bean_count = len(beans) if beans is not None and not getattr(beans, "empty", True) else 0
    news_count = len(news) if news is not None and not getattr(news, "empty", True) else 0

    eval_ctx = dict(ctx)
    # Drop irrelevant context type per intent to avoid noise in CP scoring.
    # Route by CASE intent (dataset label = ground truth), not pipeline intent —
    # this keeps eval honest when the classifier mislabels (e.g. NS query
    # misrouted to product_search would otherwise have its news contexts
    # dropped before scoring).
    if case_intent in PRODUCT_INTENTS:
        # News articles are always irrelevant for product retrieval.
        eval_ctx["news"] = None
        if beans is not None and len(beans) > MAX_EVAL_CONTEXTS:
            eval_ctx["beans"] = beans.head(MAX_EVAL_CONTEXTS)
    elif case_intent in NEWS_INTENTS:
        # Bean contexts are irrelevant for news queries.
        eval_ctx["beans"] = None
        if news is not None and len(news) > MAX_EVAL_CONTEXTS:
            eval_ctx["news"] = news.head(MAX_EVAL_CONTEXTS)
    else:
        if beans is not None and len(beans) > MAX_EVAL_CONTEXTS:
            eval_ctx["beans"] = beans.head(MAX_EVAL_CONTEXTS)
        if news is not None and len(news) > MAX_EVAL_CONTEXTS:
            eval_ctx["news"] = news.head(MAX_EVAL_CONTEXTS)
    retrieved_contexts = build_retrieved_contexts(eval_ctx)
    reference_contexts = filter_reference_contexts_for_intent(
        case_intent,
        case.get("ground_truth_contexts", []),
    )

    response = None
    if args.mode == "full":
        messages = build_prompt(question, ctx["beans"], ctx["news"])
        response = generate_structured(messages, CoffeeResponse, client=rag.llm_client)

    return {
        "question": question,
        "reference": case["ground_truth"],
        "response": response_to_text(response),
        "response_summary": response_summary_text(response),
        "retrieved_contexts": retrieved_contexts,
        "reference_contexts": reference_contexts,
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
    for name in (
        "context_precision",
        "context_recall",
        "retrieval_recall_soft",
        "context_recall_soft",
        "faithfulness",
        "answer_relevancy",
    ):
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
        print(
            f"     beans={sample['bean_count']}  news={sample['news_count']}  "
            f"contexts={len(sample['retrieved_contexts'])}  "
            f"ref_contexts={len(sample.get('reference_contexts', []))}"
        )
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
    metric_names = args.metrics or DEFAULT_METRICS[args.mode]
    unsupported = sorted(set(metric_names) - {
        "faithfulness",
        "context_precision",
        "context_recall",
        "context_recall_soft",
        "retrieval_recall_soft",
        "answer_relevancy",
    })
    if unsupported:
        raise ValueError(f"Unsupported metrics: {', '.join(unsupported)}")
    if args.mode == "retrieval":
        metric_names = [
            name
            for name in metric_names
            if name in {"context_precision", "context_recall", "context_recall_soft", "retrieval_recall_soft"}
        ]

    if _uses_openai(metric_names, args.embedding_model, args.ar_embedding_model) and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for selected Ragas/OpenAI-backed metrics.")

    skip_ids = load_existing_ids(args.out) if args.limit == 0 else None
    cases = load_cases(args.dataset, args.limit, args.offset, args.intent, args.language, skip_ids=skip_ids)
    if not cases:
        print("No cases matched filters.")
        return []

    metrics = build_metrics(metric_names, args.evaluator_model, args.embedding_model,
                             ar_embedding_model=args.ar_embedding_model)
    rag = CoffeeRAG()
    total = len(cases)
    stats = RunningStats()
    semaphore = asyncio.Semaphore(args.workers)
    rows: list[dict[str, Any]] = [None] * total  # type: ignore[list-item]
    t_start = time.perf_counter()

    ZERO_PRECISION_STOP_RATIO = 0.2
    stop_event = asyncio.Event()
    zero_precision_count = 0
    retrieval_case_count = sum(1 for c in cases if c.get("intent", "") in RETRIEVAL_INTENTS)
    zero_precision_threshold = math.ceil(ZERO_PRECISION_STOP_RATIO * max(retrieval_case_count, 1))

    print(f"\n{'=' * 60}")
    print(f"  Ragas evaluation  |  mode={args.mode}  metrics={metric_names}")
    print(f"  cases={total} (retrieval={retrieval_case_count})  workers={args.workers}  evaluator={args.evaluator_model}")
    print(f"  embeddings: {args.embedding_model}  AR: {args.ar_embedding_model}")
    early_stop_enabled = (
        args.use_rrf
        and "context_precision" in metric_names
        and "retrieval_recall_soft" not in metric_names
    )
    if early_stop_enabled:
        print(f"  early stop: context_precision=0 >= {zero_precision_threshold} retrieval cases ({ZERO_PRECISION_STOP_RATIO:.0%} of {retrieval_case_count})")
    else:
        print("  early stop: DISABLED")
    print(f"  intent-aware: context metrics skipped for knowledge_qa, comparison, exploration, edge_case")
    print(f"{'=' * 60}\n")

    async def process_one(index: int, case: dict[str, Any]):
        nonlocal zero_precision_count

        if stop_event.is_set():
            return

        async with semaphore:
            if stop_event.is_set():
                return

            t0 = time.perf_counter()

            sample = await asyncio.to_thread(retrieve_one, rag, case, args)
            intent = case.get("intent", sample.get("intent", ""))
            case_metrics = _metrics_for_intent(intent, metrics)
            scores = await score_all_metrics(case_metrics, sample)

            elapsed = time.perf_counter() - t0
            stats.update(scores)

            row = {
                "id": case.get("id", ""),
                "intent": case.get("intent", ""),
                "difficulty": case.get("difficulty", ""),
                "language": case.get("language", ""),
                "question": case["question"],
                "retrieved_context_count": len(sample["retrieved_contexts"]),
                "reference_context_count": len(sample["reference_contexts"]),
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

            cp = scores.get("context_precision")
            if cp is not None and cp != "" and float(cp) == 0.0 and intent in RETRIEVAL_INTENTS:
                zero_precision_count += 1
            if early_stop_enabled and zero_precision_count >= zero_precision_threshold:
                print(f"\n{'!' * 60}")
                print(f"  EARLY STOP: {zero_precision_count} cases with context_precision=0 "
                      f"(reached threshold {zero_precision_threshold}, {ZERO_PRECISION_STOP_RATIO:.0%} of {total})")
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

    preferred = [
        "id", "intent", "difficulty", "language", "question",
        "retrieved_context_count", "reference_context_count",
        "bean_count", "news_count", "elapsed_s",
    ]
    metric_cols = [
        "context_precision",
        "context_recall",
        "retrieval_recall_soft",
        "context_recall_soft",
        "faithfulness",
        "answer_relevancy",
    ]
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
        for name in (
            "faithfulness",
            "context_precision",
            "context_recall",
            "retrieval_recall_soft",
            "context_recall_soft",
            "answer_relevancy",
        )
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
            ctx_note = "" if intent in RETRIEVAL_INTENTS else "  (context metrics skipped)"
            print(f"    {intent:<20s}  n={len(subset):>3d}  {', '.join(parts)}{ctx_note}")

    # Breakdown by language
    languages = sorted({row.get("language", "") for row in rows} - {""})
    if len(languages) > 1:
        print(f"\n  By language:")
        for lang in languages:
            subset = [row for row in rows if row.get("language") == lang]
            parts = []
            for mn in metric_names:
                vals = [float(r[mn]) for r in subset if r.get(mn) not in ("", None)]
                if vals:
                    parts.append(f"{mn}={sum(vals)/len(vals):.3f}")
            print(f"    {lang:<20s}  n={len(subset):>3d}  {', '.join(parts)}")

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
    parser.add_argument("--metrics", nargs="+", choices=[
        "faithfulness",
        "context_precision",
        "context_recall",
        "context_recall_soft",
        "retrieval_recall_soft",
        "answer_relevancy",
    ])
    parser.add_argument("--top-k-beans", type=int, default=int(os.getenv("TOP_K_BEANS", "5")))
    parser.add_argument("--top-k-news", type=int, default=int(os.getenv("TOP_K_NEWS", "5")))
    parser.add_argument("--use-rrf", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Reciprocal Rank Fusion when merging candidate lists. "
                             "Use --no-use-rrf to fall back to priority-order concat + dedupe "
                             "(same candidate sources, no RRF).")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent samples scored in parallel.")
    parser.add_argument("--responses-out", type=Path, default=None,
                        help="Output JSON file for question-response pairs. Defaults to <out>_responses.json.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show entities, context counts, running averages per sample.")
    parser.add_argument("--evaluator-model", default=os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--ar-embedding-model", default=os.getenv("RAGAS_AR_EMBEDDING_MODEL", "BAAI/bge-m3"),
                        help="Embedding model for Answer Relevancy (multilingual). Defaults to BAAI/bge-m3.")
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
