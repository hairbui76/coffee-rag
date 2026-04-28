"""Generate RAG evaluation dataset grounded in real coffee_beans + coffee_news data.

v3 changes vs v2:
- Removed exploration + edge_case intents (incompatible with context precision eval)
- **Retrieval-grounded**: ground_truth + ground_truth_contexts are generated from
  actual RAG retrieval results (not pre-selected products), so the dataset naturally
  aligns with what the system retrieves at eval time → much fewer CP=0 cases
- Aligned ground_truth_contexts format with ragas_eval retrieved_contexts format
- Tighter _good_beans filter (description >100 chars, non-empty origin)
- Redistributed: PS=130, SS=80, CP=70, KQ=70, NS=60 (total 410)

Usage:
    python evaluation/generate_dataset.py --dry-run
    python evaluation/generate_dataset.py --out ragas_eval_dataset_v3.json
    python evaluation/generate_dataset.py --workers 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BEANS_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"
NEWS_PATH = ROOT / "data" / "processed" / "news_chunks.parquet"
NEWS_CLEAN_PATH = ROOT / "data" / "processed" / "news_clean.parquet"

random.seed(42)
np.random.seed(42)


# ── Helpers ────────────────────────────────────────────────────

def _arr(val) -> list[str]:
    if isinstance(val, np.ndarray):
        return [str(x) for x in val]
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def _bean_summary(row: pd.Series) -> str:
    name = row.get("product_name", "")
    roaster = row.get("roaster_name", "")
    origin = row.get("origin", "")
    country = row.get("country", "")
    roast = row.get("roast_level_clean", "Unknown")
    flavors = ", ".join(_arr(row.get("flavor_notes_clean", [])))
    processing = ", ".join(_arr(row.get("processing_clean", [])))
    species = ", ".join(_arr(row.get("species", [])))
    desc = str(row.get("about_description", ""))[:200]
    return (
        f"{name} by {roaster}. Origin: {origin}, {country}. "
        f"Roast: {roast}. Flavor: {flavors}. Processing: {processing}. "
        f"Species: {species}. Description: {desc}"
    )


def _bean_context_text(row: pd.Series) -> str:
    """Format a bean row identically to ragas_eval._bean_contexts()."""
    name = str(row.get("product_name", ""))
    roaster = str(row.get("roaster_name", ""))
    origin = str(row.get("origin", ""))
    country = str(row.get("country", ""))
    roast = str(row.get("roast_level_clean", "") or row.get("roast_level", ""))
    flavors = ", ".join(map(str, _arr(row.get("flavor_notes_clean", []))))
    processing = ", ".join(map(str, _arr(row.get("processing_clean", []))))
    species = ", ".join(map(str, _arr(row.get("species", []))))
    desc = str(row.get("about_description", ""))
    url = str(row.get("product_url", ""))
    return (
        f"Bean: {name}. Roaster: {roaster}. Origin: {origin}. Country: {country}. "
        f"Roast: {roast}. Flavor: {flavors}. Processing: {processing}. "
        f"Species: {species}. Description: {desc}. URL: {url}"
    )


def _news_context(row: pd.Series) -> str:
    title = row.get("title", "")
    source = row.get("source", "")
    date = str(row.get("publish_datetime", ""))[:10]
    text = str(row.get("text", ""))
    url = row.get("article_url", "")
    return f"Article: {title}. Source: {source}. Date: {date}. Content: {text}. URL: {url}"


def _good_beans(beans: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (beans["country"] != "")
        & (beans["roast_level_clean"] != "Unknown")
        & (beans["flavor_notes_clean"].apply(lambda x: len(_arr(x)) >= 2))
        & (beans["about_description"].str.len() > 100)
        & (beans["origin"] != "")
    )
    return beans[mask].copy()


def _assign_difficulty(n: int) -> list[str]:
    easy = int(n * 0.30)
    hard = int(n * 0.25)
    medium = n - easy - hard
    return ["easy"] * easy + ["medium"] * medium + ["hard"] * hard


def _assign_language(n: int, vi_ratio: float = 0.6) -> list[str]:
    vi = int(n * vi_ratio)
    en = n - vi
    langs = ["vi"] * vi + ["en"] * en
    random.shuffle(langs)
    return langs


# ── Async LLM ─────────────────────────────────────────────────

async def _llm_generate(
    client: AsyncOpenAI, model: str, system: str, user: str, sem: asyncio.Semaphore
) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=800,
        )
    return resp.choices[0].message.content.strip()


QUESTION_SYSTEM = """\
You generate natural-sounding questions for a coffee RAG evaluation dataset.
- If language=vi, write in Vietnamese.
- If language=en, write in English.
- Make questions sound like real users asking a coffee chatbot.
- Vary question styles (polite, direct, casual).
- Return ONLY the question, nothing else."""

GT_SYSTEM = """\
You write ground-truth answers for a coffee RAG evaluation dataset.
The answer MUST be based ONLY on the provided product/article data.
- Reference specific product names, roasters, and details from the data.
- Be factual and concise (3-6 sentences).
- If language=vi, write in Vietnamese. If language=en, write in English.
- Return ONLY the answer, nothing else."""


# ── Build "specs" (data-only, no LLM) then resolve async ──────

def _build_product_search_specs(beans: pd.DataFrame, n: int) -> list[dict]:
    good = _good_beans(beans)
    specs = []
    difficulties = _assign_difficulty(n)
    languages = _assign_language(n)

    top_countries = good["country"].value_counts().head(12).index.tolist()
    top_roasts = ["Light", "Medium", "Medium-Light", "Medium-Dark", "Dark"]

    all_flavors = Counter()
    for fn in good["flavor_notes_clean"]:
        for f in _arr(fn):
            all_flavors[f] += 1
    top_flavors = [f for f, _ in all_flavors.most_common(25)]

    combos_used: set[tuple] = set()
    for i in range(n):
        selected = None
        country = roast = flavor = ""
        for _ in range(50):
            country = random.choice(top_countries)
            roast = random.choice(top_roasts)
            flavor = random.choice(top_flavors[:15])
            combo_key = (country, roast, flavor)
            if combo_key in combos_used:
                continue
            mask = (
                (good["country"] == country)
                & (good["roast_level_clean"] == roast)
                & (good["flavor_notes_clean"].apply(lambda x: flavor in _arr(x)))
            )
            matches = good[mask]
            if len(matches) >= 2:
                combos_used.add(combo_key)
                selected = matches.sample(min(3, len(matches)))
                break

        if selected is None:
            selected = good.sample(3)
            country = selected.iloc[0]["country"]
            roast = selected.iloc[0]["roast_level_clean"]
            fns = _arr(selected.iloc[0]["flavor_notes_clean"])
            flavor = fns[0] if fns else "chocolate"

        specs.append({
            "id": f"PS_{i+1:03d}",
            "intent": "product_search",
            "difficulty": difficulties[i],
            "language": languages[i],
            "selected": selected,
            "country": country,
            "roast": roast,
            "flavor": flavor,
        })
    return specs


def _build_similar_search_specs(beans: pd.DataFrame, n: int) -> list[dict]:
    good = _good_beans(beans)
    specs = []
    difficulties = _assign_difficulty(n)
    languages = _assign_language(n)
    used_seeds: set[str] = set()

    for i in range(n):
        seed = similar_selected = None
        for _ in range(50):
            seed = good.sample(1).iloc[0]
            if seed["product_name"] in used_seeds:
                continue
            country = seed["country"]
            roast = seed["roast_level_clean"]
            flavors = set(_arr(seed["flavor_notes_clean"]))

            similar_mask = (
                (good["product_name"] != seed["product_name"])
                & ((good["country"] == country) | (good["roast_level_clean"] == roast))
                & (good["flavor_notes_clean"].apply(lambda x: len(flavors & set(_arr(x))) >= 1))
            )
            similar = good[similar_mask]
            if len(similar) >= 2:
                used_seeds.add(seed["product_name"])
                similar_selected = similar.sample(min(3, len(similar)))
                break

        if seed is None or similar_selected is None:
            seed = good.sample(1).iloc[0]
            similar_selected = good.sample(3)

        specs.append({
            "id": f"SS_{i+1:03d}",
            "intent": "similar_search",
            "difficulty": difficulties[i],
            "language": languages[i],
            "seed": seed,
            "similar_selected": similar_selected,
        })
    return specs


def _build_comparison_specs(beans: pd.DataFrame, n: int) -> list[dict]:
    good = _good_beans(beans)
    specs = []
    difficulties = _assign_difficulty(n)
    languages = _assign_language(n)

    concept_comparisons = [
        ("Natural", "Washed", "processing"),
        ("Light", "Dark", "roast"),
        ("Arabica", "Robusta", "species"),
        ("Honey", "Washed", "processing"),
        ("Natural", "Honey", "processing"),
        ("Light", "Medium", "roast"),
        ("Medium", "Dark", "roast"),
    ]

    for i in range(n):
        if i < len(concept_comparisons):
            a_val, b_val, field = concept_comparisons[i]
            if field == "processing":
                a_beans = good[good["processing_clean"].apply(lambda x: a_val in _arr(x))]
                b_beans = good[good["processing_clean"].apply(lambda x: b_val in _arr(x))]
            elif field == "roast":
                a_beans = good[good["roast_level_clean"] == a_val]
                b_beans = good[good["roast_level_clean"] == b_val]
            else:
                a_beans = good[good["species"].apply(lambda x: a_val in _arr(x))]
                b_beans = good[good["species"].apply(lambda x: b_val in _arr(x))]
            a_beans = a_beans.sample(min(2, len(a_beans)))
            b_beans = b_beans.sample(min(2, len(b_beans)))
            all_beans = pd.concat([a_beans, b_beans])
            comparison_label = f"{a_val} vs {b_val} ({field})"
        else:
            all_beans = good.sample(2)
            comparison_label = f"{all_beans.iloc[0]['product_name']} vs {all_beans.iloc[1]['product_name']}"

        specs.append({
            "id": f"CP_{i+1:03d}",
            "intent": "comparison",
            "difficulty": difficulties[i],
            "language": languages[i],
            "all_beans": all_beans,
            "comparison_label": comparison_label,
        })
    return specs


def _build_knowledge_qa_specs(beans: pd.DataFrame, n: int) -> list[dict]:
    good = _good_beans(beans)
    specs = []
    difficulties = _assign_difficulty(n)
    languages = _assign_language(n)

    topics = [
        ("processing", "Washed", "What is Washed processing?"),
        ("processing", "Natural", "What is Natural processing?"),
        ("processing", "Honey", "What is Honey processing?"),
        ("processing", "Anaerobic", "What is Anaerobic fermentation?"),
        ("roast", "Light", "What defines a light roast coffee?"),
        ("roast", "Medium", "What defines a medium roast coffee?"),
        ("roast", "Dark", "What defines a dark roast coffee?"),
        ("species", "Arabica", "What is Arabica coffee?"),
        ("species", "Robusta", "What is Robusta coffee?"),
        ("flavor", "Chocolate", "What coffees have chocolate notes?"),
        ("flavor", "Floral", "What coffees have floral notes?"),
        ("flavor", "Citrus Fruit", "What coffees have citrus notes?"),
        ("flavor", "Honey", "Which coffees have honey flavor notes?"),
        ("flavor", "Jasmine", "What coffees have jasmine notes?"),
        ("flavor", "Blueberry", "What coffees taste like blueberry?"),
        ("origin", "Ethiopia", "What is special about Ethiopian coffee?"),
        ("origin", "Colombia", "What characterizes Colombian coffee?"),
        ("origin", "Brazil", "What is Brazilian coffee known for?"),
        ("origin", "Panama", "What makes Panamanian coffee special?"),
        ("origin", "Kenya", "What characterizes Kenyan coffee?"),
    ]

    for i in range(n):
        base_q = "General coffee knowledge question"
        if i < len(topics):
            field, value, base_q = topics[i]
            if field == "processing":
                examples = good[good["processing_clean"].apply(lambda x: value in " ".join(_arr(x)))]
            elif field == "roast":
                examples = good[good["roast_level_clean"] == value]
            elif field == "species":
                examples = good[good["species"].apply(lambda x: value in _arr(x))]
            elif field == "flavor":
                examples = good[good["flavor_notes_clean"].apply(lambda x: value in _arr(x))]
            else:
                examples = good[good["country"] == value]
            selected = examples.sample(min(3, len(examples))) if len(examples) > 0 else good.sample(3)
        else:
            selected = good.sample(3)

        specs.append({
            "id": f"KQ_{i+1:03d}",
            "intent": "knowledge_qa",
            "difficulty": difficulties[i],
            "language": languages[i],
            "selected": selected,
            "base_q": base_q,
        })
    return specs


def _build_news_search_specs(
    news_chunks: pd.DataFrame, news_clean: pd.DataFrame, n: int
) -> list[dict]:
    specs = []
    difficulties = _assign_difficulty(n)
    languages = _assign_language(n)

    usable = news_clean[
        (news_clean["title"].str.len() > 20)
        & (news_clean["summary"].str.len() > 50)
    ].copy()
    sampled = usable.sample(min(n, len(usable)), replace=len(usable) < n)

    for i, (_, article) in enumerate(sampled.iterrows()):
        if i >= n:
            break
        title = article["title"]
        summary = article.get("summary", "")
        url = article.get("article_url", "")
        source = article.get("source", "")

        chunks = news_chunks[news_chunks["article_url"] == url]
        if chunks.empty:
            chunks = news_chunks[news_chunks["title"] == title]
        ctx_texts = (
            [_news_context(r) for _, r in chunks.head(3).iterrows()]
            if not chunks.empty
            else [f"Article: {title}. Source: {source}. Summary: {summary}. URL: {url}"]
        )

        specs.append({
            "id": f"NS_{i+1:03d}",
            "intent": "news_search",
            "difficulty": difficulties[i],
            "language": languages[i],
            "title": title,
            "summary": summary,
            "url": url,
            "source": source,
            "ctx_texts": ctx_texts,
        })
    return specs


# ── Async resolve: turn specs into cases with LLM calls ───────

async def _resolve_spec(
    spec: dict, client: AsyncOpenAI, model: str, sem: asyncio.Semaphore,
    rag: Any = None,
) -> dict:
    intent = spec["intent"]
    lang = spec["language"]
    diff = spec["difficulty"]

    # ── Phase 1: Build question prompt from pre-selected data ────
    metadata: dict[str, Any] = {}

    if intent == "product_search":
        selected = spec["selected"]
        q_prompt = (
            f"Language: {lang}\nIntent: product_search\nDifficulty: {diff}\n"
            f"Generate a question asking for coffee from {spec['country']}, "
            f"{spec['roast']} roast, with {spec['flavor']} flavor notes.\n"
            f"If hard, add extra constraints (processing, brew method, species)."
        )
        metadata["expected_entities"] = {
            "country": spec["country"], "roast": spec["roast"], "flavor": [spec["flavor"]],
        }
        metadata["sampled_product_names"] = [r["product_name"] for _, r in selected.iterrows()]

    elif intent == "similar_search":
        seed = spec["seed"]
        seed_summary = _bean_summary(seed)
        q_prompt = (
            f"Language: {lang}\nIntent: similar_search\nDifficulty: {diff}\n"
            f"Generate a question asking for coffees similar to '{seed['product_name']}' by {seed['roaster_name']}.\n"
            f"The seed coffee is: {seed_summary}"
        )
        metadata["reference_product"] = seed["product_name"]
        metadata["reference_roaster"] = seed["roaster_name"]

    elif intent == "comparison":
        all_beans = spec["all_beans"]
        summaries = "\n".join([_bean_summary(r) for _, r in all_beans.iterrows()])
        q_prompt = (
            f"Language: {lang}\nIntent: comparison\nDifficulty: {diff}\n"
            f"Generate a question comparing: {spec['comparison_label']}\n"
            f"Products:\n{summaries}"
        )
        metadata["comparison_label"] = spec["comparison_label"]

    elif intent == "knowledge_qa":
        selected = spec["selected"]
        summaries = "\n".join([_bean_summary(r) for _, r in selected.iterrows()])
        q_prompt = (
            f"Language: {lang}\nIntent: knowledge_qa\nDifficulty: {diff}\n"
            f"Topic: {spec['base_q']}\n"
            f"Reference products:\n{summaries}\n"
            f"If hard, ask deeper (chemistry, history, technique). Vary from the base topic."
        )
        metadata["topic"] = spec["base_q"]

    elif intent == "news_search":
        q_prompt = (
            f"Language: {lang}\nIntent: news_search\nDifficulty: {diff}\n"
            f"Generate a question about this news topic:\n"
            f"Title: {spec['title']}\nSource: {spec['source']}\nSummary: {spec['summary'][:200]}"
        )
        metadata["source_article"] = spec["url"]
        metadata["source_title"] = spec["title"]
        metadata["source_name"] = spec["source"]

    else:
        raise ValueError(f"Unknown intent: {intent}")

    # ── Phase 2: Generate question ───────────────────────────────
    question = await _llm_generate(client, model, QUESTION_SYSTEM, q_prompt, sem)

    # ── Phase 3: Retrieve actual contexts ────────────────────────
    ctx = await asyncio.to_thread(rag.retrieve, question, 5, 5)
    retrieved_beans = ctx.get("beans")
    retrieved_news = ctx.get("news")

    contexts: list[str] = []
    gt_parts: list[str] = []

    if retrieved_beans is not None and not retrieved_beans.empty:
        top_beans = retrieved_beans.head(5)
        contexts += [_bean_context_text(r) for _, r in top_beans.iterrows()]
        bean_text = "\n".join([_bean_summary(r) for _, r in top_beans.iterrows()])
        gt_parts.append(f"Retrieved products:\n{bean_text}")
        metadata["retrieved_product_names"] = top_beans["product_name"].tolist()

    if retrieved_news is not None and not retrieved_news.empty:
        top_news = retrieved_news.head(3)
        contexts += [_news_context(r) for _, r in top_news.iterrows()]
        news_text = "\n".join(
            [f"- {r.get('title', '')} ({r.get('source', '')}): {str(r.get('text', ''))[:150]}"
             for _, r in top_news.iterrows()]
        )
        gt_parts.append(f"Retrieved articles:\n{news_text}")

    # ── Phase 4: Generate ground truth from retrieved data ───────
    gt_data = "\n\n".join(gt_parts) if gt_parts else "(no relevant documents retrieved)"
    gt_prompt = f"Language: {lang}\nQuestion: {question}\n\nAnswer based on this RETRIEVED data:\n{gt_data}"
    ground_truth = await _llm_generate(client, model, GT_SYSTEM, gt_prompt, sem)

    return {
        "id": spec["id"],
        "intent": intent,
        "difficulty": diff,
        "language": lang,
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_contexts": contexts,
        "metadata": metadata,
    }


# ── Main ──────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace):
    print("Loading data...")
    beans = pd.read_parquet(BEANS_PATH)
    news_chunks = pd.read_parquet(NEWS_PATH)
    news_clean = pd.read_parquet(NEWS_CLEAN_PATH)
    print(f"  Beans: {len(beans)}, News chunks: {len(news_chunks)}, News articles: {len(news_clean)}")

    good = _good_beans(beans)
    print(f"  Good beans (known country+roast+flavors+description): {len(good)}")

    INTENT_COUNTS = {
        "product_search": 130,
        "similar_search": 80,
        "comparison": 70,
        "knowledge_qa": 70,
        "news_search": 60,
    }

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        for name, cnt in INTENT_COUNTS.items():
            print(f"  {name}: {cnt}")
        print(f"  TOTAL: {sum(INTENT_COUNTS.values())}")
        return

    existing_ids: set[str] = set()
    existing_cases: list[dict] = []
    partial_path = args.out.with_suffix(".partial.json")
    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as f:
            existing_cases = json.load(f)
        existing_ids = {c["id"] for c in existing_cases}
        print(f"  Resuming: found {len(existing_cases)} existing cases in partial file")

    print("\nBuilding specs (sampling real data)...")
    all_specs: list[dict] = []
    all_specs += _build_product_search_specs(beans, n=INTENT_COUNTS["product_search"])
    all_specs += _build_similar_search_specs(beans, n=INTENT_COUNTS["similar_search"])
    all_specs += _build_comparison_specs(beans, n=INTENT_COUNTS["comparison"])
    all_specs += _build_knowledge_qa_specs(beans, n=INTENT_COUNTS["knowledge_qa"])
    all_specs += _build_news_search_specs(news_chunks, news_clean, n=INTENT_COUNTS["news_search"])

    pending = [s for s in all_specs if s["id"] not in existing_ids]
    print(f"  Total specs: {len(all_specs)}, pending: {len(pending)}")

    if not pending:
        print("All cases already generated!")
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(existing_cases, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.out}")
        return

    # ── Initialize RAG pipeline for retrieval-grounded generation ──
    from src.pipeline import CoffeeRAG
    print("Loading RAG pipeline...")
    rag = CoffeeRAG()
    print("  RAG pipeline ready")

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = args.model
    sem = asyncio.Semaphore(args.workers)
    print(f"Using LLM: {model}, workers: {args.workers}")

    t0 = time.perf_counter()
    completed = list(existing_cases)
    done_count = len(existing_cases)

    batch_size = 20
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        tasks = [_resolve_spec(spec, client, model, sem, rag=rag) for spec in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for spec, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"  ERROR {spec['id']}: {result}")
                continue
            completed.append(result)
            done_count += 1

        with partial_path.open("w", encoding="utf-8") as f:
            json.dump(completed, f, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - t0
        rate = (done_count - len(existing_cases)) / max(elapsed, 0.1)
        remaining = (len(pending) - (batch_start + len(batch))) / max(rate, 0.01)
        last_id = batch[-1]["id"]
        print(f"  [{done_count}/{len(all_specs)}] {last_id} done  "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.perf_counter() - t0
    print(f"\nGenerated {len(completed)} cases in {elapsed:.1f}s")

    questions = [c["question"] for c in completed]
    dupes = len(questions) - len(set(questions))
    if dupes:
        print(f"WARNING: {dupes} duplicate questions found")

    id_order = {s["id"]: i for i, s in enumerate(all_specs)}
    completed.sort(key=lambda c: id_order.get(c["id"], 999))

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")

    if partial_path.exists():
        partial_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Generate grounded RAG eval dataset.")
    parser.add_argument("--out", type=Path, default=ROOT / "ragas_eval_dataset_v3.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent LLM calls.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
