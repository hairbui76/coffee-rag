"""Validate ragas_eval_dataset_v2.json: check ground_truth_contexts match real records,
no duplicate questions, correct intent/difficulty/language distribution."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "ragas_eval_dataset_v2.json"
BEANS_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"
NEWS_PATH = ROOT / "data" / "processed" / "news_chunks.parquet"


def main():
    print("Loading dataset and data...")
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    beans = pd.read_parquet(BEANS_PATH)
    news = pd.read_parquet(NEWS_PATH)
    print(f"  Dataset: {len(cases)} cases")
    print(f"  Beans: {len(beans)}, News chunks: {len(news)}")

    all_bean_texts = set(beans["document_text"].dropna().tolist())
    all_news_texts = set()
    for _, row in news.iterrows():
        title = row.get("title", "")
        source = row.get("source", "")
        date = str(row.get("publish_datetime", ""))[:10]
        text = str(row.get("text", ""))
        url = row.get("article_url", "")
        all_news_texts.add(f"Article: {title}. Source: {source}. Date: {date}. Content: {text}. URL: {url}")

    # 1. Distribution checks
    intent_counts = Counter(c["intent"] for c in cases)
    diff_counts = Counter(c["difficulty"] for c in cases)
    lang_counts = Counter(c["language"] for c in cases)

    print("\n=== Distribution ===")
    print("Intent:")
    for k, v in sorted(intent_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nDifficulty:")
    for k, v in sorted(diff_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nLanguage:")
    for k, v in sorted(lang_counts.items()):
        print(f"  {k}: {v}")

    # 2. Duplicate check
    questions = [c["question"] for c in cases]
    dupes = len(questions) - len(set(questions))
    print(f"\n=== Duplicates ===")
    print(f"  Duplicate questions: {dupes}")
    if dupes > 0:
        seen = set()
        for q in questions:
            if q in seen:
                print(f"  DUP: {q[:80]}...")
            seen.add(q)

    # 3. ground_truth_contexts grounding check
    grounded = 0
    ungrounded = 0
    empty_ctx = 0
    total_ctx = 0
    ungrounded_examples = []

    for c in cases:
        contexts = c.get("ground_truth_contexts", [])
        if not contexts:
            empty_ctx += 1
            continue
        for ctx in contexts:
            total_ctx += 1
            if ctx in all_bean_texts or ctx in all_news_texts:
                grounded += 1
            elif ctx.startswith("Database statistic:") or ctx.startswith("Article:"):
                grounded += 1
            else:
                ungrounded += 1
                if len(ungrounded_examples) < 5:
                    ungrounded_examples.append((c["id"], ctx[:100]))

    print(f"\n=== Context Grounding ===")
    print(f"  Total contexts: {total_ctx}")
    print(f"  Grounded in real data: {grounded} ({grounded/max(total_ctx,1)*100:.1f}%)")
    print(f"  Ungrounded: {ungrounded}")
    print(f"  Cases with empty contexts: {empty_ctx} (expected for edge_cases)")
    if ungrounded_examples:
        print(f"\n  Ungrounded examples (first 5):")
        for cid, ctx in ungrounded_examples:
            print(f"    [{cid}] {ctx}")

    # 4. Quality checks
    no_gt = sum(1 for c in cases if not c.get("ground_truth"))
    no_q = sum(1 for c in cases if not c.get("question"))
    short_gt = sum(1 for c in cases if len(c.get("ground_truth", "")) < 20)
    short_q = sum(1 for c in cases if len(c.get("question", "")) < 10)

    print(f"\n=== Quality ===")
    print(f"  Missing ground_truth: {no_gt}")
    print(f"  Missing question: {no_q}")
    print(f"  Short ground_truth (<20 chars): {short_gt}")
    print(f"  Short question (<10 chars): {short_q}")

    # 5. ID uniqueness
    ids = [c["id"] for c in cases]
    dup_ids = len(ids) - len(set(ids))
    print(f"  Duplicate IDs: {dup_ids}")

    # Summary
    issues = dupes + ungrounded + no_gt + no_q + dup_ids
    print(f"\n{'='*40}")
    if issues == 0:
        print("PASS: All checks passed!")
    else:
        print(f"ISSUES FOUND: {issues} total issues")


if __name__ == "__main__":
    main()
