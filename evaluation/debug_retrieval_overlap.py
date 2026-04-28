"""Diagnostic: compare ground_truth beans (from dataset) vs current retrieval.

Usage (on server):
    python -m evaluation.debug_retrieval_overlap --dataset ragas_eval_dataset_v4.json --limit 30
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.pipeline import CoffeeRAG
from src.query.intent_classifier import classify_intent
from src.query.entity_extractor import extract_entities


def extract_bean_names_from_contexts(contexts: list[str]) -> list[str]:
    """Extract bean names from 'Bean: NAME. Roaster:...' format."""
    names = []
    for ctx in contexts:
        if ctx.startswith("Bean: "):
            end = ctx.find(". Roaster:")
            if end > 6:
                names.append(ctx[6:end])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "ragas_eval_dataset_v4.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--intent", default=None)
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8") as f:
        cases = json.load(f)

    if args.intent:
        cases = [c for c in cases if c["intent"] == args.intent]
    cases = cases[:args.limit]

    rag = CoffeeRAG()

    overlap_scores = []
    zero_overlap_cases = []
    intent_stats = defaultdict(list)

    for i, case in enumerate(cases):
        cid = case["id"]
        question = case["question"]
        intent = case["intent"]

        # Ground truth beans from dataset
        gt_beans = extract_bean_names_from_contexts(case.get("ground_truth_contexts", []))

        if not gt_beans:
            continue  # skip non-product cases

        # Expected entities (from dataset generation)
        expected_entities = case.get("metadata", {}).get("expected_entities", {})
        gen_retrieved = case.get("metadata", {}).get("retrieved_product_names", [])

        # Current retrieval
        ctx = rag.retrieve(question, top_k_beans=10, top_k_news=5)
        ret_beans = []
        if ctx["beans"] is not None and not ctx["beans"].empty:
            ret_beans = ctx["beans"]["product_name"].tolist()

        current_entities = ctx.get("entities", {})

        # Compare
        gt_set = set(gt_beans)
        ret_set = set(ret_beans)
        overlap = gt_set & ret_set
        overlap_pct = len(overlap) / max(len(gt_set), 1)
        overlap_scores.append(overlap_pct)
        intent_stats[intent].append(overlap_pct)

        # Entity comparison
        exp_ent_str = json.dumps(expected_entities, ensure_ascii=False) if expected_entities else "N/A"
        cur_ent_str = json.dumps({k: v for k, v in current_entities.items() if v}, ensure_ascii=False)

        status = "OK" if overlap_pct > 0.5 else "LOW" if overlap_pct > 0 else "ZERO"

        print(f"[{i+1}/{len(cases)}] {cid} ({intent}) overlap={len(overlap)}/{len(gt_set)} ({overlap_pct:.0%}) [{status}]")

        if overlap_pct < 0.5:
            print(f"  Q: {question[:100]}")
            print(f"  Expected entities: {exp_ent_str}")
            print(f"  Current entities:  {cur_ent_str}")
            print(f"  GT beans:  {gt_beans[:3]}")
            print(f"  Retrieved: {ret_beans[:5]}")
            if gen_retrieved:
                print(f"  Gen-time retrieved: {gen_retrieved[:3]}")
            print()

        if overlap_pct == 0:
            zero_overlap_cases.append(cid)

    # Summary
    print("=" * 60)
    print(f"SUMMARY ({len(overlap_scores)} cases with GT beans)")
    print(f"  Mean overlap:  {sum(overlap_scores)/max(len(overlap_scores),1):.1%}")
    print(f"  Zero overlap:  {len(zero_overlap_cases)} ({len(zero_overlap_cases)*100//max(len(overlap_scores),1)}%)")
    print(f"  Full overlap:  {sum(1 for s in overlap_scores if s >= 1.0)}")

    print("\n  By intent:")
    for intent in sorted(intent_stats):
        vals = intent_stats[intent]
        avg = sum(vals) / len(vals)
        zeros = sum(1 for v in vals if v == 0)
        print(f"    {intent:20s}  n={len(vals):3d}  overlap={avg:.1%}  zeros={zeros}")

    if zero_overlap_cases:
        print(f"\n  Zero-overlap IDs: {zero_overlap_cases[:20]}")


if __name__ == "__main__":
    main()
