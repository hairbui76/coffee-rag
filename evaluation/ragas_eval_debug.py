"""Debug script for investigating 0-score RAGAS evaluation cases.

Runs retrieval for a list of case IDs and prints detailed diagnostics:
  - Dataset entry info
  - Pipeline retrieval results (intent, entities, beans, news)
  - Overlap analysis between expected and retrieved products
  - Ground truth vs retrieved context comparison

Usage:
    python -m evaluation.ragas_eval_debug
    python -m evaluation.ragas_eval_debug --ids PS_003 PS_006 PS_007
    python -m evaluation.ragas_eval_debug --intent exploration --limit 5
    python -m evaluation.ragas_eval_debug --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── Default case IDs to debug ────────────────────────────────
DEFAULT_DEBUG_IDS = [
    # Product Search (PS)
    "PS_003", "PS_006", "PS_007", "PS_009", "PS_011", "PS_012",
    "PS_016", "PS_018", "PS_019", "PS_022", "PS_023", "PS_025",
    "PS_026", "PS_027", "PS_028", "PS_030", "PS_031", "PS_037",
    "PS_038", "PS_040", "PS_042", "PS_043", "PS_051", "PS_055",
    "PS_057", "PS_058", "PS_062", "PS_063", "PS_065", "PS_067",
    "PS_070", "PS_071", "PS_072", "PS_074", "PS_076", "PS_077",
    "PS_079", "PS_084", "PS_086", "PS_090", "PS_091", "PS_094",
    "PS_095", "PS_096", "PS_097", "PS_098", "PS_099",
    # Comparison (CP)
    "CP_002", "CP_008", "CP_012", "CP_023", "CP_028", "CP_030",
    "CP_034", "CP_040", "CP_044", "CP_046", "CP_047", "CP_051",
    "CP_052", "CP_054", "CP_055", "CP_058",
    # Knowledge QA (KQ)
    "KQ_010", "KQ_011", "KQ_012", "KQ_014", "KQ_023", "KQ_039",
    "KQ_046", "KQ_057", "KQ_059", "KQ_074",
    # News Search (NS)
    "NS_008", "NS_010", "NS_013", "NS_018", "NS_040",
    # Exploration (EX)
    "EX_001", "EX_002", "EX_003", "EX_004", "EX_005", "EX_006",
    "EX_007", "EX_008", "EX_009", "EX_011", "EX_012", "EX_015",
    "EX_016", "EX_017", "EX_018", "EX_021", "EX_022", "EX_023",
    "EX_024", "EX_025", "EX_026", "EX_027", "EX_028", "EX_030",
    "EX_031", "EX_032", "EX_033", "EX_034", "EX_035", "EX_037",
    "EX_038", "EX_040",
    # Edge Case (EC)
    "EC_001", "EC_002", "EC_003", "EC_004", "EC_010",
]


def load_dataset(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"]: c for c in data}


def print_section(title: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def debug_case(rag, case: dict, verbose: bool = False):
    case_id = case["id"]
    intent = case.get("intent", "")
    question = case["question"]
    metadata = case.get("metadata", {})
    expected_names = metadata.get("real_product_names", [])
    gt_contexts = case.get("ground_truth_contexts", [])

    print_section(f"{case_id}  ({intent}, {case.get('difficulty', '')}, {case.get('language', '')})")
    print(f"Question: {question}")

    if verbose:
        print(f"\nGround Truth:\n  {case.get('ground_truth', '')[:300]}...")
        if gt_contexts:
            print(f"\nGround Truth Contexts ({len(gt_contexts)}):")
            for i, ctx in enumerate(gt_contexts):
                print(f"  [{i}] {ctx[:150]}...")

    # ── Run retrieval ──
    ctx = rag.retrieve(question, top_k_beans=10, top_k_news=5)

    detected_intent = ctx.get("intent", "")
    entities = ctx.get("entities", {})
    beans = ctx.get("beans")
    news = ctx.get("news")
    bean_count = len(beans) if beans is not None and not getattr(beans, "empty", True) else 0
    news_count = len(news) if news is not None and not getattr(news, "empty", True) else 0

    print(f"\nDetected intent : {detected_intent}" +
          (f"  ⚠ expected={intent}" if detected_intent != intent else ""))
    print(f"Entities        : {json.dumps(entities, ensure_ascii=False)}")
    print(f"Retrieved       : beans={bean_count}, news={news_count}")

    if detected_intent == "exploration":
        print(f"  (exploration: full dataset returned, showing first 10 beans)")

    # ── Retrieved beans ──
    display_beans = beans.head(10) if beans is not None and bean_count > 10 else beans
    retrieved_names = []
    if display_beans is not None and not display_beans.empty:
        print("\nRetrieved beans:")
        for i, (_, row) in enumerate(display_beans.iterrows()):
            name = str(row.get("product_name", ""))
            roaster = str(row.get("roaster_name", ""))
            country = str(row.get("country", ""))
            roast = str(row.get("roast_level_clean", "") or row.get("roast_level", ""))
            flavors = row.get("flavor_notes_clean", [])
            if hasattr(flavors, "tolist"):
                flavors = flavors.tolist()
            retrieved_names.append(name)
            marker = " ✓" if name in expected_names else ""
            print(f"  [{i+1:>2}] {name} | {roaster} | {country} | {roast} | {flavors}{marker}")

    # ── Retrieved news ──
    if verbose and news is not None and not news.empty:
        print("\nRetrieved news:")
        for i, (_, row) in enumerate(news.iterrows()):
            title = str(row.get("title", ""))
            source = str(row.get("source", ""))
            print(f"  [{i+1}] {title} | {source}")

    # ── Overlap analysis ──
    if expected_names:
        overlap = set(expected_names) & set(retrieved_names)
        missing = set(expected_names) - set(retrieved_names)
        print(f"\nOverlap: {len(overlap)}/{len(expected_names)} expected products found")
        if overlap:
            print(f"  Found  : {sorted(overlap)}")
        if missing:
            print(f"  Missing: {sorted(missing)}")
    elif intent in ("exploration", "edge_case"):
        print(f"\n⚠ Intent '{intent}' — no expected product names (aggregate/non-retrieval query)")
    else:
        print("\n⚠ No expected product names in metadata")

    # ── Entity quality check ──
    expected_entities = metadata.get("expected_entities", {})
    if expected_entities:
        print(f"\nEntity check:")
        for key, expected_val in expected_entities.items():
            got = entities.get(key)
            match = "✓" if got == expected_val else "✗"
            print(f"  {match} {key}: expected={expected_val}, got={got}")

    return {
        "id": case_id,
        "intent": intent,
        "detected_intent": detected_intent,
        "bean_count": bean_count,
        "news_count": news_count,
        "expected_products": len(expected_names),
        "found_products": len(set(expected_names) & set(retrieved_names)) if expected_names else None,
        "entities_match": all(
            entities.get(k) == v for k, v in expected_entities.items()
        ) if expected_entities else None,
    }


def print_summary(results: list[dict]):
    print_section("DEBUG SUMMARY", "=", 70)

    total = len(results)
    intent_groups: dict[str, list[dict]] = {}
    for r in results:
        intent_groups.setdefault(r["intent"], []).append(r)

    for intent, group in sorted(intent_groups.items()):
        n = len(group)
        with_expected = [r for r in group if r["expected_products"] is not None and r["expected_products"] > 0]
        if with_expected:
            found_any = sum(1 for r in with_expected if r["found_products"] and r["found_products"] > 0)
            found_all = sum(1 for r in with_expected if r["found_products"] == r["expected_products"])
            avg_found = sum(r["found_products"] or 0 for r in with_expected) / len(with_expected)
            avg_expected = sum(r["expected_products"] for r in with_expected) / len(with_expected)
            print(f"\n  {intent} ({n} cases, {len(with_expected)} with expected products):")
            print(f"    Found at least 1 product : {found_any}/{len(with_expected)}")
            print(f"    Found all products       : {found_all}/{len(with_expected)}")
            print(f"    Avg found/expected       : {avg_found:.1f}/{avg_expected:.1f}")
        else:
            print(f"\n  {intent} ({n} cases, no expected products)")

        ent_cases = [r for r in group if r["entities_match"] is not None]
        if ent_cases:
            ent_ok = sum(1 for r in ent_cases if r["entities_match"])
            print(f"    Entity extraction correct: {ent_ok}/{len(ent_cases)}")

        intent_mismatch = sum(1 for r in group if r["detected_intent"] != r["intent"])
        if intent_mismatch:
            print(f"    Intent mismatch          : {intent_mismatch}/{n}")

    no_beans = sum(1 for r in results if r["bean_count"] == 0)
    no_news = sum(1 for r in results if r["news_count"] == 0)
    if no_beans:
        print(f"\n  ⚠ {no_beans} cases returned 0 beans")
    if no_news:
        print(f"\n  ⚠ {no_news} cases returned 0 news")

    print(f"\n  Total cases debugged: {total}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug 0-score RAGAS eval cases.")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "ragas_eval_dataset_v2.json")
    parser.add_argument("--ids", nargs="+",
                        help="Specific case IDs to debug. Default: built-in list of 0-score cases.")
    parser.add_argument("--intent",
                        help="Filter by intent (e.g. exploration, product_search)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of cases to debug (0 = all matching)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show ground truth contexts and retrieved news")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    ids = args.ids or DEFAULT_DEBUG_IDS

    cases = [dataset[cid] for cid in ids if cid in dataset]
    missing = [cid for cid in ids if cid not in dataset]
    if missing:
        print(f"⚠ {len(missing)} IDs not found in dataset: {missing[:10]}...")

    if args.intent:
        cases = [c for c in cases if c.get("intent") == args.intent]
    if args.limit > 0:
        cases = cases[:args.limit]

    if not cases:
        print("No cases matched. Exiting.")
        return

    print(f"Loading CoffeeRAG pipeline...")
    from src.pipeline import CoffeeRAG
    rag = CoffeeRAG()

    print(f"Debugging {len(cases)} cases...\n")
    results = []
    for case in cases:
        try:
            result = debug_case(rag, case, verbose=args.verbose)
            results.append(result)
        except Exception as exc:
            print(f"\n  ❌ ERROR on {case['id']}: {exc}")
            results.append({
                "id": case["id"], "intent": case.get("intent", ""),
                "detected_intent": "", "bean_count": 0, "news_count": 0,
                "expected_products": None, "found_products": None,
                "entities_match": None,
            })

    print_summary(results)


if __name__ == "__main__":
    main()
