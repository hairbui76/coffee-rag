"""Debug a single case: dataset entry + pipeline retrieval + RAGAS inputs.

Usage:
    python -m evaluation._debug_ps002 PS_001
    python -m evaluation._debug_ps002 PS_001 --dataset ragas_eval_dataset_v4.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("case_id", nargs="?", default="PS_001")
parser.add_argument("--dataset", type=Path, default=ROOT / "ragas_eval_dataset.json")
args = parser.parse_args()

# 1. Load dataset entry
with open(args.dataset, encoding="utf-8") as f:
    data = json.load(f)
matches = [c for c in data if c["id"] == args.case_id]
if not matches:
    print(f"Case {args.case_id} not found in {args.dataset}")
    sys.exit(1)
case = matches[0]

print("=" * 70)
print(f"DATASET ENTRY: {args.case_id}")
print("=" * 70)
print(f"Question: {case['question']}")
print(f"Language: {case['language']}, Difficulty: {case['difficulty']}")
print(f"Metadata: {json.dumps(case['metadata'], ensure_ascii=False, indent=2)}")
print(f"\nGround Truth:\n{case['ground_truth']}")
print(f"\nGround Truth Contexts ({len(case['ground_truth_contexts'])} items):")
for i, ctx in enumerate(case["ground_truth_contexts"]):
    print(f"  [{i}] {ctx[:200]}...")

# 2. Run pipeline retrieval
print("\n" + "=" * 70)
print("PIPELINE RETRIEVAL")
print("=" * 70)

from src.pipeline import CoffeeRAG
rag = CoffeeRAG()

question = case["question"]
ctx = rag.retrieve(question, top_k_beans=10, top_k_news=5)

print(f"Detected intent: {ctx.get('intent')}")
print(f"Extracted entities: {json.dumps(ctx.get('entities', {}), ensure_ascii=False)}")

beans = ctx.get("beans")
news = ctx.get("news")
print(f"Retrieved beans: {len(beans) if beans is not None else 0}")
print(f"Retrieved news: {len(news) if news is not None else 0}")

if beans is not None and not beans.empty:
    print("\nRetrieved beans:")
    for i, (_, row) in enumerate(beans.iterrows()):
        name = row.get("product_name", "")
        roaster = row.get("roaster_name", "")
        country = row.get("country", "")
        roast = row.get("roast_level_clean", "")
        flavors = row.get("flavor_notes_clean", [])
        if hasattr(flavors, "tolist"):
            flavors = flavors.tolist()
        print(f"  [{i+1}] {name} | {roaster} | {country} | {roast} | {flavors}")

# 3. Check overlap
print("\n" + "=" * 70)
print("OVERLAP ANALYSIS")
print("=" * 70)

expected_names = case["metadata"].get("real_product_names",
                    case["metadata"].get("retrieved_product_names", []))
retrieved_names = [row["product_name"] for _, row in beans.iterrows()] if beans is not None else []

print(f"Expected products: {expected_names}")
print(f"Retrieved products: {retrieved_names}")

overlap = set(expected_names) & set(retrieved_names)
print(f"\nOverlap: {len(overlap)} / {len(expected_names)} expected products found")
if overlap:
    print(f"  Found: {list(overlap)}")
missing = set(expected_names) - set(retrieved_names)
if missing:
    print(f"  Missing: {list(missing)}")

# 4. Show RAGAS inputs
print("\n" + "=" * 70)
print("RAGAS INPUTS (what the LLM judge sees)")
print("=" * 70)

from evaluation.ragas_eval import build_retrieved_contexts

eval_ctx = dict(ctx)
if beans is not None and len(beans) > 20:
    eval_ctx["beans"] = beans.head(20)
if news is not None and len(news) > 20:
    eval_ctx["news"] = news.head(20)
retrieved_contexts = build_retrieved_contexts(eval_ctx)

print(f"\nuser_input (question):")
print(f"  {question}")
print(f"\nreference (ground_truth):")
print(f"  {case['ground_truth']}")
print(f"\nretrieved_contexts ({len(retrieved_contexts)} items):")
for i, rc in enumerate(retrieved_contexts):
    # Highlight relevance signals
    q_lower = question.lower()
    signals = []
    if "colombia" in rc.lower():
        signals.append("Colombia✓")
    if "medium" in rc.lower() and "dark" not in rc.lower() and "light" not in rc.lower():
        signals.append("Medium✓")
    elif "medium" in rc.lower():
        roast_part = rc.split("Roast:")[1].split(".")[0].strip() if "Roast:" in rc else ""
        signals.append(f"Roast:{roast_part}")
    if any(w in rc.lower() for w in ["chocolate", "cocoa", "cacao"]):
        signals.append("Choco✓")
    if any(w in rc.lower() for w in ["nutty", "nut", "hazelnut", "almond", "hạt dẻ"]):
        signals.append("Nutty✓")
    signal_str = " | ".join(signals) if signals else "NO MATCH"
    print(f"\n  [{i+1}] [{signal_str}]")
    print(f"      {rc[:250]}")

# Summary: how many contexts are fully relevant?
print("\n" + "=" * 70)
print("RELEVANCE SUMMARY")
print("=" * 70)
full_match = 0
partial_match = 0
no_match = 0
for rc in retrieved_contexts:
    rc_l = rc.lower()
    has_country = "colombia" in rc_l
    has_roast = "medium" in rc_l and "dark" not in rc_l.split("roast:")[1].split(".")[0] if "roast:" in rc_l else False
    has_choco = any(w in rc_l for w in ["chocolate", "cocoa", "cacao"])
    has_nutty = any(w in rc_l for w in ["nutty", "nut", "hazelnut", "almond"])
    score = sum([has_country, has_roast, has_choco, has_nutty])
    if score >= 3:
        full_match += 1
    elif score >= 1:
        partial_match += 1
    else:
        no_match += 1

total = len(retrieved_contexts)
print(f"  Full match (≥3/4 criteria):   {full_match}/{total}")
print(f"  Partial match (1-2 criteria): {partial_match}/{total}")
print(f"  No match (0 criteria):        {no_match}/{total}")
print(f"\n  → Expected CP ≈ {full_match/max(total,1):.2f} (rough estimate)")
