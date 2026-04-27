"""Debug PS_002: check dataset entry + run pipeline retrieval."""
import json, sys, os
sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(".env")

# 1. Load dataset entry
with open("ragas_eval_dataset_v2.json", encoding="utf-8") as f:
    data = json.load(f)
case = [c for c in data if c["id"] == "PS_002"][0]

print("=" * 70)
print("DATASET ENTRY: PS_002")
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

expected_names = case["metadata"].get("real_product_names", [])
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
