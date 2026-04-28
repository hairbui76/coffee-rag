"""Fix ground_truth_contexts in the RAGAS eval dataset.

Replaces fabricated bean names with attribute-based patterns derived from
actual data in beans_clean.parquet. Also validates that product_search /
similar_search ground_truth references attributes that exist in the DB.

Usage:
    python -m evaluation.fix_ground_truth
    python -m evaluation.fix_ground_truth --input ragas_eval_dataset.json --output ragas_eval_dataset_fixed.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_array(x: Any) -> bool:
    return isinstance(x, (list, np.ndarray))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BEANS_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"
DEFAULT_INPUT = ROOT / "ragas_eval_dataset.json"
DEFAULT_OUTPUT = ROOT / "ragas_eval_dataset.json"


def load_beans() -> pd.DataFrame:
    return pd.read_parquet(BEANS_PATH)


def _bean_to_context(row: pd.Series) -> str:
    """Build a ground_truth_context string from a real bean row."""
    parts = [f"Bean: {row.get('product_name', '')}"]
    if row.get("roaster_name"):
        parts.append(f"Roaster: {row['roaster_name']}")
    if row.get("origin"):
        parts.append(f"Origin: {row['origin']}")
    if row.get("country"):
        parts.append(f"Country: {row['country']}")
    if row.get("roast_level_clean") and row["roast_level_clean"] != "Unknown":
        parts.append(f"Roast: {row['roast_level_clean']}")
    flavors = row.get("flavor_notes_clean")
    if _is_array(flavors) and len(flavors):
        parts.append(f"Flavor: {', '.join(flavors[:6])}")
    proc = row.get("processing_clean")
    if _is_array(proc) and len(proc):
        parts.append(f"Processing: {', '.join(proc)}")
    species = row.get("species")
    if _is_array(species) and len(species):
        parts.append(f"Species: {', '.join(species)}")
    return ". ".join(parts) + "."


ROAST_NORMALIZE = {
    "medium light": "Medium-Light",
    "medium dark": "Medium-Dark",
    "medium-light": "Medium-Light",
    "medium-dark": "Medium-Dark",
    "light": "Light",
    "medium": "Medium",
    "dark": "Dark",
}


def find_matching_beans(
    beans: pd.DataFrame,
    entities: dict,
    limit: int = 3,
) -> list[str]:
    """Find real beans matching the expected entities and return context strings."""
    mask = pd.Series(True, index=beans.index)

    country = entities.get("country")
    if country:
        if isinstance(country, list):
            country_pat = "|".join(re.escape(c) for c in country)
        else:
            country_pat = re.escape(country)
        mask &= (
            beans["country"].str.contains(country_pat, case=False, na=False)
            | beans["origin"].str.contains(country_pat, case=False, na=False)
        )

    roast = entities.get("roast")
    if roast:
        if isinstance(roast, list):
            roast_normed = [ROAST_NORMALIZE.get(r.lower(), r) for r in roast]
            roast_pat = "|".join(re.escape(r) for r in roast_normed)
        else:
            roast_normed = ROAST_NORMALIZE.get(roast.lower(), roast)
            roast_pat = re.escape(roast_normed)
        mask &= beans["roast_level_clean"].str.contains(roast_pat, case=False, na=False)

    flavor = entities.get("flavor") or entities.get("flavor_notes")
    if flavor:
        if isinstance(flavor, str):
            flavor = [flavor]
        flat = beans["flavor_notes_clean"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        for f in flavor:
            mask &= flat.str.contains(re.escape(f.lower()), na=False)

    processing = entities.get("processing")
    if processing:
        if isinstance(processing, list):
            proc_pat = "|".join(re.escape(p) for p in processing)
        else:
            proc_pat = re.escape(processing)
        proc_flat = beans["processing_clean"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        mask &= proc_flat.str.contains(proc_pat.lower(), na=False)

    typology = entities.get("typology")
    if typology:
        if isinstance(typology, list):
            typo_pat = "|".join(re.escape(t) for t in typology)
        else:
            typo_pat = re.escape(typology)
        species_flat = beans["species"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        mask &= species_flat.str.contains(typo_pat.lower(), na=False)

    matched = beans[mask]
    if matched.empty:
        # Relax: drop processing, typology, roast in order
        for drop in ["processing", "typology", "roast"]:
            if entities.get(drop):
                relaxed = {k: v for k, v in entities.items() if k != drop}
                return find_matching_beans(beans, relaxed, limit)
        return []

    sample = matched.head(limit)
    return [_bean_to_context(row) for _, row in sample.iterrows()]


def fix_case(case: dict, beans: pd.DataFrame) -> dict:
    """Fix ground_truth_contexts for one case."""
    case = dict(case)  # shallow copy
    intent = case.get("intent", "")
    metadata = case.get("metadata", {})
    expected = metadata.get("expected_entities", {})

    if intent == "product_search":
        if expected:
            real_contexts = find_matching_beans(beans, expected, limit=3)
            if real_contexts:
                case["ground_truth_contexts"] = real_contexts
            else:
                case["ground_truth_contexts"] = [
                    f"No exact match found for entities: {expected}. "
                    "Ground truth describes general characteristics of this coffee category."
                ]

    elif intent == "similar_search":
        # Only fix similar_search contexts if they contain fabricated "Bean:" entries
        old_ctx = case.get("ground_truth_contexts", [])
        has_fabricated = any(ctx.startswith("Bean:") for ctx in old_ctx)
        if has_fabricated and expected:
            real_contexts = find_matching_beans(beans, expected, limit=3)
            if real_contexts:
                case["ground_truth_contexts"] = real_contexts
        # Otherwise keep original (Reference: patterns are fine)

    elif intent == "comparison":
        # Keep ground_truth_contexts as descriptive patterns
        # Replace fabricated "Bean:" entries with generic attribute patterns
        old_ctx = case.get("ground_truth_contexts", [])
        new_ctx = []
        for ctx in old_ctx:
            if ctx.startswith("Bean:") or ctx.startswith("Bean "):
                # Convert to attribute description
                new_ctx.append(ctx.replace("Bean:", "Example bean matching:").replace("Bean ", "Example bean: "))
            else:
                new_ctx.append(ctx)
        case["ground_truth_contexts"] = new_ctx

    elif intent == "knowledge_qa":
        # Knowledge QA ground_truth_contexts are already generic placeholders
        pass

    elif intent == "news_search":
        # Keep news contexts as-is (they reference article chunks)
        pass

    elif intent == "exploration":
        # Keep exploration contexts as-is
        pass

    elif intent == "edge_case":
        # Keep edge case contexts as-is
        pass

    return case


def main():
    parser = argparse.ArgumentParser(description="Fix ground_truth_contexts in RAGAS eval dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    print(f"Loading beans from {BEANS_PATH}...")
    beans = load_beans()
    print(f"  {len(beans)} beans loaded.")

    print(f"Loading dataset from {args.input}...")
    with args.input.open(encoding="utf-8") as f:
        cases = json.load(f)
    print(f"  {len(cases)} cases loaded.")

    fixed_count = 0
    fixed_cases = []
    for case in cases:
        fixed = fix_case(case, beans)
        if fixed.get("ground_truth_contexts") != case.get("ground_truth_contexts"):
            fixed_count += 1
            if args.dry_run:
                print(f"  CHANGED {case['id']} ({case['intent']})")
                print(f"    OLD: {case.get('ground_truth_contexts', [])[:1]}")
                print(f"    NEW: {fixed.get('ground_truth_contexts', [])[:1]}")
        fixed_cases.append(fixed)

    print(f"\n{fixed_count} cases updated.")

    if not args.dry_run:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(fixed_cases, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output}")
    else:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
