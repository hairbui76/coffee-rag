"""Draw a stratified annotation sample from a ``*_contexts.jsonl`` dump.

Strata: intent × difficulty × language (3 × 3 × 2 = 18). With ``--per-stratum 10``
that is 180 cases. The sample hides ``relevant_ids`` and the reference text from
annotators, so labels are not anchored on the template ground truth.

Usage:
    python -m evaluation.annotation.sample_cases \
        --contexts evaluation/results/v6/bge_rrf_r1_contexts.jsonl \
        --out evaluation/annotation/annotation_sample.json --per-stratum 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FIELD_RE = re.compile(r"(Bean|Roaster|Origin|Country|Roast|Flavor|Processing|Species|Description|URL|Article|Source|Date|Content):\s*")


def _fields(context: str) -> dict[str, str]:
    """Split 'Key: value. Key: value.' context strings into a dict for display."""
    parts = _FIELD_RE.split(context)
    out: dict[str, str] = {}
    # parts = ['', 'Bean', 'name. ', 'Roaster', 'x. ', ...]
    for i in range(1, len(parts) - 1, 2):
        key = parts[i]
        val = parts[i + 1].strip()
        if key not in ("Description", "Content"):
            val = val.rstrip(". ")
        out[key] = val
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("evaluation/annotation/annotation_sample.json"))
    p.add_argument("--per-stratum", type=int, default=10)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    entries = [json.loads(l) for l in args.contexts.read_text(encoding="utf-8").splitlines() if l.strip()]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        strata[(e.get("intent"), e.get("difficulty"), e.get("language"))].append(e)

    sample = []
    for key in sorted(strata):
        pool = strata[key]
        rng.shuffle(pool)
        take = pool[:args.per_stratum]
        if len(take) < args.per_stratum:
            print(f"  stratum {key}: only {len(take)} cases")
        sample += take

    rng.shuffle(sample)
    out = []
    for e in sample:
        ctxs = e.get("retrieved_contexts", [])[:args.k]
        ids = e.get("retrieved_ids", [])[:args.k]
        out.append({
            "id": e["id"],
            "intent": e.get("intent"),
            "difficulty": e.get("difficulty"),
            "language": e.get("language"),
            "question": e["question"],
            "contexts": [
                {"rank": i + 1, "doc_id": ids[i] if i < len(ids) else "", "fields": _fields(c), "raw": c}
                for i, c in enumerate(ctxs)
            ],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": str(args.contexts), "k": args.k, "seed": args.seed,
                                    "cases": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}: {len(out)} cases, {sum(len(c['contexts']) for c in out)} contexts to label")


if __name__ == "__main__":
    main()
