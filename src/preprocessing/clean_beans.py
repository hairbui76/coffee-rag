"""Clean raw coffee_beans.json → data/processed/beans_clean.parquet"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "coffee_beans.json"
OUT_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"

ROAST_MAP = {
    "Light": "Light",
    "Medium Light": "Medium-Light",
    "Medium": "Medium",
    "Medium Dark": "Medium-Dark",
    "Dark": "Dark",
    "—": "Unknown",
    "": "Unknown",
}

NAV_JUNK = {"Back to Bean Discovery", "View", ""}


def _clean_list_field(items: list[str]) -> list[str]:
    """Remove entries that contain '\\nORIGIN' or stray newlines from crawling."""
    cleaned = []
    for item in items:
        if "\n" in item:
            parts = item.split("\n")
            for p in parts:
                p = p.strip()
                if p and not p.startswith("ORIGIN"):
                    cleaned.append(p)
        else:
            cleaned.append(item.strip())
    return [c for c in cleaned if c]


def _parse_typology(typo_list: list[str]) -> tuple[list[str], list[str]]:
    """Split typology into species (Arabica/Robusta) and cultivar names."""
    species, cultivars = [], []
    for t in typo_list:
        parts = [p.strip() for p in t.split("\n") if p.strip()]
        if parts:
            species.append(parts[0])
        if len(parts) > 1:
            cultivars.extend(parts[1:])
    return list(dict.fromkeys(species)), list(dict.fromkeys(cultivars))


def _clean_similar(sim_list: list[dict]) -> list[dict]:
    return [s for s in sim_list if s.get("name") not in NAV_JUNK]


def _build_document(row: pd.Series) -> str:
    """Build composite text for embedding."""
    parts = [
        row["product_name"],
        row.get("about_description") or "",
        f"Origin: {row.get('origin') or ''}",
        f"Country: {row.get('country') or ''}",
        f"Roast: {row.get('roast_level_clean') or ''}",
        f"Flavor: {', '.join(row.get('flavor_notes_clean') or [])}",
        f"Processing: {', '.join(row.get('processing_clean') or [])}",
        f"Type: {', '.join(row.get('species') or [])}",
    ]
    return ". ".join(p for p in parts if p and p.split(": ", 1)[-1])


def clean_beans() -> pd.DataFrame:
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    df["flavor_notes_clean"] = df["flavor_notes"].apply(
        lambda x: _clean_list_field(x) if isinstance(x, list) else []
    )
    df["processing_clean"] = df["processing"].apply(
        lambda x: _clean_list_field(x) if isinstance(x, list) else []
    )

    parsed = df["typology"].apply(
        lambda x: _parse_typology(x) if isinstance(x, list) else ([], [])
    )
    df["species"] = parsed.apply(lambda x: x[0])
    df["cultivars"] = parsed.apply(lambda x: x[1])

    df["roast_level_clean"] = (
        df["roast_level"].fillna("").map(ROAST_MAP).fillna("Unknown")
    )

    df["similar_products_clean"] = df["similar_products"].apply(
        lambda x: _clean_similar(x) if isinstance(x, list) else []
    )

    df["document_text"] = df.apply(_build_document, axis=1)

    keep_cols = [
        "product_name", "product_url", "roaster_name",
        "about_description", "origin", "country",
        "flavor_notes_clean", "processing_clean",
        "species", "cultivars", "roast_level_clean",
        "buy_links", "similar_products_clean", "document_text",
    ]
    df = df[keep_cols].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} beans -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    clean_beans()
