"""Module 2A: Filter beans by structured metadata fields."""

import re

import numpy as np
import pandas as pd


def _is_array(x) -> bool:
    return isinstance(x, (list, np.ndarray))


def _as_str(value) -> str:
    """Coerce a value (possibly list) to a single regex-escaped string for str.contains."""
    if isinstance(value, list):
        return "|".join(re.escape(v) for v in value if v)
    return re.escape(str(value))


MIN_RESULTS = 3

_ROAST_NORMALIZE = {
    "medium light": "Medium-Light",
    "medium dark": "Medium-Dark",
}


def _normalize_roast(value: str) -> str:
    return _ROAST_NORMALIZE.get(value.lower().strip(), value)


def _apply_filters(beans_df: pd.DataFrame, entities: dict, skip: set[str] | None = None) -> pd.DataFrame:
    """Apply entity filters with optional skip set for relaxation."""
    skip = skip or set()
    mask = pd.Series(True, index=beans_df.index)

    if entities.get("origin") and "origin" not in skip:
        origin_pat = _as_str(entities["origin"])
        mask &= (
            beans_df["country"].str.contains(origin_pat, case=False, na=False) |
            beans_df["origin"].str.contains(origin_pat, case=False, na=False)
        )

    if entities.get("roast") and "roast" not in skip:
        roast_val = _normalize_roast(entities["roast"])
        mask &= beans_df["roast_level_clean"].str.contains(_as_str(roast_val), case=False, na=False)

    if entities.get("flavor") and "flavor" not in skip:
        flavors = entities["flavor"]
        if isinstance(flavors, str):
            flavors = [flavors]
        flat = beans_df["flavor_notes_clean"].apply(lambda x: " ".join(x).lower() if _is_array(x) else "")
        for f in flavors:
            mask &= flat.str.contains(re.escape(f.lower()), na=False)

    if entities.get("typology") and "typology" not in skip:
        species_flat = beans_df["species"].apply(lambda x: " ".join(x).lower() if _is_array(x) else "")
        mask &= species_flat.str.contains(_as_str(entities["typology"]).lower(), na=False)

    if entities.get("processing") and "processing" not in skip:
        proc_flat = beans_df["processing_clean"].apply(lambda x: " ".join(x).lower() if _is_array(x) else "")
        mask &= proc_flat.str.contains(_as_str(entities["processing"]).lower(), na=False)

    return beans_df[mask]


def structured_filter(beans_df: pd.DataFrame, entities: dict) -> pd.DataFrame:
    """Filter beans DataFrame using extracted entities.

    entities example:
        {"origin": "Vietnam", "roast": "Medium", "flavor": ["chocolate"],
         "typology": "Arabica", "processing": "Washed"}

    If strict AND filtering returns fewer than MIN_RESULTS, progressively
    relax by dropping the least important filter (processing → typology)
    until enough results are found. Roast and origin are never relaxed
    because they are critical user constraints.
    """
    result = _apply_filters(beans_df, entities)
    if len(result) >= MIN_RESULTS:
        return result

    relaxation_order = ["processing", "typology"]
    skipped: set[str] = set()
    for field in relaxation_order:
        if not entities.get(field):
            continue
        skipped.add(field)
        result = _apply_filters(beans_df, entities, skip=skipped)
        if len(result) >= MIN_RESULTS:
            return result

    return result
