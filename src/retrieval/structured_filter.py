"""Module 2A: Filter beans by structured metadata fields."""

import re

import pandas as pd


def _as_str(value) -> str:
    """Coerce a value (possibly list) to a single regex-escaped string for str.contains."""
    if isinstance(value, list):
        return "|".join(re.escape(v) for v in value if v)
    return re.escape(str(value))


def structured_filter(beans_df: pd.DataFrame, entities: dict) -> pd.DataFrame:
    """Filter beans DataFrame using extracted entities.

    entities example:
        {"origin": "Vietnam", "roast": "Medium", "flavor": ["chocolate"],
         "typology": "Arabica", "processing": "Washed"}
    """
    mask = pd.Series(True, index=beans_df.index)

    if entities.get("origin"):
        mask &= beans_df["country"].str.contains(_as_str(entities["origin"]), case=False, na=False)

    if entities.get("roast"):
        mask &= beans_df["roast_level_clean"].str.contains(_as_str(entities["roast"]), case=False, na=False)

    if entities.get("flavor"):
        flavors = entities["flavor"]
        if isinstance(flavors, str):
            flavors = [flavors]
        flat = beans_df["flavor_notes_clean"].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
        for f in flavors:
            mask &= flat.str.contains(re.escape(f.lower()), na=False)

    if entities.get("typology"):
        species_flat = beans_df["species"].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
        mask &= species_flat.str.contains(_as_str(entities["typology"]).lower(), na=False)

    if entities.get("processing"):
        proc_flat = beans_df["processing_clean"].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
        mask &= proc_flat.str.contains(_as_str(entities["processing"]).lower(), na=False)

    return beans_df[mask]
