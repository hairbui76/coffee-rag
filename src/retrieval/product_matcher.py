"""Product name matching: exact and fuzzy lookup on product_name and roaster_name."""

import re
from difflib import SequenceMatcher

import pandas as pd


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"[''""\"'\u2018\u2019\u201C\u201D]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def match_by_product_name(
    beans_df: pd.DataFrame,
    product_name: str | None = None,
    roaster_name: str | None = None,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Find beans matching a specific product name (and optionally roaster).

    Returns matching rows sorted by match quality (best first).
    Empty DataFrame if no matches above threshold.
    """
    if not product_name:
        return pd.DataFrame()

    query_norm = _normalize(product_name)
    if len(query_norm) < 2:
        return pd.DataFrame()

    names = beans_df["product_name"].fillna("").apply(_normalize)

    exact_mask = names.str.contains(re.escape(query_norm), na=False)
    if exact_mask.any():
        result = beans_df[exact_mask].copy()
        if roaster_name:
            roaster_norm = _normalize(roaster_name)
            roaster_mask = result["roaster_name"].fillna("").apply(_normalize).str.contains(
                re.escape(roaster_norm), na=False
            )
            if roaster_mask.any():
                return result[roaster_mask].reset_index(drop=True)
        return result.reset_index(drop=True)

    scores = names.apply(lambda n: SequenceMatcher(None, query_norm, n).ratio())
    above = scores[scores >= threshold]
    if above.empty:
        return pd.DataFrame()

    result = beans_df.loc[above.index].copy()
    result["match_score"] = above.values
    result = result.sort_values("match_score", ascending=False)

    if roaster_name:
        roaster_norm = _normalize(roaster_name)
        roaster_scores = result["roaster_name"].fillna("").apply(
            lambda r: SequenceMatcher(None, roaster_norm, _normalize(r)).ratio()
        )
        result["match_score"] = result["match_score"] * 0.7 + roaster_scores * 0.3
        result = result.sort_values("match_score", ascending=False)

    return result.reset_index(drop=True)
