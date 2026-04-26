"""Module 3: Hybrid re-ranking via Reciprocal Rank Fusion (RRF)."""

import pandas as pd


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    *result_lists: pd.DataFrame,
    id_col: str = "product_url",
    top_k: int = 10,
    k: int = 60,
) -> pd.DataFrame:
    """Merge multiple ranked result DataFrames using RRF.

    Each DataFrame must have `id_col` to identify unique items.
    Returns the fused top-K results sorted by RRF score.
    """
    scores: dict[str, float] = {}
    row_map: dict[str, pd.Series] = {}

    for result_df in result_lists:
        if result_df is None or result_df.empty:
            continue
        for rank, (_, row) in enumerate(result_df.iterrows(), start=1):
            doc_id = row[id_col]
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank, k)
            if doc_id not in row_map:
                row_map[doc_id] = row

    if not scores:
        return pd.DataFrame()

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    rows = [row_map[did] for did in sorted_ids]
    fused = pd.DataFrame(rows).reset_index(drop=True)
    fused["rrf_score"] = [scores[did] for did in sorted_ids]
    return fused
