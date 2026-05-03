"""Full RAG pipeline: Query → Understand → Retrieve → Re-rank → Generate."""

import os
import re

import numpy as np
import pandas as pd

from src.query.intent_classifier import classify_intent
from src.query.entity_extractor import extract_entities
from src.retrieval.semantic_search import SemanticSearcher
from src.retrieval.structured_filter import structured_filter
from src.retrieval.product_matcher import match_by_product_name
from src.retrieval.reranker import reciprocal_rank_fusion
from src.generation.prompt_templates import build_prompt
from src.generation.llm_client import get_client, generate_structured
from src.generation.schemas import CoffeeResponse


def _prioritize_matching(beans: "pd.DataFrame", entities: dict, top_k: int) -> "pd.DataFrame":
    """Re-order beans so those matching critical entities (origin, roast) come first.

    Scores each bean 0-2 based on origin and roast match, then sorts by
    (match_score DESC, original_rank ASC) and returns top_k.
    """
    origin = entities.get("origin", "") or ""
    roast = entities.get("roast", "") or ""

    scores = []
    for _, row in beans.iterrows():
        s = 0
        if origin:
            row_country = str(row.get("country", ""))
            row_origin = str(row.get("origin", ""))
            if re.search(re.escape(origin), row_country, re.IGNORECASE) or \
               re.search(re.escape(origin), row_origin, re.IGNORECASE):
                s += 1
        if roast:
            row_roast = str(row.get("roast_level_clean", ""))
            if row_roast.lower().strip() == roast.lower().strip():
                s += 1
        scores.append(s)

    beans = beans.copy()
    beans["_match_score"] = scores
    beans["_orig_rank"] = range(len(beans))
    beans = beans.sort_values(["_match_score", "_orig_rank"], ascending=[False, True])
    beans = beans.head(top_k).drop(columns=["_match_score", "_orig_rank"])
    return beans.reset_index(drop=True)


class CoffeeRAG:
    def __init__(self):
        self.searcher = SemanticSearcher()
        self.llm_client = get_client()
        self._bean_vecs = np.load(self.searcher._emb_dir / "beans_embeddings.npy")

    def retrieve(self, query: str,
                 top_k_beans: int | None = None,
                 top_k_news: int | None = None,
                 use_rrf: bool = True):
        top_k_beans = top_k_beans or int(os.getenv("TOP_K_BEANS", "5"))
        top_k_news = top_k_news or int(os.getenv("TOP_K_NEWS", "5"))
        intent = classify_intent(query)
        entities = extract_entities(query, client=self.llm_client)

        if intent == "exploration":
            return {
                "intent": intent,
                "entities": entities,
                "beans": self.searcher.beans,
                "news": self.searcher.news_chunks,
            }

        if intent == "edge_case":
            sem_beans = self.searcher.search_beans(query, top_k=top_k_beans)
            sem_news = self.searcher.search_news(query, top_k=2)
            return {
                "intent": intent,
                "entities": entities,
                "beans": sem_beans,
                "news": sem_news,
            }

        product_name = entities.get("product")
        roaster_name = entities.get("roaster")

        sem_beans = self.searcher.search_beans(query, top_k=top_k_beans * 3)
        news_candidate_k = top_k_news * 3
        sem_news = self.searcher.search_news(query, top_k=news_candidate_k)
        if use_rrf:
            bm25_news = self.searcher.search_news_bm25(query, top_k=news_candidate_k)
            news_lists = [sem_news, bm25_news] if not bm25_news.empty else [sem_news]
            if len(news_lists) > 1:
                sem_news = reciprocal_rank_fusion(
                    *news_lists, id_col="_chunk_id", top_k=news_candidate_k,
                )

        product_match = None
        if product_name:
            product_match = match_by_product_name(
                self.searcher.beans, product_name, roaster_name
            )

        struct_beans = None
        has_filters = any(entities.get(k) for k in ("origin", "roast", "flavor", "typology", "processing"))
        if intent in ("product_search", "similar_search", "comparison", "knowledge_qa") and has_filters:
            struct_beans = structured_filter(self.searcher.beans, entities)
            if not struct_beans.empty:
                struct_idx = struct_beans.index.tolist()
                qvec = self.searcher._encode_query(query)
                bean_vecs = self._bean_vecs[struct_idx]
                sims = (bean_vecs @ qvec.T).flatten()
                struct_beans = struct_beans.copy()
                struct_beans["_sim"] = sims
                struct_beans = struct_beans.sort_values("_sim", ascending=False).head(top_k_beans * 3)
                struct_beans = struct_beans.drop(columns=["_sim"])

        result_lists = [sem_beans]
        if product_match is not None and not product_match.empty:
            result_lists.insert(0, product_match)
        if struct_beans is not None and not struct_beans.empty:
            result_lists.append(struct_beans)

        if len(result_lists) > 1:
            if use_rrf:
                beans = reciprocal_rank_fusion(*result_lists, top_k=top_k_beans * 2)
            else:
                beans = pd.concat(result_lists, ignore_index=True) \
                          .drop_duplicates(subset="product_url", keep="first") \
                          .head(top_k_beans * 2) \
                          .reset_index(drop=True)
        else:
            beans = sem_beans.head(top_k_beans * 2)

        # Post-filter: prioritize beans matching critical entities (origin, roast),
        # then pad with remaining results if needed.
        if has_filters and intent in ("product_search", "similar_search"):
            beans = _prioritize_matching(beans, entities, top_k_beans)
        else:
            beans = beans.head(top_k_beans)

        return {
            "intent": intent,
            "entities": entities,
            "beans": beans,
            "news": sem_news.head(top_k_news) if intent in ("news_search", "knowledge_qa") else sem_news.head(2),
        }

    def ask(self, query: str) -> CoffeeResponse:
        ctx = self.retrieve(query)
        messages = build_prompt(query, ctx["beans"], ctx["news"])
        return generate_structured(messages, CoffeeResponse, client=self.llm_client)
