"""Module 2B: Semantic search using FAISS + sentence-transformers."""

import logging
import os
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

EMB_DIR = ROOT / "data" / "embeddings"
DATA_DIR = ROOT / "data" / "processed"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

_TOKEN_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + whitespace split.

    Suboptimal for VI (no word segmentation) but fine for BM25's main job
    on news: matching proper nouns and rare tokens that dense retrieval misses.
    """
    if not text:
        return []
    cleaned = _TOKEN_RE.sub(" ", str(text).lower())
    return cleaned.split()


class SemanticSearcher:
    def __init__(self, model_name: str = MODEL_NAME):
        if model_name.startswith("text-embedding-"):
            raise ValueError(
                "EMBEDDING_MODEL is used by SentenceTransformer for FAISS retrieval, "
                f"but got OpenAI model '{model_name}'. Set EMBEDDING_MODEL=BAAI/bge-m3 "
                "to match the current local FAISS indices. RAGAS_EMBEDDING_MODEL can "
                "remain text-embedding-3-small for Ragas evaluator metrics."
            )
        self.model = SentenceTransformer(model_name)
        self._emb_dir = EMB_DIR

        self.beans = pd.read_parquet(DATA_DIR / "beans_clean.parquet")
        self.news_chunks = pd.read_parquet(DATA_DIR / "news_chunks.parquet")

        self.beans_index = faiss.read_index(str(EMB_DIR / "beans.index"))
        self.news_index = faiss.read_index(str(EMB_DIR / "news.index"))
        self._validate_index_dimensions(model_name)

        self._bm25_news = self._build_bm25_news()

    def _validate_index_dimensions(self, model_name: str) -> None:
        model_dim = self.model.get_sentence_embedding_dimension()
        if model_dim is None:
            model_dim = self.model.get_embedding_dimension()

        mismatches = []
        for label, index in (("beans", self.beans_index), ("news", self.news_index)):
            if index.d != model_dim:
                mismatches.append(f"{label}.index dim={index.d}")

        if mismatches:
            raise ValueError(
                "FAISS index dimension does not match EMBEDDING_MODEL. "
                f"EMBEDDING_MODEL={model_name!r} produces dim={model_dim}, but "
                f"{', '.join(mismatches)}. Rebuild local embeddings with: "
                "python -m src.preprocessing.build_embeddings --force"
            )

    def _build_bm25_news(self) -> BM25Okapi:
        title = self.news_chunks["title"].fillna("").astype(str)
        text = self.news_chunks["text"].fillna("").astype(str)
        docs = (title + " " + text).tolist()
        tokenized = [_tokenize(d) for d in docs]
        return BM25Okapi(tokenized)

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True)
        return np.ascontiguousarray(vec, dtype=np.float32)

    def search_beans(self, query: str, top_k: int = 10) -> pd.DataFrame:
        qvec = self._encode_query(query)
        scores, indices = self.beans_index.search(qvec, top_k)
        results = self.beans.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return results.reset_index(drop=True)

    def search_news(self, query: str, top_k: int = 5) -> pd.DataFrame:
        qvec = self._encode_query(query)
        scores, indices = self.news_index.search(qvec, top_k)
        positions = indices[0]
        results = self.news_chunks.iloc[positions].copy()
        results["_chunk_id"] = positions
        results["score"] = scores[0]
        return results.reset_index(drop=True)

    def search_news_bm25(self, query: str, top_k: int = 5,
                         idf_threshold: float = 2.5) -> pd.DataFrame:
        """BM25 search over news chunks.

        Filters query tokens by IDF threshold to prevent multilingual stopword
        bleed: VI queries contain many common VI tokens that, when matched
        against VI articles, accumulate enough BM25 score to bury proper-noun
        signals (Turabo, Bucharest) that only appear in 1-2 EN articles.
        Threshold 2.5 drops universal coffee-corpus noise ("coffee"≈1.96,
        "cà"/"phê"≈2.28) while keeping discriminative tokens like "Highlands"
        (3.16) and proper nouns ("Turabo" 7.80, "Bucharest" 6.83).
        """
        tokens = _tokenize(query)
        if not tokens:
            return self._empty_news_bm25()
        rare_tokens = [t for t in tokens if self._bm25_news.idf.get(t, 0.0) >= idf_threshold]
        # Fallback to all tokens if filter wipes everything (e.g. all-common query).
        query_tokens = rare_tokens or tokens
        scores = self._bm25_news.get_scores(query_tokens)
        positions = np.argsort(scores)[::-1][:top_k]
        positions = [int(p) for p in positions if scores[p] > 0]
        if not positions:
            return self._empty_news_bm25()
        results = self.news_chunks.iloc[positions].copy()
        results["_chunk_id"] = positions
        results["bm25_score"] = scores[positions]
        return results.reset_index(drop=True)

    def _empty_news_bm25(self) -> pd.DataFrame:
        return self.news_chunks.iloc[0:0].assign(_chunk_id=[], bm25_score=[]).reset_index(drop=True)
