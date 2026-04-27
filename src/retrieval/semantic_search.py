"""Module 2B: Semantic search using FAISS + sentence-transformers."""

import logging
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

EMB_DIR = ROOT / "data" / "embeddings"
DATA_DIR = ROOT / "data" / "processed"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class SemanticSearcher:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

        self.beans = pd.read_parquet(DATA_DIR / "beans_clean.parquet")
        self.news_chunks = pd.read_parquet(DATA_DIR / "news_chunks.parquet")

        self.beans_index = faiss.read_index(str(EMB_DIR / "beans.index"))
        self.news_index = faiss.read_index(str(EMB_DIR / "news.index"))

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
        results = self.news_chunks.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return results.reset_index(drop=True)
