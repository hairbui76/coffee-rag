"""Embed cleaned data and build FAISS indices."""

import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

BEANS_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"
CHUNKS_PATH = ROOT / "data" / "processed" / "news_chunks.parquet"
EMB_DIR = ROOT / "data" / "embeddings"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


def build_all(model_name: str = MODEL_NAME):
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # --- Beans ---
    beans = pd.read_parquet(BEANS_PATH)
    texts = beans["document_text"].fillna("").tolist()
    print(f"Encoding {len(texts)} bean documents …")
    bean_vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    bean_vecs = np.ascontiguousarray(bean_vecs, dtype=np.float32)

    np.save(EMB_DIR / "beans_embeddings.npy", bean_vecs)
    index_beans = faiss.IndexFlatIP(bean_vecs.shape[1])
    index_beans.add(bean_vecs)
    faiss.write_index(index_beans, str(EMB_DIR / "beans.index"))
    print(f"Beans index: {index_beans.ntotal} vectors, dim={bean_vecs.shape[1]}")

    # --- News chunks ---
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunk_texts = chunks["text"].fillna("").tolist()
    print(f"Encoding {len(chunk_texts)} news chunks …")
    chunk_vecs = model.encode(chunk_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    chunk_vecs = np.ascontiguousarray(chunk_vecs, dtype=np.float32)

    np.save(EMB_DIR / "news_embeddings.npy", chunk_vecs)
    index_news = faiss.IndexFlatIP(chunk_vecs.shape[1])
    index_news.add(chunk_vecs)
    faiss.write_index(index_news, str(EMB_DIR / "news.index"))
    print(f"News index: {index_news.ntotal} vectors, dim={chunk_vecs.shape[1]}")

    print("Done.")


if __name__ == "__main__":
    build_all()
