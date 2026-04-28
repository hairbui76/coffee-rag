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

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


def _build_and_save(model, texts, npy_path, index_path, label):
    print(f"Encoding {len(texts)} {label} …")
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)

    np.save(npy_path, vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_path))
    print(f"{label} index: {index.ntotal} vectors, dim={vecs.shape[1]}")


def build_all(model_name: str = MODEL_NAME, force: bool = False):
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    beans_npy = EMB_DIR / "beans_embeddings.npy"
    beans_idx = EMB_DIR / "beans.index"
    news_npy = EMB_DIR / "news_embeddings.npy"
    news_idx = EMB_DIR / "news.index"

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    expected_dim = model.get_embedding_dimension()

    def _needs_rebuild(npy_path, idx_path):
        if force:
            return True
        if not npy_path.exists() or not idx_path.exists():
            return True
        existing = np.load(npy_path)
        if existing.shape[1] != expected_dim:
            print(f"  Dimension mismatch ({existing.shape[1]} != {expected_dim}), rebuilding.")
            return True
        return False

    if _needs_rebuild(beans_npy, beans_idx):
        beans = pd.read_parquet(BEANS_PATH)
        texts = beans["document_text"].fillna("").tolist()
        _build_and_save(model, texts, beans_npy, beans_idx, "bean documents")
    else:
        print(f"Beans index already exists (dim={expected_dim}), skipping. Use --force to rebuild.")

    if _needs_rebuild(news_npy, news_idx):
        chunks = pd.read_parquet(CHUNKS_PATH)
        chunk_texts = chunks["text"].fillna("").tolist()
        _build_and_save(model, chunk_texts, news_npy, news_idx, "news chunks")
    else:
        print(f"News index already exists (dim={expected_dim}), skipping. Use --force to rebuild.")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild all indices even if they exist.")
    args = parser.parse_args()
    build_all(force=args.force)
