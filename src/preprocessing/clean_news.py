"""Clean raw coffee_news.json → data/processed/news_clean.parquet + news_chunks.parquet"""

import json
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "coffee_news.json"
OUT_CLEAN = ROOT / "data" / "processed" / "news_clean.parquet"
OUT_CHUNKS = ROOT / "data" / "processed" / "news_chunks.parquet"

NAV_KEYWORDS = [
    "Skip to content", "Login", "Subscribe", "[has_child]",
    "[subitem]", "Share on", "Privacy Policy", "@",
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "],
)


def _clean_tags(tags: list[str]) -> list[str]:
    return [t for t in tags if not any(kw in t for kw in NAV_KEYWORDS)]


def clean_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    df = df[df["status"] == "ok"].copy()
    df["tags_clean"] = df["tags"].apply(
        lambda x: _clean_tags(x) if isinstance(x, list) else []
    )

    keep_cols = [
        "title", "article_url", "source", "author", "section",
        "summary", "content_text", "tags_clean",
        "publish_datetime", "language",
    ]
    df = df[keep_cols].copy()
    df["publish_datetime"] = pd.to_datetime(df["publish_datetime"], errors="coerce")

    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_CLEAN, index=False)
    print(f"Saved {len(df)} articles -> {OUT_CLEAN}")

    chunks = []
    for idx, row in df.iterrows():
        text = row.get("content_text") or ""
        if not text.strip():
            continue
        parts = splitter.split_text(text)
        for ci, chunk_text in enumerate(parts):
            chunks.append({
                "article_idx": idx,
                "chunk_index": ci,
                "text": chunk_text,
                "title": row["title"],
                "source": row["source"],
                "article_url": row["article_url"],
                "publish_datetime": row["publish_datetime"],
            })

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(OUT_CHUNKS, index=False)
    print(f"Saved {len(chunks_df)} chunks -> {OUT_CHUNKS}")

    return df, chunks_df


if __name__ == "__main__":
    clean_news()
