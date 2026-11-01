"""Collect the files the Kaggle notebook needs into one folder to upload as a Kaggle Dataset.

Run on your machine:
    python -m evaluation.kaggle.pack_data            # -> kaggle_upload/ (~240 MB)
    python -m evaluation.kaggle.pack_data --zip      # -> kaggle_upload.zip as well

Upload the folder (or the zip) at https://www.kaggle.com/datasets  →  New Dataset,
title "coffee-rag-data". The notebook mounts it at /kaggle/input/coffee-rag-data/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "ragas_eval_dataset_v6.json",
    "data/processed/beans_clean.parquet",
    "data/processed/news_clean.parquet",
    "data/processed/news_chunks.parquet",
    "data/embeddings/beans.index",
    "data/embeddings/beans_embeddings.npy",
    "data/embeddings/news.index",
    "data/embeddings/news_embeddings.npy",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "kaggle_upload")
    p.add_argument("--zip", action="store_true")
    args = p.parse_args()

    total = 0
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing: {src}")
        if src.stat().st_size < 1000 and src.suffix in (".index", ".npy", ".parquet"):
            raise SystemExit(f"{src} looks like a git-lfs pointer, run `git lfs pull` first")
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += src.stat().st_size
        print(f"  {rel}  ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"\n{len(FILES)} files, {total / 1e6:.0f} MB -> {args.out}")
    if args.zip:
        z = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
        print(f"zip -> {z}")


if __name__ == "__main__":
    main()
