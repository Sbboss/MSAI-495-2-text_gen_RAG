#!/usr/bin/env python3
"""
Build FAISS index from retrieval corpus JSONL.

Input:
  data/processed/retrieval_corpus.jsonl

Outputs (default):
  artifacts/retrieval/index.faiss
  artifacts/retrieval/meta.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import orjson
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunked corpus.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/retrieval_corpus.jsonl"),
        help="Input chunked corpus JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/retrieval"),
        help="Directory for FAISS index and metadata output.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence encoder model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoder batch size.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for corpus rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int | None = None) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    metas: list[dict] = []
    with path.open("rb") as fh:
        for idx, line in enumerate(fh):
            if max_rows is not None and idx >= max_rows:
                break
            item = orjson.loads(line)
            text = item.get("text", "")
            if not text:
                continue
            texts.append(text)
            metas.append(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "row_index": item.get("row_index"),
                    "chunk_index": item.get("chunk_index"),
                    "text": text,
                }
            )
    return texts, metas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.output_dir / "index.faiss"
    meta_path = args.output_dir / "meta.jsonl"

    print(f"Reading corpus: {args.input_path}")
    texts, metas = read_jsonl(args.input_path, max_rows=args.max_rows)
    if not texts:
        raise ValueError("No rows found in input corpus. Check your input path.")

    print(f"Loaded rows: {len(texts):,}")
    print(f"Loading sentence encoder: {args.model_name}")
    encoder = SentenceTransformer(args.model_name)

    print("Encoding chunks...")
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print("Building FAISS index (IndexFlatIP)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Saving metadata to: {meta_path}")
    with meta_path.open("wb") as fh:
        for meta in tqdm(metas, desc="Writing metadata", unit="row"):
            fh.write(orjson.dumps(meta))
            fh.write(b"\n")

    print("\nDone.")
    print(f"Indexed chunks: {index.ntotal:,}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
