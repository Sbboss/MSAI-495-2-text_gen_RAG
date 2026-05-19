#!/usr/bin/env python3
"""
Create chunked retrieval corpus from locally saved Hugging Face datasets.

Inputs are expected from:
  data/raw/stackoverflow-posts
  data/raw/wikipedia-20220301.en

Output:
  data/processed/retrieval_corpus.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson
from datasets import load_from_disk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunked retrieval corpus JSONL.")
    parser.add_argument("--so-path", type=Path, default=Path("data/raw/stackoverflow-posts"))
    parser.add_argument(
        "--wiki-path", type=Path, default=Path("data/raw/wikipedia-20220301.en")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/retrieval_corpus.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=220,
        help="Approximate words per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Word overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--max-so-rows",
        type=int,
        default=None,
        help="Optional limit for number of StackOverflow rows.",
    )
    parser.add_argument(
        "--max-wiki-rows",
        type=int,
        default=None,
        help="Optional limit for number of Wikipedia rows.",
    )
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(safe_text(v) for v in value if safe_text(v))
    if isinstance(value, dict):
        # Flatten small dict-like text structures into readable lines.
        lines = []
        for k, v in value.items():
            sv = safe_text(v)
            if sv:
                lines.append(f"{k}: {sv}")
        return "\n".join(lines)
    return str(value).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        segment = words[i : i + chunk_size]
        if not segment:
            continue
        chunks.append(" ".join(segment))
        if i + chunk_size >= len(words):
            break
    return chunks


def get_so_document(row: dict[str, Any]) -> str:
    parts = [
        safe_text(row.get("title")),
        safe_text(row.get("question")),
        safe_text(row.get("body")),
        safe_text(row.get("answer")),
        safe_text(row.get("answers")),
        safe_text(row.get("text")),
    ]
    text = "\n\n".join(p for p in parts if p)
    return text.strip()


def get_wiki_document(row: dict[str, Any]) -> str:
    title = safe_text(row.get("title"))
    text = safe_text(row.get("text"))
    if title and text:
        return f"{title}\n\n{text}".strip()
    return (title or text).strip()


def maybe_slice(ds, max_rows: int | None):
    if max_rows is None:
        return ds
    if max_rows <= 0:
        raise ValueError("Row limits must be positive integers.")
    return ds.select(range(min(len(ds), max_rows)))


def write_chunks(
    rows,
    source_name: str,
    extractor,
    fh,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    total_docs = 0
    total_chunks = 0
    for idx, row in tqdm(
        enumerate(rows), desc=f"Chunking {source_name}", total=len(rows), unit="row"
    ):
        doc = extractor(row)
        if not doc:
            continue
        total_docs += 1
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=chunk_overlap)
        for c_idx, chunk in enumerate(chunks):
            payload = {
                "id": f"{source_name}-{idx}-{c_idx}",
                "source": source_name,
                "row_index": idx,
                "chunk_index": c_idx,
                "text": chunk,
            }
            fh.write(orjson.dumps(payload))
            fh.write(b"\n")
            total_chunks += 1
    return total_docs, total_chunks


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading local datasets...")
    so_ds = load_from_disk(str(args.so_path))
    wiki_ds = load_from_disk(str(args.wiki_path))
    so_ds = maybe_slice(so_ds, args.max_so_rows)
    wiki_ds = maybe_slice(wiki_ds, args.max_wiki_rows)

    print(f"Writing chunked corpus to: {args.output_path}")
    with args.output_path.open("wb") as fh:
        so_docs, so_chunks = write_chunks(
            rows=so_ds,
            source_name="stackoverflow",
            extractor=get_so_document,
            fh=fh,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        wiki_docs, wiki_chunks = write_chunks(
            rows=wiki_ds,
            source_name="wikipedia",
            extractor=get_wiki_document,
            fh=fh,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print("\nDone.")
    print(f"StackOverflow docs: {so_docs:,}, chunks: {so_chunks:,}")
    print(f"Wikipedia docs:    {wiki_docs:,}, chunks: {wiki_chunks:,}")
    print(f"Total chunks:      {so_chunks + wiki_chunks:,}")


if __name__ == "__main__":
    main()
