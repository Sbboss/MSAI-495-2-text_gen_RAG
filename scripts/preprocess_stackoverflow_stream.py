#!/usr/bin/env python3
"""
Memory-safe StackOverflow preprocessing.

Reads qa_pairs.parquet in tiny Arrow scanner batches (not whole row groups),
formats + tokenizes one row at a time, writes many small chunk files.

Run from project root (NOT inside Jupyter — saves ~2–4GB RAM):
  python scripts/preprocess_stackoverflow_stream.py
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from so_text_utils import (  # noqa: E402
    SPECIAL_TOKENS,
    format_qa_row,
    is_val,
    pad_ids,
    to_text,
    train_tokenizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-preprocess StackOverflow Q&A pairs.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--row-batch-size", type=int, default=64, help="Arrow scanner batch (keep <=128).")
    p.add_argument("--max-text-chars", type=int, default=12_000, help="Truncate text before tokenize.")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--vocab-size", type=int, default=16_000)
    p.add_argument("--tokenizer-sample-rows", type=int, default=50_000)
    p.add_argument("--skip-tokenizer", action="store_true", help="Reuse existing tokenizer.json.")
    p.add_argument("--skip-tokenize", action="store_true", help="Only train tokenizer.")
    p.add_argument("--cleanup-parts", action="store_true", help="Delete _join_parts if present.")
    return p.parse_args()


def write_chunk(out_dir: Path, split: str, chunk_idx: int, ids_batch: list[list[int]]) -> None:
    if not ids_batch:
        return
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"chunk_{chunk_idx:07d}.parquet"
    table = pa.table({"input_ids": pa.array(ids_batch, type=pa.list_(pa.int32()))})
    pq.write_table(table, path, compression="zstd")


def collect_tokenizer_sample(
    dataset: ds.Dataset,
    sample_path: Path,
    max_rows: int,
    max_chars: int,
    val_pct: float,
    batch_size: int,
) -> int:
    cols = ["question_id", "title", "question_body", "tags", "answer_body"]
    scanner = dataset.scanner(columns=cols, batch_size=batch_size)
    n = 0
    with sample_path.open("w", encoding="utf-8") as fh:
        for batch in scanner.to_batches():
            col = {name: batch.column(i).to_pylist() for i, name in enumerate(cols)}
            for qid, title, qbody, tags, abody in zip(*col.values()):
                if is_val(qid, val_pct):
                    continue
                line = format_qa_row(title, qbody, tags, abody, max_chars)
                fh.write(line.replace("\n", " ") + "\n")
                n += 1
                if n >= max_rows:
                    return n
            gc.collect()
    return n


def stream_tokenize(
    dataset: ds.Dataset,
    out_dir: Path,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_chars: int,
    val_pct: float,
    batch_size: int,
) -> dict:
    pad_id = tokenizer.token_to_id("<pad>") or 0
    cols = ["question_id", "title", "question_body", "tags", "answer_body"]
    scanner = dataset.scanner(columns=cols, batch_size=batch_size)

    train_ids: list[list[int]] = []
    val_ids: list[list[int]] = []
    train_chunk = val_chunk = 0
    stats = {"train": 0, "val": 0, "skipped": 0}

    def flush(split: str, buf: list, idx: int) -> tuple[list, int]:
        write_chunk(out_dir, split, idx, buf)
        return [], idx + 1

    for batch in tqdm(scanner.to_batches(), desc="Tokenize (single pass)"):
        col = {name: batch.column(i).to_pylist() for i, name in enumerate(cols)}
        for qid, title, qbody, tags, abody in zip(*col.values()):
            try:
                text = format_qa_row(title, qbody, tags, abody, max_chars)
                ids = tokenizer.encode(text).ids[:max_seq_len]
                if len(ids) < max_seq_len:
                    ids.extend([pad_id] * (max_seq_len - len(ids)))
                if is_val(qid, val_pct):
                    val_ids.append(ids)
                    stats["val"] += 1
                    if len(val_ids) >= batch_size:
                        val_ids, val_chunk = flush("val", val_ids, val_chunk)
                else:
                    train_ids.append(ids)
                    stats["train"] += 1
                    if len(train_ids) >= batch_size:
                        train_ids, train_chunk = flush("train", train_ids, train_chunk)
            except Exception:
                stats["skipped"] += 1
        del batch, col
        gc.collect()

    if train_ids:
        flush("train", train_ids, train_chunk)
    if val_ids:
        flush("val", val_ids, val_chunk)

    return stats


def write_manifest(out_dir: Path, stats: dict, args: argparse.Namespace) -> None:
    manifest = {
        "train_glob": str(out_dir / "train" / "chunk_*.parquet"),
        "val_glob": str(out_dir / "val" / "chunk_*.parquet"),
        "max_seq_len": args.max_seq_len,
        "stats": stats,
        "row_batch_size": args.row_batch_size,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print("Wrote manifest:", path)


def main() -> None:
    args = parse_args()
    processed = args.processed_dir
    qa_pairs = processed / "qa_pairs.parquet"
    tokenizer_dir = processed / "tokenizer"
    tokenized_dir = processed / "tokenized"
    sample_path = processed / "tokenizer_sample.txt"

    if not qa_pairs.exists():
        raise FileNotFoundError(f"Missing {qa_pairs}. Run join step first.")

    if args.cleanup_parts:
        parts = processed / "_join_parts"
        if parts.exists():
            shutil.rmtree(parts)
            print("Removed", parts)

    # Fresh chunk dirs
    for sub in ("train", "val"):
        d = tokenized_dir / sub
        if d.exists():
            shutil.rmtree(d)

    print("Opening dataset with scanner batch_size =", args.row_batch_size)
    dataset = ds.dataset(str(qa_pairs), format="parquet")

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not args.skip_tokenizer:
        print("Collecting tokenizer sample...")
        if sample_path.exists():
            sample_path.unlink()
        n = collect_tokenizer_sample(
            dataset, sample_path, args.tokenizer_sample_rows, args.max_text_chars, args.val_fraction, args.row_batch_size
        )
        print(f"Sample lines: {n:,}")
        print("Training BPE tokenizer...")
        train_tokenizer(sample_path, tokenizer_dir, args.vocab_size)
        if sample_path.exists():
            sample_path.unlink()
    elif not tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path}")

    if args.skip_tokenize:
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stats = stream_tokenize(
        dataset, tokenized_dir, tokenizer, args.max_seq_len, args.max_text_chars, args.val_fraction, args.row_batch_size
    )
    write_manifest(tokenized_dir, stats, args)
    print("Done.", stats)
    print("Train chunks:", len(list((tokenized_dir / "train").glob("chunk_*.parquet"))))
    print("Val chunks:", len(list((tokenized_dir / "val").glob("chunk_*.parquet"))))


if __name__ == "__main__":
    main()
