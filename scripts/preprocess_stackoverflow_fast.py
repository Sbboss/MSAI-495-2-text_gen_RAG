#!/usr/bin/env python3
"""
Fast StackOverflow preprocessing (high RAM, many CPUs).

1) DuckDB: split qa_pairs -> qa_raw_train.parquet / qa_raw_val.parquet
2) Multiprocessing + encode_batch -> tokenized/{train,val}/chunk_*.parquet

Does NOT use HuggingFace `datasets` (avoids duplicate cache + disk blow-up).

Run:
  python scripts/check_preprocess_layout.py --project-root .
  python scripts/preprocess_stackoverflow_fast.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from so_text_utils import format_qa_row, pad_ids, train_tokenizer

VAL_SPLIT_SQL = "((question_id::BIGINT * 2654435761) % 10000)"
ROW_COLS = ["question_id", "title", "question_body", "tags", "answer_body"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast parallel StackOverflow preprocessing.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--memory-limit", default="80GB")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    p.add_argument("--encode-batch-size", type=int, default=4096)
    p.add_argument("--max-text-chars", type=int, default=12_000)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--vocab-size", type=int, default=16_000)
    p.add_argument("--tokenizer-sample-rows", type=int, default=200_000)
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--skip-tokenizer", action="store_true")
    p.add_argument("--skip-tokenize", action="store_true")
    p.add_argument("--cleanup-parts", action="store_true")
    p.add_argument("--sanity-only", action="store_true")
    return p.parse_args()


def sanity_check(processed: Path, qa_pairs: Path) -> None:
    print("=== Sanity check ===")
    print(f"processed-dir: {processed.resolve()}")
    missing = []
    for name, path in [
        ("qa_pairs.parquet", qa_pairs),
    ]:
        if path.exists():
            gb = path.stat().st_size / 1e9
            print(f"  OK  {name} ({gb:.2f} GB)")
        else:
            print(f"  MISSING {name}")
            missing.append(str(path))
    for name in ("qa_raw_train.parquet", "qa_raw_val.parquet"):
        p = processed / name
        if p.exists():
            print(f"  OK  {name} ({p.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  --  {name} (not created yet)")
    tok = processed / "tokenizer" / "tokenizer.json"
    print(f"  {'OK' if tok.exists() else '--'}  tokenizer.json")
    print("=== Required pip packages: duckdb pyarrow tokenizers tqdm ===\n")
    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(missing))


def configure_duckdb(con: duckdb.DuckDBPyConnection, memory_limit: str, threads: int) -> None:
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    if threads > 0:
        con.execute(f"SET threads={threads}")


def duckdb_split(
    qa_pairs: Path, train_raw: Path, val_raw: Path, val_fraction: float, memory_limit: str, threads: int
) -> tuple[int, int]:
    threshold = int(val_fraction * 10_000)
    con = duckdb.connect()
    configure_duckdb(con, memory_limit, threads)
    src = f"read_parquet('{qa_pairs}')"
    cols = "question_id, title, question_body, tags, answer_body, answer_score"

    print("DuckDB: writing train split...")
    con.execute(
        f"COPY (SELECT {cols} FROM {src} WHERE {VAL_SPLIT_SQL} >= {threshold}) "
        f"TO '{train_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print("DuckDB: writing val split...")
    con.execute(
        f"COPY (SELECT {cols} FROM {src} WHERE {VAL_SPLIT_SQL} < {threshold}) "
        f"TO '{val_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_train = con.execute(f"SELECT count(*) FROM read_parquet('{train_raw}')").fetchone()[0]
    n_val = con.execute(f"SELECT count(*) FROM read_parquet('{val_raw}')").fetchone()[0]
    con.close()
    return n_train, n_val


def collect_tokenizer_sample(train_raw: Path, sample_path: Path, max_rows: int, max_chars: int) -> int:
    n = 0
    with sample_path.open("w", encoding="utf-8") as fh:
        for batch in pds.dataset(str(train_raw), format="parquet").scanner(
            columns=["title", "question_body", "tags", "answer_body"], batch_size=8192
        ).to_batches():
            for row in zip(*(batch.column(i).to_pylist() for i in range(4))):
                fh.write(format_qa_row(*row, max_chars=max_chars).replace("\n", " ") + "\n")
                n += 1
                if n >= max_rows:
                    return n
    return n


# --- multiprocessing workers (top-level for pickling) ---
_G_TOK = None
_G_PAD = 0
_G_MAX_LEN = 1024
_G_MAX_TEXT = 12000


def _init_worker(tokenizer_path: str, max_seq_len: int, max_text_chars: int) -> None:
    global _G_TOK, _G_PAD, _G_MAX_LEN, _G_MAX_TEXT
    _G_TOK = Tokenizer.from_file(tokenizer_path)
    _G_PAD = _G_TOK.token_to_id("<pad>") or 0
    _G_MAX_LEN = max_seq_len
    _G_MAX_TEXT = max_text_chars


def _encode_rows(rows: list[tuple]) -> list[list[int]]:
    texts = [format_qa_row(r[1], r[2], r[3], r[4], _G_MAX_TEXT) for r in rows]
    encodings = _G_TOK.encode_batch(texts)
    return [pad_ids(e.ids, _G_MAX_LEN, _G_PAD) for e in encodings]


def write_chunk_file(out_dir: Path, chunk_idx: int, ids_batch: list[list[int]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chunk_{chunk_idx:07d}.parquet"
    pq.write_table(
        pa.table({"input_ids": pa.array(ids_batch, type=pa.list_(pa.int32()))}),
        path,
        compression="zstd",
    )
    return path


def tokenize_parquet_parallel(
    raw_parquet: Path,
    out_dir: Path,
    tokenizer_path: Path,
    max_seq_len: int,
    max_text_chars: int,
    workers: int,
    batch_size: int,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pds.dataset(str(raw_parquet), format="parquet")
    scanner = dataset.scanner(columns=ROW_COLS, batch_size=batch_size)

    total = 0
    chunk_idx = 0
    pending: list[tuple] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(tokenizer_path), max_seq_len, max_text_chars),
    ) as pool:
        futures = []

        def flush_futures(block: bool = False) -> None:
            nonlocal chunk_idx, total, futures
            if not futures:
                return
            if block:
                done = futures
                futures = []
            else:
                done = [f for f in futures if f.done()]
                futures = [f for f in futures if not f.done()]
            for f in done:
                ids_batch = f.result()
                write_chunk_file(out_dir, chunk_idx, ids_batch)
                chunk_idx += 1
                total += len(ids_batch)

        for batch in tqdm(scanner.to_batches(), desc=f"Tokenize {raw_parquet.name}"):
            col = [batch.column(i).to_pylist() for i in range(len(ROW_COLS))]
            rows = list(zip(*col))
            fut = pool.submit(_encode_rows, rows)
            futures.append(fut)
            if len(futures) >= workers * 2:
                flush_futures(block=False)
            del batch, col, rows
            gc.collect()

        while futures:
            flush_futures(block=True)

    return total


def main() -> None:
    args = parse_args()
    processed = args.processed_dir.resolve()
    qa_pairs = processed / "qa_pairs.parquet"
    tokenizer_dir = processed / "tokenizer"
    tokenized_dir = processed / "tokenized"
    train_raw = processed / "qa_raw_train.parquet"
    val_raw = processed / "qa_raw_val.parquet"
    sample_path = processed / "tokenizer_sample.txt"
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    sanity_check(processed, qa_pairs)
    if args.sanity_only:
        return

    if args.cleanup_parts:
        parts = processed / "_join_parts"
        if parts.exists():
            shutil.rmtree(parts)
            print("Removed", parts)

    if not args.skip_split:
        print(f"DuckDB split (memory={args.memory_limit})...")
        n_train, n_val = duckdb_split(
            qa_pairs, train_raw, val_raw, args.val_fraction, args.memory_limit, args.threads
        )
        print(f"Train rows: {n_train:,} | Val rows: {n_val:,}")

    if not args.skip_tokenizer:
        print("Training BPE tokenizer...")
        sample_path.unlink(missing_ok=True)
        n = collect_tokenizer_sample(train_raw, sample_path, args.tokenizer_sample_rows, args.max_text_chars)
        print(f"Sample lines: {n:,}")
        train_tokenizer(sample_path, tokenizer_dir, args.vocab_size)
        sample_path.unlink(missing_ok=True)
    elif not tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path}")

    if args.skip_tokenize:
        return

    print(f"Parallel tokenize: workers={args.workers}, batch_size={args.encode_batch_size}")
    train_out = tokenized_dir / "train"
    val_out = tokenized_dir / "val"
    n_tr = tokenize_parquet_parallel(
        train_raw, train_out, tokenizer_path, args.max_seq_len, args.max_text_chars, args.workers, args.encode_batch_size
    )
    n_va = tokenize_parquet_parallel(
        val_raw, val_out, tokenizer_path, args.max_seq_len, args.max_text_chars, args.workers, args.encode_batch_size
    )

    manifest = {
        "mode": "fast-chunks",
        "train_glob": str(train_out / "chunk_*.parquet"),
        "val_glob": str(val_out / "chunk_*.parquet"),
        "max_seq_len": args.max_seq_len,
        "stats": {"train": n_tr, "val": n_va},
        "workers": args.workers,
    }
    (tokenized_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.", manifest["stats"])


if __name__ == "__main__":
    main()
