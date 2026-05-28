#!/usr/bin/env python3
"""
Fast StackOverflow preprocessing (high RAM, many CPUs).

1) DuckDB: split qa_pairs -> qa_raw_{train,val,test}.parquet
2) Multiprocessing + encode_batch -> tokenized/{train,val,test}/chunk_*.parquet

Does NOT use HuggingFace `datasets` (avoids duplicate cache + disk blow-up).

Run:
  python scripts/check_preprocess_layout.py --project-root .
  python scripts/preprocess_stackoverflow_fast.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
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

SCRIPT_VERSION = "4-fast-tokenize-train-val-test"  # bump when syncing to cluster

VAL_SPLIT_SQL = "((question_id::BIGINT * 2654435761) % 10000)"
ROW_COLS = ["question_id", "title", "question_body", "tags", "answer_body"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast parallel StackOverflow preprocessing.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--memory-limit", default="80GB")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="1 = fastest (encode_batch in main process). 4–8 only if CPU-bound. >16 hurts.",
    )
    p.add_argument("--encode-batch-size", type=int, default=8192)
    p.add_argument(
        "--rows-per-chunk-file",
        type=int,
        default=65_536,
        help="Rows per output parquet (fewer files = faster I/O).",
    )
    p.add_argument("--max-text-chars", type=int, default=12_000)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument(
        "--min-answer-score",
        type=int,
        default=1,
        help="Keep only answers with score >= this value (quality filter).",
    )
    p.add_argument(
        "--min-answer-chars",
        type=int,
        default=40,
        help="Keep only rows whose answer text has at least this many chars.",
    )
    p.add_argument(
        "--min-answer-words",
        type=int,
        default=8,
        help="Keep only rows whose answer text has at least this many word-like tokens.",
    )
    p.add_argument(
        "--drop-short-thanks",
        action="store_true",
        help="Drop trivial boilerplate answers like 'thanks', 'solved', 'works now'.",
    )
    p.add_argument(
        "--best-answer-only",
        action="store_true",
        help="Keep only top answer per question_id (by answer_score, then answer length).",
    )
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
    for name in ("qa_raw_train.parquet", "qa_raw_val.parquet", "qa_raw_test.parquet"):
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
    qa_pairs: Path,
    train_raw: Path,
    val_raw: Path,
    test_raw: Path,
    val_fraction: float,
    test_fraction: float,
    min_answer_score: int,
    min_answer_chars: int,
    min_answer_words: int,
    drop_short_thanks: bool,
    best_answer_only: bool,
    memory_limit: str,
    threads: int,
) -> tuple[int, int, int]:
    if val_fraction <= 0 or test_fraction < 0:
        raise ValueError("val_fraction must be > 0 and test_fraction must be >= 0")
    if (val_fraction + test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    val_threshold = int(val_fraction * 10_000)
    test_threshold = int(test_fraction * 10_000)
    split_threshold = val_threshold + test_threshold
    con = duckdb.connect()
    configure_duckdb(con, memory_limit, threads)
    src = f"read_parquet('{qa_pairs}')"
    cols = "question_id, title, question_body, tags, answer_body, answer_score"
    # Keep punctuation/code text intact. We only filter out low-signal rows.
    word_count_expr = (
        "CASE "
        "WHEN length(trim(regexp_replace(COALESCE(CAST(answer_body AS VARCHAR), ''), '[^A-Za-z0-9_]+', ' ', 'g'))) = 0 THEN 0 "
        "ELSE length(trim(regexp_replace(COALESCE(CAST(answer_body AS VARCHAR), ''), '[^A-Za-z0-9_]+', ' ', 'g'))) "
        "- length(replace(trim(regexp_replace(COALESCE(CAST(answer_body AS VARCHAR), ''), '[^A-Za-z0-9_]+', ' ', 'g')), ' ', '')) + 1 "
        "END"
    )
    quality_where = (
        f"COALESCE(answer_score, 0) >= {int(min_answer_score)} "
        f"AND length(COALESCE(CAST(answer_body AS VARCHAR), '')) >= {int(min_answer_chars)} "
        f"AND ({word_count_expr}) >= {int(min_answer_words)}"
    )
    if drop_short_thanks:
        quality_where += (
            " AND NOT regexp_matches("
            "lower(trim(COALESCE(CAST(answer_body AS VARCHAR), ''))), "
            "'^(thanks!?|thank you!?|solved!?|works!?|works now!?|nvm|fixed)$'"
            ")"
        )
    select_src = src
    if best_answer_only:
        select_src = (
            f"(SELECT {cols} FROM ("
            f"SELECT {cols}, "
            "row_number() OVER (PARTITION BY question_id ORDER BY COALESCE(answer_score, -1000000) DESC, "
            "length(COALESCE(CAST(answer_body AS VARCHAR), '')) DESC) AS _rn "
            f"FROM {src}"
            ") t WHERE _rn = 1)"
        )

    print("DuckDB: writing train split...")
    con.execute(
        f"COPY (SELECT {cols} FROM {select_src} WHERE {quality_where} AND {VAL_SPLIT_SQL} >= {split_threshold}) "
        f"TO '{train_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print("DuckDB: writing val split...")
    con.execute(
        f"COPY (SELECT {cols} FROM {select_src} WHERE {quality_where} AND {VAL_SPLIT_SQL} >= {test_threshold} "
        f"AND {VAL_SPLIT_SQL} < {split_threshold}) "
        f"TO '{val_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print("DuckDB: writing test split...")
    con.execute(
        f"COPY (SELECT {cols} FROM {select_src} WHERE {quality_where} AND {VAL_SPLIT_SQL} < {test_threshold}) "
        f"TO '{test_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_train = con.execute(f"SELECT count(*) FROM read_parquet('{train_raw}')").fetchone()[0]
    n_val = con.execute(f"SELECT count(*) FROM read_parquet('{val_raw}')").fetchone()[0]
    n_test = con.execute(f"SELECT count(*) FROM read_parquet('{test_raw}')").fetchone()[0]
    con.close()
    return n_train, n_val, n_test


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


def _encode_rows_with_tokenizer(
    rows: list[tuple],
    tok: Tokenizer,
    pad_id: int,
    max_seq_len: int,
    max_text_chars: int,
) -> list[list[int]]:
    texts = [format_qa_row(r[1], r[2], r[3], r[4], max_text_chars) for r in rows]
    encodings = tok.encode_batch(texts)
    return [pad_ids(e.ids, max_seq_len, pad_id) for e in encodings]


def tokenize_parquet_fast(
    raw_parquet: Path,
    out_dir: Path,
    tokenizer_path: Path,
    max_seq_len: int,
    max_text_chars: int,
    workers: int,
    batch_size: int,
    rows_per_chunk_file: int,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu = os.cpu_count() or 8
    if workers > 16:
        print(f"  WARNING: capping workers {workers} -> 16 (too many processes slow tokenization)")
        workers = 16

    dataset = pds.dataset(str(raw_parquet), format="parquet")
    scanner = dataset.scanner(columns=ROW_COLS, batch_size=batch_size)
    n_batches = (scanner.count_rows() + batch_size - 1) // batch_size
    print(f"  rows={scanner.count_rows():,} batch_size={batch_size} batches≈{n_batches:,}")

    if workers <= 1:
        return _tokenize_single_process(
            scanner, out_dir, tokenizer_path, max_seq_len, max_text_chars, rows_per_chunk_file, n_batches
        )
    return _tokenize_multiprocess(
        scanner,
        out_dir,
        tokenizer_path,
        max_seq_len,
        max_text_chars,
        workers,
        rows_per_chunk_file,
        n_batches,
    )


def _tokenize_single_process(
    scanner,
    out_dir: Path,
    tokenizer_path: Path,
    max_seq_len: int,
    max_text_chars: int,
    rows_per_chunk_file: int,
    n_batches: int,
) -> int:
    """Usually fastest: tokenizers encode_batch releases the GIL (Rust)."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tok.token_to_id("<pad>") or 0
    total = 0
    chunk_idx = 0
    buffer: list[list[int]] = []

    for batch in tqdm(scanner.to_batches(), total=n_batches, desc="Tokenize"):
        col = [batch.column(i).to_pylist() for i in range(len(ROW_COLS))]
        rows = list(zip(*col))
        buffer.extend(_encode_rows_with_tokenizer(rows, tok, pad_id, max_seq_len, max_text_chars))
        if len(buffer) >= rows_per_chunk_file:
            write_chunk_file(out_dir, chunk_idx, buffer)
            total += len(buffer)
            chunk_idx += 1
            buffer = []

    if buffer:
        write_chunk_file(out_dir, chunk_idx, buffer)
        total += len(buffer)
    return total


def _flush_buffer(buffer: list, out_dir: Path, chunk_idx: int, rows_per_chunk_file: int) -> tuple[list, int, int]:
    total = 0
    while len(buffer) >= rows_per_chunk_file:
        write_chunk_file(out_dir, chunk_idx, buffer[:rows_per_chunk_file])
        total += rows_per_chunk_file
        chunk_idx += 1
        buffer = buffer[rows_per_chunk_file:]
    return buffer, chunk_idx, total


def _tokenize_multiprocess(
    scanner,
    out_dir: Path,
    tokenizer_path: Path,
    max_seq_len: int,
    max_text_chars: int,
    workers: int,
    rows_per_chunk_file: int,
    n_batches: int,
) -> int:
    total = 0
    chunk_idx = 0
    buffer: list[list[int]] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(tokenizer_path), max_seq_len, max_text_chars),
    ) as pool:
        futures = []
        for batch in tqdm(scanner.to_batches(), total=n_batches, desc="Tokenize (pool)"):
            col = [batch.column(i).to_pylist() for i in range(len(ROW_COLS))]
            futures.append(pool.submit(_encode_rows, list(zip(*col))))

            still_pending = []
            for f in futures:
                if f.done():
                    buffer.extend(f.result())
                else:
                    still_pending.append(f)
            futures = still_pending
            buffer, chunk_idx, added = _flush_buffer(buffer, out_dir, chunk_idx, rows_per_chunk_file)
            total += added

        for f in as_completed(futures):
            buffer.extend(f.result())
        buffer, chunk_idx, added = _flush_buffer(buffer, out_dir, chunk_idx, rows_per_chunk_file)
        total += added
        if buffer:
            write_chunk_file(out_dir, chunk_idx, buffer)
            total += len(buffer)
    return total


def main() -> None:
    args = parse_args()
    processed = args.processed_dir.resolve()
    qa_pairs = processed / "qa_pairs.parquet"
    tokenizer_dir = processed / "tokenizer"
    tokenized_dir = processed / "tokenized"
    train_raw = processed / "qa_raw_train.parquet"
    val_raw = processed / "qa_raw_val.parquet"
    test_raw = processed / "qa_raw_test.parquet"
    sample_path = processed / "tokenizer_sample.txt"
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    print(f"Script version: {SCRIPT_VERSION} (expect: no 'datasets', no 'Generating train split')")
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
        print(
            f"Quality filters: min_answer_score={args.min_answer_score}, "
            f"min_answer_chars={args.min_answer_chars}, "
            f"min_answer_words={args.min_answer_words}, "
            f"drop_short_thanks={args.drop_short_thanks}, "
            f"best_answer_only={args.best_answer_only}"
        )
        n_train, n_val, n_test = duckdb_split(
            qa_pairs,
            train_raw,
            val_raw,
            test_raw,
            args.val_fraction,
            args.test_fraction,
            args.min_answer_score,
            args.min_answer_chars,
            args.min_answer_words,
            args.drop_short_thanks,
            args.best_answer_only,
            args.memory_limit,
            args.threads,
        )
        print(f"Train rows: {n_train:,} | Val rows: {n_val:,} | Test rows: {n_test:,}")

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

    print(
        f"Tokenize: workers={args.workers} (1=fastest), "
        f"batch_size={args.encode_batch_size}, rows/file={args.rows_per_chunk_file}"
    )
    train_out = tokenized_dir / "train"
    val_out = tokenized_dir / "val"
    test_out = tokenized_dir / "test"
    n_tr = tokenize_parquet_fast(
        train_raw,
        train_out,
        tokenizer_path,
        args.max_seq_len,
        args.max_text_chars,
        args.workers,
        args.encode_batch_size,
        args.rows_per_chunk_file,
    )
    n_va = tokenize_parquet_fast(
        val_raw,
        val_out,
        tokenizer_path,
        args.max_seq_len,
        args.max_text_chars,
        args.workers,
        args.encode_batch_size,
        args.rows_per_chunk_file,
    )
    n_te = tokenize_parquet_fast(
        test_raw,
        test_out,
        tokenizer_path,
        args.max_seq_len,
        args.max_text_chars,
        args.workers,
        args.encode_batch_size,
        args.rows_per_chunk_file,
    )

    manifest = {
        "mode": "fast-chunks",
        "train_glob": str(train_out / "chunk_*.parquet"),
        "val_glob": str(val_out / "chunk_*.parquet"),
        "test_glob": str(test_out / "chunk_*.parquet"),
        "max_seq_len": args.max_seq_len,
        "stats": {"train": n_tr, "val": n_va, "test": n_te},
        "fractions": {"val": args.val_fraction, "test": args.test_fraction},
        "filters": {
            "min_answer_score": args.min_answer_score,
            "min_answer_chars": args.min_answer_chars,
            "min_answer_words": args.min_answer_words,
            "drop_short_thanks": args.drop_short_thanks,
            "best_answer_only": args.best_answer_only,
        },
        "workers": args.workers,
    }
    (tokenized_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.", manifest["stats"])


if __name__ == "__main__":
    main()
