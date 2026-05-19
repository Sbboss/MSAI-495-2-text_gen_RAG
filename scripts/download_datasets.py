#!/usr/bin/env python3
"""
Download QueryGPT datasets with minimal disk use.

StackOverflow: downloads only the first N Parquet shards directly into
data/raw/stackoverflow-posts/ (one copy on disk — no full-dataset reload +
save_to_disk duplicate).

Wikipedia: downloads Parquet shards from legacy-datasets/wikipedia (20220301.en)
and saves a capped subset as a single Parquet file.

Examples:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --max-so-shards 20
  python scripts/download_datasets.py --with-wiki   # phase 2 (RAG index)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

SO_REPO = "mikex86/stackoverflow-posts"
SO_TOTAL_SHARDS = 59
SO_SHARD_PATTERN = "stackoverflow-posts-{idx:05d}-of-00058.parquet"

WIKI_REPO = "legacy-datasets/wikipedia"
WIKI_CONFIG = "20220301.en"
WIKI_TOTAL_SHARDS = 41
WIKI_SHARD_PATTERN = "data/20220301.en/train-{idx:05d}-of-00041.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download StackOverflow/Wikipedia subsets for QueryGPT."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root folder for stored datasets (default: data/raw).",
    )
    parser.add_argument(
        "--max-so-shards",
        type=int,
        default=20,
        help=f"Number of StackOverflow Parquet shards to download (max {SO_TOTAL_SHARDS}).",
    )
    parser.add_argument(
        "--wiki-config",
        type=str,
        default="20220301.en",
        help="Wikipedia dataset config (default: 20220301.en).",
    )
    parser.add_argument(
        "--max-wiki-rows",
        type=int,
        default=200_000,
        help="Max Wikipedia articles to store (default: 200000).",
    )
    parser.add_argument(
        "--with-wiki",
        action="store_true",
        help="Also download Wikipedia (phase 2 — RAG index; skipped by default).",
    )
    parser.add_argument(
        "--skip-so",
        action="store_true",
        help="Skip StackOverflow download (e.g. retry Wikipedia only).",
    )
    parser.add_argument(
        "--clear-hf-cache",
        action="store_true",
        default=True,
        help="Remove Hugging Face hub/cache copies for these datasets after download.",
    )
    parser.add_argument(
        "--no-clear-hf-cache",
        action="store_false",
        dest="clear_hf_cache",
        help="Keep Hugging Face cache after download.",
    )
    parser.add_argument(
        "--clear-cache-only",
        action="store_true",
        help="Only remove Hugging Face cache for these datasets, then exit.",
    )
    return parser.parse_args()


def download_stackoverflow_shards(output_dir: Path, num_shards: int) -> Path:
    if num_shards <= 0 or num_shards > SO_TOTAL_SHARDS:
        raise ValueError(f"--max-so-shards must be between 1 and {SO_TOTAL_SHARDS}.")

    so_dir = output_dir / "stackoverflow-posts"
    so_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_shards} StackOverflow shard(s) to: {so_dir}")
    for i in range(num_shards):
        shard_name = SO_SHARD_PATTERN.format(idx=i)
        print(f"  [{i + 1}/{num_shards}] {shard_name}")
        hf_hub_download(
            repo_id=SO_REPO,
            filename=shard_name,
            repo_type="dataset",
            local_dir=so_dir,
        )

    parquet_files = sorted(so_dir.glob("*.parquet"))
    print(f"Done. {len(parquet_files)} Parquet file(s) in {so_dir.resolve()}")
    return so_dir


def download_wikipedia_subset(output_dir: Path, wiki_config: str, max_rows: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if wiki_config != WIKI_CONFIG:
        raise ValueError(
            f"Only {WIKI_CONFIG} is supported via parquet shards. Got: {wiki_config}"
        )
    if max_rows <= 0:
        raise ValueError("--max-wiki-rows must be a positive integer.")

    wiki_dir = output_dir / f"wikipedia-{wiki_config}"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    subset_path = wiki_dir / "subset.parquet"
    if subset_path.exists():
        table = pq.read_table(subset_path)
        print(f"Wikipedia subset already exists ({table.num_rows:,} rows): {subset_path}")
        return wiki_dir

    tmp_dir = wiki_dir / "_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Wikipedia ({wiki_config}) parquet shards, target {max_rows:,} rows...")
    collected: list[dict] = []
    shard_idx = 0

    while len(collected) < max_rows and shard_idx < WIKI_TOTAL_SHARDS:
        shard_name = WIKI_SHARD_PATTERN.format(idx=shard_idx)
        print(f"  [{shard_idx + 1}] {shard_name}")
        shard_path = hf_hub_download(
            repo_id=WIKI_REPO,
            filename=shard_name,
            repo_type="dataset",
            local_dir=tmp_dir,
        )
        table = pq.read_table(shard_path)
        for batch in table.to_batches():
            batch_dict = batch.to_pydict()
            n = len(next(iter(batch_dict.values())))
            for i in range(n):
                collected.append({k: batch_dict[k][i] for k in batch_dict})
                if len(collected) >= max_rows:
                    break
            if len(collected) >= max_rows:
                break
        shard_idx += 1

    subset_table = pa.Table.from_pylist(collected[:max_rows])
    pq.write_table(subset_table, subset_path)
    shutil.rmtree(tmp_dir)

    print(f"Saved Wikipedia rows: {subset_table.num_rows:,} -> {subset_path.resolve()}")
    return wiki_dir


def clear_huggingface_cache() -> None:
    home_cache = Path.home() / ".cache" / "huggingface"
    targets = [
        home_cache / "hub" / "datasets--mikex86--stackoverflow-posts",
        home_cache / "hub" / ".locks" / "datasets--mikex86--stackoverflow-posts",
        home_cache / "datasets" / "mikex86___stackoverflow-posts",
        home_cache / "hub" / "datasets--wikipedia",
        home_cache / "hub" / "datasets--legacy-datasets--wikipedia",
        home_cache / "hub" / "datasets--wikimedia--wikipedia",
        home_cache / "datasets" / "wikipedia",
        home_cache / "datasets" / "legacy-datasets___wikipedia",
        home_cache / "datasets" / "wikimedia___wikipedia",
    ]

    for path in (home_cache / "datasets").glob("*wikipedia*"):
        targets.append(path)

    removed = 0
    for path in targets:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed cache: {path}")
            removed += 1

    if removed == 0:
        print("No matching Hugging Face cache entries found to remove.")


def main() -> None:
    args = parse_args()

    if args.clear_cache_only:
        clear_huggingface_cache()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_so:
        download_stackoverflow_shards(args.output_dir, args.max_so_shards)
    else:
        print("Skipping StackOverflow download.")

    if args.with_wiki:
        download_wikipedia_subset(args.output_dir, args.wiki_config, args.max_wiki_rows)
    else:
        print("Skipping Wikipedia (phase 1: StackOverflow training only).")

    if args.clear_hf_cache:
        print("\nClearing Hugging Face cache for downloaded datasets...")
        clear_huggingface_cache()

    print("\nAll done.")
    print(f"Data location: {args.output_dir.resolve()}")
    print("StackOverflow is stored as Parquet shards only (no duplicate Arrow export).")


if __name__ == "__main__":
    main()
