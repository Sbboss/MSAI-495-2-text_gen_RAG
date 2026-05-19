#!/usr/bin/env python3
"""
Download and store QueryGPT datasets locally.

Datasets:
1) mikex86/stackoverflow-posts
2) wikipedia (20220301.en)

Examples:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --output-dir data/raw --max-so-rows 50000 --max-wiki-rows 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download StackOverflow and Wikipedia datasets and save them to disk."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root folder to store datasets (default: data/raw).",
    )
    parser.add_argument(
        "--so-split",
        type=str,
        default="train",
        help="Split to load for mikex86/stackoverflow-posts (default: train).",
    )
    parser.add_argument(
        "--wiki-config",
        type=str,
        default="20220301.en",
        help="Wikipedia dataset config name (default: 20220301.en).",
    )
    parser.add_argument(
        "--wiki-split",
        type=str,
        default="train",
        help="Split to load for wikipedia dataset (default: train).",
    )
    parser.add_argument(
        "--max-so-rows",
        type=int,
        default=None,
        help="Optional row cap for StackOverflow (useful for quick tests).",
    )
    parser.add_argument(
        "--max-wiki-rows",
        type=int,
        default=None,
        help="Optional row cap for Wikipedia (useful for quick tests).",
    )
    return parser.parse_args()


def maybe_select_rows(ds: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None:
        return ds
    if max_rows <= 0:
        raise ValueError("Row limits must be positive integers.")
    return ds.select(range(min(len(ds), max_rows)))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    so_dir = output_dir / "stackoverflow-posts"
    wiki_dir = output_dir / f"wikipedia-{args.wiki_config}"

    print("Loading StackOverflow dataset...")
    so_ds = load_dataset("mikex86/stackoverflow-posts", split=args.so_split)
    so_ds = maybe_select_rows(so_ds, args.max_so_rows)
    print(f"Saving StackOverflow dataset to: {so_dir}")
    so_ds.save_to_disk(str(so_dir))
    print(f"Saved StackOverflow rows: {len(so_ds):,}")

    print("\nLoading Wikipedia dataset...")
    wiki_ds = load_dataset("wikipedia", args.wiki_config, split=args.wiki_split)
    wiki_ds = maybe_select_rows(wiki_ds, args.max_wiki_rows)
    print(f"Saving Wikipedia dataset to: {wiki_dir}")
    wiki_ds.save_to_disk(str(wiki_dir))
    print(f"Saved Wikipedia rows: {len(wiki_ds):,}")

    print("\nDone.")
    print(f"Datasets stored under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

