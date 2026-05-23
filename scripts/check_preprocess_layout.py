#!/usr/bin/env python3
"""Print expected data layout, file sizes, and disk/quota usage."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def print_path(label: str, path: Path) -> None:
    if path.is_file():
        print(f"  {'OK' if path.exists() else 'MISSING':7} {label}: {path} ({human(path.stat().st_size)})")
    elif path.is_dir() and path.exists():
        print(f"  OK      {label}: {path}/ ({human(dir_size(path))})")
    else:
        print(f"  MISSING {label}: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--processed-dir", type=Path, default=None)
    args = p.parse_args()

    root = args.project_root
    if (root / "notebooks").exists() and not (root / "scripts").exists():
        root = root
    processed = args.processed_dir or root / "data" / "processed"
    raw_so = root / "data" / "raw" / "stackoverflow-posts"

    print("=== Disk / quota ===\n")
    for cmd in ("df -h .", "df -h " + str(processed), "quota -s 2>/dev/null || true"):
        print(f"$ {cmd}")
        subprocess.run(cmd, shell=True)

    print("\n=== HuggingFace cache (often fills quota) ===\n")
    hf = Path.home() / ".cache" / "huggingface"
    if hf.exists():
        print(f"  {hf}: {human(dir_size(hf))}")
    else:
        print("  (no ~/.cache/huggingface)")

    print("\n=== Expected layout ===\n")
    print(f"Project root: {root.resolve()}\n")
    print_path("Raw SO shards", raw_so)
    print_path("qa_pairs.parquet", processed / "qa_pairs.parquet")
    print_path("questions_index (optional)", processed / "questions_index.parquet")
    print_path("_join_parts (delete if done)", processed / "_join_parts")
    print_path("qa_raw_train.parquet", processed / "qa_raw_train.parquet")
    print_path("qa_raw_val.parquet", processed / "qa_raw_val.parquet")
    print_path("tokenizer/", processed / "tokenizer")
    print_path("tokenized/train/", processed / "tokenized" / "train")
    print_path("tokenized/val/", processed / "tokenized" / "val")
    print_path("manifest.json", processed / "tokenized" / "manifest.json")

    if processed.exists():
        print(f"\n  TOTAL processed/: {human(dir_size(processed))}")

    print("\n=== Python imports (preprocess needs) ===\n")
    for mod in ("duckdb", "pyarrow", "tokenizers", "tqdm"):
        try:
            __import__(mod)
            print(f"  OK  {mod}")
        except ImportError:
            print(f"  MISSING {mod}  -> pip install {mod}")


if __name__ == "__main__":
    main()
