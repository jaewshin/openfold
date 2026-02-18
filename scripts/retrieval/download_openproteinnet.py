#!/usr/bin/env python3
"""
Download OpenProteinSet v1 from AWS S3 (Registry of Open Data on AWS).
Source: s3://openfold/  (public, no credentials required)

What this downloads by default:
  - pdb subset:                  structure files only (if present under prefix)
  - uniclust30_filtered subset:  predicted structures (.pdb)

Always skipped:
  - MSA files (.a3m / .a3m.gz)
  - Template-hit files (.hhr / .hhr.gz)

This script does NOT compute Neff/MSA depth.

Requirements:
  pip install boto3 botocore tqdm

Usage examples:
  # Download structures only:
  python scripts/retrieval/download_openproteinnet.py --output-dir ./openproteinset

  # Download only PDB subset:
  python scripts/retrieval/download_openproteinnet.py --output-dir ./openproteinset --subsets pdb

  # Download a small test subset (first N entries only):
  python scripts/retrieval/download_openproteinnet.py --output-dir ./openproteinset --limit 100

  # Parallel downloads:
  python scripts/retrieval/download_openproteinnet.py --output-dir ./openproteinset --workers 16
"""

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("Missing dependency: pip install boto3 botocore")

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback progress bar.
    class tqdm:  # type: ignore
        def __init__(self, *a, **kw):
            self.total = kw.get("total", 0)
            self._n = 0

        def update(self, n=1):
            self._n += n
            print(f"\r{self._n}/{self.total}", end="", flush=True)

        def close(self):
            print()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

        def set_postfix_str(self, s):
            pass


BUCKET = "openfold"

SUBSET_PREFIXES = {
    "pdb": "pdb/",
    "uniclust30_filtered": "uniclust30/",
    "data_caches": "data_caches/",
    "pdb_mmcif": "pdb_mmcif/",
}

# Always skip these content types.
SKIP_SUFFIXES = (".a3m", ".a3m.gz", ".hhr", ".hhr.gz")

# Keep only these file types by subset.
STRUCTURE_SUFFIXES = (".pdb", ".pdb.gz", ".cif", ".cif.gz")
CACHE_SUFFIXES = (".json", ".json.gz", ".txt", ".txt.gz")


def make_s3_client():
    """Create an anonymous S3 client (no AWS credentials needed)."""
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, max_pool_connections=50),
    )


def should_download(key: str, subset: str) -> bool:
    """Decide whether to download a given S3 key."""
    if key.endswith("/"):
        return False

    key_l = key.lower()
    if key_l.endswith(SKIP_SUFFIXES):
        return False

    if subset == "data_caches":
        return key_l.endswith(CACHE_SUFFIXES)

    return key_l.endswith(STRUCTURE_SUFFIXES)


def collect_download_candidates(
    client,
    subset: str,
    s3_prefix: str,
    limit: int | None,
) -> Tuple[List[Tuple[str, int]], int]:
    """Collect filtered object keys under prefix.

    Returns:
      to_download: list of (key, size)
      scanned: total number of S3 objects scanned under the prefix
    """
    paginator = client.get_paginator("list_objects_v2")
    to_download: List[Tuple[str, int]] = []
    scanned = 0
    entries_seen = set()
    stop_early = False

    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            scanned += 1
            key = obj["Key"]
            size = int(obj["Size"])

            if limit is not None:
                rel = key[len(s3_prefix):]
                entry = rel.split("/", 1)[0] if rel else key
                if entry not in entries_seen:
                    if len(entries_seen) >= limit:
                        stop_early = True
                        break
                    entries_seen.add(entry)

            if not should_download(key, subset=subset):
                continue

            to_download.append((key, size))
        if stop_early:
            break

    return to_download, scanned


def download_object(client, key: str, dest_path: Path, progress_lock, progress_bar):
    """Download a single S3 object to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(BUCKET, key, str(dest_path))
    except ClientError as e:
        print(f"\n[WARN] Failed to download {key}: {e}", file=sys.stderr)
        return False

    with progress_lock:
        progress_bar.update(1)
    return True


def download_subset(
    client,
    subset_name: str,
    s3_prefix: str,
    output_dir: Path,
    limit: int | None,
    workers: int,
):
    print(f"\n[INFO] Listing objects under s3://{BUCKET}/{s3_prefix} ...")
    to_download, scanned = collect_download_candidates(
        client=client,
        subset=subset_name,
        s3_prefix=s3_prefix,
        limit=limit,
    )

    print(f"[INFO] Scanned {scanned:,} objects")

    if limit is not None:
        print(f"[INFO] Limited to first {limit} entries -> {len(to_download):,} files")

    total_bytes = sum(sz for _, sz in to_download)
    print(f"[INFO] Downloading {len(to_download):,} files ({total_bytes/1e9:.2f} GB)")

    if not to_download:
        print(f"[WARN] No files selected for subset '{subset_name}' under prefix '{s3_prefix}'")
        return

    progress_lock = threading.Lock()
    pbar = tqdm(total=len(to_download), unit="file", desc=s3_prefix.strip("/"))

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_object, client, key, output_dir / key, progress_lock, pbar): key
            for key, _ in to_download
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                ok = fut.result()
                if not ok:
                    failed.append(key)
            except Exception as e:
                failed.append(key)
                print(f"\n[ERROR] {key}: {e}", file=sys.stderr)

    pbar.close()

    if failed:
        print(f"[WARN] {len(failed)} files failed to download.")
        fail_log = output_dir / f"failed_{s3_prefix.strip('/').replace('/', '_')}.txt"
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"[INFO] Failed file list written to {fail_log}")

    print(f"[INFO] Done with {s3_prefix.strip('/')}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download OpenProteinSet v1 from AWS S3 (s3://openfold/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Local directory to save downloaded files",
    )
    p.add_argument(
        "--subsets", nargs="+",
        choices=["pdb", "uniclust30_filtered", "data_caches", "pdb_mmcif"],
        default=["pdb", "uniclust30_filtered"],
        help=(
            "Which subsets to download. Choices: "
            "pdb (PDB-chain prefix, structure files only), "
            "uniclust30_filtered (filtered uniclust30 prefix), "
            "data_caches (precomputed metadata), "
            "pdb_mmcif (raw mmCIF files, if present). "
            "Default: pdb uniclust30_filtered"
        ),
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit download to first N protein entries (useful for testing)",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download workers (default: 8)",
    )
    args = p.parse_args()

    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    if args.workers < 1:
        p.error("--workers must be >= 1")

    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OpenProteinSet v1 Downloader")
    print(f"  Bucket:       s3://{BUCKET}/")
    print(f"  Output dir:   {output_dir.resolve()}")
    print(f"  Subsets:      {args.subsets}")
    print("  Download mode: structures/caches only (MSA .a3m and template .hhr are skipped)")
    if args.limit:
        print(f"  Entry limit:  {args.limit}")
    print(f"  Workers:      {args.workers}")
    print("=" * 60)

    client = make_s3_client()

    for subset in args.subsets:
        prefix = SUBSET_PREFIXES[subset]
        download_subset(
            client=client,
            subset_name=subset,
            s3_prefix=prefix,
            output_dir=output_dir,
            limit=args.limit,
            workers=args.workers,
        )

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
