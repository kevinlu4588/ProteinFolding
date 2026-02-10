#!/usr/bin/env python3
"""Download data/, models/, and results/ from HuggingFace Hub."""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download

REPO_ID = "kevinlu4588/ProteinFolding"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["data", "results"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download data/, models/, and results/ from HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files (default: skip them)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=FOLDERS,
        choices=FOLDERS,
        help="Which folders to download (default: all three)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def ensure_authenticated(api: HfApi, token: str | None):
    """Make sure we have a valid HF token."""
    if token:
        api.token = token
        return
    try:
        api.whoami()
        return
    except Exception:
        pass
    print("Not logged in to HuggingFace. Launching interactive login...")
    login()


def copy_tree(src: Path, dst: Path, force: bool) -> tuple[int, int]:
    """Recursively copy files from src to dst. Returns (copied, skipped) counts."""
    copied = 0
    skipped = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if dst_file.exists() and not force:
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied, skipped


def main():
    args = parse_args()
    api = HfApi()
    ensure_authenticated(api, args.token)
    user = api.whoami()["name"]
    print(f"Authenticated as: {user}")

    # Build allow_patterns so we only download requested folders
    allow_patterns = [f"{folder}/**" for folder in args.folders]

    print(f"Downloading snapshot from {args.repo_id} ...")
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            token=args.token,
        )
    )
    print(f"Snapshot cached at: {snapshot_path}")

    total_copied = 0
    total_skipped = 0

    for folder in args.folders:
        src = snapshot_path / folder
        dst = PROJECT_ROOT / folder
        if not src.is_dir():
            print(f"WARNING: {folder}/ not found in repo, skipping.")
            continue

        copied, skipped = copy_tree(src, dst, force=args.force)
        total_copied += copied
        total_skipped += skipped
        print(f"  {folder}/: {copied} copied, {skipped} skipped")

    print(f"\nDone! {total_copied} files copied, {total_skipped} files skipped.")
    if total_skipped > 0 and not args.force:
        print("  (use --force to overwrite existing files)")


if __name__ == "__main__":
    main()
