#!/usr/bin/env python3
"""Upload data/ and results/ to HuggingFace Hub."""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "kevinlu4588/ProteinFolding"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["data", "results"]
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "*.swp",
    "*.swo",
    ".ipynb_checkpoints",
]
MAX_RETRIES = 5
RETRY_DELAY = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload data/ and results/ to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Make the repo public (default: private)",
    )
    parser.add_argument(
        "--folders", nargs="+", default=FOLDERS,
        help="Which folders to upload (default: data and results)",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def ensure_authenticated(api: HfApi, token: str | None):
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


def upload_with_retry(api: HfApi, **kwargs):
    """Call api.upload_folder with retries on 504/429 errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.upload_folder(**kwargs)
            return
        except Exception as e:
            err = str(e)
            if ("504" in err or "429" in err) and attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (2 if "429" in err else 1)
                print(f"    Error (attempt {attempt}/{MAX_RETRIES}), "
                      f"retrying in {delay}s...")
                print(e)
                time.sleep(delay)
            else:
                raise


def upload_folder_chunked(api: HfApi, folder_path: Path, folder_name: str,
                          repo_id: str, ignore_patterns: list[str]):
    """Upload per-subfolder to keep commits small."""
    skip_dirs = {"__pycache__", ".ipynb_checkpoints"}
    subdirs = sorted([d for d in folder_path.iterdir()
                      if d.is_dir() and d.name not in skip_dirs])
    files = [f for f in folder_path.iterdir()
             if f.is_file() and not any(f.match(p) for p in ignore_patterns)]

    if files:
        print(f"  Uploading {folder_name}/ (top-level files)...")
        upload_with_retry(
            api,
            folder_path=str(folder_path),
            path_in_repo=folder_name,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f.name for f in files],
            ignore_patterns=ignore_patterns,
        )

    for i, subdir in enumerate(subdirs, 1):
        print(f"  [{i}/{len(subdirs)}] Uploading {folder_name}/{subdir.name}/ ...")
        upload_with_retry(
            api,
            folder_path=str(subdir),
            path_in_repo=f"{folder_name}/{subdir.name}",
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
        )


def main():
    args = parse_args()
    api = HfApi()

    ensure_authenticated(api, args.token)
    user = api.whoami()["name"]
    print(f"Authenticated as: {user}")

    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    print(f"Repo ready: {repo_url}")

    for folder in args.folders:
        folder_path = PROJECT_ROOT / folder
        if not folder_path.is_dir():
            print(f"WARNING: {folder_path} does not exist, skipping.")
            continue

        print(f"\nUploading {folder}/ ...")
        upload_folder_chunked(api, folder_path, folder, args.repo_id, IGNORE_PATTERNS)
        print(f"  Done: {folder}/")

    print(f"\nAll uploads complete!")
    print(f"View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()