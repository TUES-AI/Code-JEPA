#!/usr/bin/env python3
"""Download and prepare CodeXGLUE clone-detection benchmarks for offline runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BIGCLONEBENCH_URLS = {
    "data.jsonl": (
        "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
        "Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl"
    ),
    "train.txt": (
        "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
        "Code-Code/Clone-detection-BigCloneBench/dataset/train.txt"
    ),
    "valid.txt": (
        "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
        "Code-Code/Clone-detection-BigCloneBench/dataset/valid.txt"
    ),
    "test.txt": (
        "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
        "Code-Code/Clone-detection-BigCloneBench/dataset/test.txt"
    ),
}
POJ_PREPROCESS_URL = (
    "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
    "Code-Code/Clone-detection-POJ-104/dataset/preprocess.py"
)
POJ_GDRIVE_ID = "0B2i-vWnOu7MxVlJwQXN6eVNONUU"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmarks/codexglue"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["bigclonebench", "poj104"],
        default=["bigclonebench", "poj104"],
    )
    parser.add_argument("--prepare-poj", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(args.output_root),
        "benchmarks": args.benchmarks,
        "sources": {},
    }

    if "bigclonebench" in args.benchmarks:
        manifest["sources"] = {
            **dict(manifest["sources"]),
            "bigclonebench": BIGCLONEBENCH_URLS,
        }
        download_bigclonebench(args)
    if "poj104" in args.benchmarks:
        manifest["sources"] = {
            **dict(manifest["sources"]),
            "poj104": {
                "preprocess.py": POJ_PREPROCESS_URL,
                "programs.tar.gz": f"https://drive.google.com/uc?id={POJ_GDRIVE_ID}",
            },
        }
        download_poj104(args)

    manifest_path = args.output_root / "download_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def download_bigclonebench(args: argparse.Namespace) -> None:
    out = args.output_root / "bigclonebench"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    for name, url in BIGCLONEBENCH_URLS.items():
        download(url, out / name, skip_existing=args.skip_existing, dry_run=args.dry_run)


def download_poj104(args: argparse.Namespace) -> None:
    out = args.output_root / "poj104"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    preprocess = out / "preprocess.py"
    download(POJ_PREPROCESS_URL, preprocess, skip_existing=args.skip_existing, dry_run=args.dry_run)

    archive = out / "programs.tar.gz"
    if args.prepare_poj:
        if not archive.exists() or not args.skip_existing:
            gdown = shutil.which("gdown")
            command = (
                [gdown, f"https://drive.google.com/uc?id={POJ_GDRIVE_ID}", "-O", str(archive)]
                if gdown
                else [
                    sys.executable,
                    "-m",
                    "gdown",
                    f"https://drive.google.com/uc?id={POJ_GDRIVE_ID}",
                    "-O",
                    str(archive),
                ]
            )
            run(command, dry_run=args.dry_run)
        if not args.dry_run:
            safe_extract_tar(archive, out)
            run([sys.executable, str(preprocess.name)], cwd=out, dry_run=False)
    else:
        print(
            "POJ-104 metadata downloaded. Re-run with --prepare-poj after installing "
            "gdown to fetch programs.tar.gz and build train/valid/test jsonl."
        )


def download(url: str, dest: Path, *, skip_existing: bool, dry_run: bool) -> None:
    if skip_existing and dest.exists():
        print(f"exists {dest}")
        return
    print(f"download {url} -> {dest}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def safe_extract_tar(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        tar.extractall(dest)


def run(command: list[str], *, cwd: Path | None = None, dry_run: bool) -> None:
    print("run", " ".join(command), f"(cwd={cwd})" if cwd else "")
    if dry_run:
        return
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        if "gdown" in command[0] or (len(command) > 2 and command[2] == "gdown"):
            raise RuntimeError(
                "POJ-104 requires gdown for the Google Drive archive. "
                "Install it on the staging machine with `python -m pip install gdown`."
            ) from exc
        raise


if __name__ == "__main__":
    main()
