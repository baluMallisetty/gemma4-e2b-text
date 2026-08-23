#!/usr/bin/env python3
"""Download the pinned official model and verify it before release building."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from model_info import MODEL_FILENAME, MODEL_SHA256, MODEL_SIZE, MODEL_URL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    actual_size = path.stat().st_size
    if actual_size != MODEL_SIZE:
        raise ValueError(f"size mismatch: expected {MODEL_SIZE}, got {actual_size}")
    actual_hash = sha256(path)
    if actual_hash != MODEL_SHA256:
        raise ValueError(f"SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}")


def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / MODEL_FILENAME
    if target.exists():
        verify(target)
        print(f"Already verified: {target}")
        return target

    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "gemma4-e2b-text-release/0.1"})
    print(f"Downloading {MODEL_SIZE / 1_000_000_000:.2f} GB to {partial}")
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        copied = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            print(f"\r{copied / MODEL_SIZE:6.1%}", end="", flush=True)
    print()
    verify(partial)
    partial.replace(target)
    print(f"Verified SHA-256: {MODEL_SHA256}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("model"))
    args = parser.parse_args()
    try:
        download(args.destination)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

