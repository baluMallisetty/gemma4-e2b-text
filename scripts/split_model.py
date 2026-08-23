#!/usr/bin/env python3
"""Split the pinned model into GitHub Release assets below the 2 GiB limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from model_info import MODEL_FILENAME, MODEL_SHA256, MODEL_SIZE


PART_SIZE = MODEL_SIZE // 2
RELEASE_TAG = "model-v0.1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> None:
    if path.stat().st_size != MODEL_SIZE:
        raise ValueError(
            f"model size mismatch: expected {MODEL_SIZE}, got {path.stat().st_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != MODEL_SHA256:
        raise ValueError(f"model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}")


def split_model(model: Path, output: Path) -> list[Path]:
    verify_model(model)
    output.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, str | int]] = []
    paths: list[Path] = []

    with model.open("rb") as source:
        for number, expected_size in enumerate(
            (PART_SIZE, MODEL_SIZE - PART_SIZE), start=1
        ):
            name = f"{MODEL_FILENAME}.part-{number:03d}"
            destination = output / name
            temporary = destination.with_suffix(destination.suffix + ".partial")
            digest = hashlib.sha256()
            written = 0
            with temporary.open("wb") as target:
                while written < expected_size:
                    chunk = source.read(min(8 * 1024 * 1024, expected_size - written))
                    if not chunk:
                        raise OSError(f"model ended while creating {name}")
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            temporary.replace(destination)
            paths.append(destination)
            parts.append(
                {"filename": name, "size": written, "sha256": digest.hexdigest()}
            )
        if source.read(1):
            raise OSError("model contains bytes beyond the pinned size")

    manifest = {
        "schema_version": 1,
        "release": RELEASE_TAG,
        "model": {
            "filename": MODEL_FILENAME,
            "size": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "parts": parts,
        },
    }
    manifest_path = output / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths.append(manifest_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("release-assets"))
    args = parser.parse_args()
    try:
        for path in split_model(args.model, args.output):
            print(f"{path} ({path.stat().st_size:,} bytes)")
    except Exception as exc:
        print(f"split failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

