#!/usr/bin/env python3
"""Build a ZIP64, uncompressed weights wheel without duplicating model data."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

from model_info import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE,
    WEIGHTS_DISTRIBUTION,
    WEIGHTS_PACKAGE,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "weights" / "src" / WEIGHTS_PACKAGE / "__init__.py"
LICENSE_SOURCE = ROOT / "LICENSE"
NOTICE_SOURCE = ROOT / "MODEL_NOTICE.md"


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _file_record_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def _metadata(version: str) -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        f"Name: {WEIGHTS_DISTRIBUTION}\n"
        f"Version: {version}\n"
        "Summary: Verified Gemma 4 E2B LiteRT-LM weights for gemma4-e2b-text\n"
        "License-Expression: Apache-2.0\n"
        "Requires-Python: >=3.10\n"
        "Project-URL: Model source, https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm\n"
        "\n"
        "This distribution contains the pinned official Gemma 4 E2B LiteRT-LM artifact.\n"
    ).encode()


def _wheel_metadata() -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: gemma4-e2b-text build_weights_wheel.py\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode()


def _write_bytes(
    archive: zipfile.ZipFile,
    records: list[tuple[str, str, str]],
    name: str,
    data: bytes,
) -> None:
    archive.writestr(name, data, compress_type=zipfile.ZIP_STORED)
    records.append((name, _record_hash(data), str(len(data))))


def build(version: str, output: Path, model: Path | None, placeholder: bool) -> Path:
    if placeholder:
        if ".dev" not in version:
            raise ValueError("placeholder builds require a developmental version such as 0.1.0.dev0")
    else:
        if model is None or not model.is_file():
            raise FileNotFoundError("stable weights build requires --model PATH")
        if model.stat().st_size != MODEL_SIZE:
            raise ValueError(f"model size mismatch: expected {MODEL_SIZE}, got {model.stat().st_size}")
        actual_hash = _file_record_hash(model).removeprefix("sha256=")
        expected_b64 = base64.urlsafe_b64encode(bytes.fromhex(MODEL_SHA256)).rstrip(b"=").decode()
        if actual_hash != expected_b64:
            raise ValueError("model SHA-256 does not match the pinned official artifact")

    normalized = WEIGHTS_DISTRIBUTION.replace("-", "_")
    filename = f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    output.mkdir(parents=True, exist_ok=True)
    wheel_path = output / filename
    records: list[tuple[str, str, str]] = []

    with zipfile.ZipFile(wheel_path, "w", allowZip64=True) as archive:
        _write_bytes(archive, records, f"{WEIGHTS_PACKAGE}/__init__.py", PACKAGE_SOURCE.read_bytes())
        _write_bytes(archive, records, f"{dist_info}/METADATA", _metadata(version))
        _write_bytes(archive, records, f"{dist_info}/WHEEL", _wheel_metadata())
        _write_bytes(archive, records, f"{dist_info}/licenses/LICENSE", LICENSE_SOURCE.read_bytes())
        _write_bytes(archive, records, f"{dist_info}/licenses/MODEL_NOTICE.md", NOTICE_SOURCE.read_bytes())
        if model is not None and not placeholder:
            arcname = f"{WEIGHTS_PACKAGE}/models/{MODEL_FILENAME}"
            archive.write(model, arcname, compress_type=zipfile.ZIP_STORED)
            records.append((arcname, _file_record_hash(model), str(MODEL_SIZE)))
        else:
            marker = b"Developmental placeholder; no model payload is included.\n"
            _write_bytes(archive, records, f"{WEIGHTS_PACKAGE}/models/PLACEHOLDER.txt", marker)

        record_name = f"{dist_info}/RECORD"
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(records)
        writer.writerow((record_name, "", ""))
        archive.writestr(record_name, buffer.getvalue().encode(), compress_type=zipfile.ZIP_STORED)

    return wheel_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--placeholder", action="store_true")
    args = parser.parse_args()
    try:
        wheel = build(args.version, args.out, args.model, args.placeholder)
    except Exception as exc:
        print(f"weights build failed: {exc}", file=sys.stderr)
        return 1
    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

