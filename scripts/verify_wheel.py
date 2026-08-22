#!/usr/bin/env python3
"""Verify wheel structure, metadata, and RECORD hashes without installing it."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("wheel must contain exactly one RECORD")
        record_name = record_names[0]
        rows = csv.reader(io.StringIO(archive.read(record_name).decode()))
        seen: set[str] = set()
        for name, digest, size in rows:
            seen.add(name)
            if name == record_name:
                if digest or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            hash_object = hashlib.sha256()
            actual_size = 0
            with archive.open(name) as member:
                for chunk in iter(lambda: member.read(8 * 1024 * 1024), b""):
                    hash_object.update(chunk)
                    actual_size += len(chunk)
            actual = base64.urlsafe_b64encode(hash_object.digest()).rstrip(b"=").decode()
            if digest != f"sha256={actual}":
                raise ValueError(f"bad hash for {name}")
            if size != str(actual_size):
                raise ValueError(f"bad size for {name}")
        if seen != set(names):
            raise ValueError("RECORD and archive members differ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify(args.wheel)
    print(f"Verified: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
