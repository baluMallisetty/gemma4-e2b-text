"""Resumable download and verified local caching of the Gemma 4 model."""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


MODEL_FILENAME = "gemma-4-E2B-it.litertlm"
MODEL_SIZE = 2_588_147_712
MODEL_SHA256 = "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
MODEL_RELEASE = "model-v0.1.0"
MODEL_CACHE_ENVIRONMENT_VARIABLE = "GEMMA4_E2B_CACHE_DIR"
REPOSITORY = "baluMallisetty/gemma4-e2b-text"
RELEASE_ROOT = f"https://github.com/{REPOSITORY}/releases/download/{MODEL_RELEASE}"


@dataclass(frozen=True)
class Asset:
    filename: str
    size: int
    url: str


PART_SIZE = MODEL_SIZE // 2
MODEL_ASSETS = (
    Asset(
        f"{MODEL_FILENAME}.part-001",
        PART_SIZE,
        f"{RELEASE_ROOT}/{MODEL_FILENAME}.part-001",
    ),
    Asset(
        f"{MODEL_FILENAME}.part-002",
        MODEL_SIZE - PART_SIZE,
        f"{RELEASE_ROOT}/{MODEL_FILENAME}.part-002",
    ),
)

ProgressCallback = Callable[[int, int], None]


def default_model_directory() -> Path:
    """Return the platform-appropriate persistent model-cache directory."""
    configured = os.environ.get(MODEL_CACHE_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "gemma4-e2b-text" / "models"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(
    path: Path,
    *,
    expected_size: int = MODEL_SIZE,
    expected_sha256: str = MODEL_SHA256,
) -> None:
    """Fail if a model does not match the pinned official artifact."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"model size mismatch: expected {expected_size}, got {actual_size}")
    actual_hash = sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"model SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )


@contextmanager
def _download_lock(
    directory: Path,
    *,
    timeout: float = 6 * 60 * 60,
    stale_after: float = 24 * 60 * 60,
) -> Iterator[None]:
    lock = directory / ".model-download.lock"
    deadline = time.monotonic() + timeout
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode())
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > stale_after
            except FileNotFoundError:
                continue
            if stale:
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for another model download")
            time.sleep(1)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def _request(asset: Asset, offset: int) -> tuple[BinaryIO, bool]:
    headers = {"User-Agent": "gemma4-e2b-text/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = urllib.request.urlopen(
        urllib.request.Request(asset.url, headers=headers), timeout=120
    )
    status = getattr(response, "status", None)
    return response, bool(offset and status == 206)


def _download_assets(
    target: Path,
    assets: Sequence[Asset],
    *,
    expected_size: int,
    expected_sha256: str,
    progress: ProgressCallback | None = None,
) -> Path:
    partial = target.with_suffix(target.suffix + ".partial")
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > expected_size:
        partial.unlink()
        downloaded = 0

    cumulative = 0
    for asset in assets:
        asset_end = cumulative + asset.size
        if downloaded >= asset_end:
            cumulative = asset_end
            continue

        offset = max(0, downloaded - cumulative)
        response, resumed = _request(asset, offset)
        if offset and not resumed:
            response.close()
            with partial.open("r+b") as output:
                output.truncate(cumulative)
            downloaded = cumulative
            offset = 0
            response, _ = _request(asset, 0)

        remaining = asset.size - offset
        copied = 0
        with response, partial.open("ab") as output:
            while copied < remaining:
                chunk = response.read(min(8 * 1024 * 1024, remaining - copied))
                if not chunk:
                    break
                output.write(chunk)
                copied += len(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, expected_size)
        if copied != remaining:
            raise OSError(
                f"incomplete download for {asset.filename}: expected {remaining}, got {copied}"
            )
        cumulative = asset_end

    if partial.stat().st_size != expected_size:
        raise OSError(
            f"assembled model size mismatch: expected {expected_size}, "
            f"got {partial.stat().st_size}"
        )
    actual_hash = sha256(partial)
    if actual_hash != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"assembled model SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    partial.replace(target)
    return target


def download_model(
    directory: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download, assemble, verify, and cache the model from GitHub Releases.

    Interrupted downloads resume from the existing ``.partial`` file. The
    completed model is moved into place only after its official SHA-256 passes.
    """
    model_directory = (
        Path(directory).expanduser() if directory is not None else default_model_directory()
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    target = model_directory / MODEL_FILENAME
    if target.is_file() and not force:
        if target.stat().st_size == MODEL_SIZE:
            return target.resolve()
        target.unlink()

    with _download_lock(model_directory):
        if target.is_file() and not force and target.stat().st_size == MODEL_SIZE:
            return target.resolve()
        if force:
            target.unlink(missing_ok=True)
            target.with_suffix(target.suffix + ".partial").unlink(missing_ok=True)
        return _download_assets(
            target,
            MODEL_ASSETS,
            expected_size=MODEL_SIZE,
            expected_sha256=MODEL_SHA256,
            progress=progress,
        ).resolve()
