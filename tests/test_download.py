from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemma4_e2b_text.download import Asset, _download_assets


class DownloadTests(unittest.TestCase):
    def _fixture(self, directory: Path):
        first_data = b"first-model-part"
        second_data = b"second-model-part"
        first = directory / "part-001"
        second = directory / "part-002"
        first.write_bytes(first_data)
        second.write_bytes(second_data)
        assets = (
            Asset(first.name, len(first_data), first.as_uri()),
            Asset(second.name, len(second_data), second.as_uri()),
        )
        complete = first_data + second_data
        return assets, complete

    def test_joins_parts_and_verifies_hash(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            assets, complete = self._fixture(directory)
            target = directory / "model.litertlm"
            progress = []
            result = _download_assets(
                target,
                assets,
                expected_size=len(complete),
                expected_sha256=hashlib.sha256(complete).hexdigest(),
                progress=lambda current, total: progress.append((current, total)),
            )
            self.assertEqual(result.read_bytes(), complete)
            self.assertFalse(target.with_suffix(".litertlm.partial").exists())
            self.assertEqual(progress[-1], (len(complete), len(complete)))

    def test_resumes_at_part_boundary(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            assets, complete = self._fixture(directory)
            target = directory / "model.litertlm"
            partial = target.with_suffix(".litertlm.partial")
            partial.write_bytes((directory / "part-001").read_bytes())
            _download_assets(
                target,
                assets,
                expected_size=len(complete),
                expected_sha256=hashlib.sha256(complete).hexdigest(),
            )
            self.assertEqual(target.read_bytes(), complete)

    def test_resumes_inside_a_part_with_http_range(self):
        first_data = b"first-part"
        second_data = b"second-part"
        complete = first_data + second_data
        assets = (
            Asset("part-001", len(first_data), "https://example.test/part-001"),
            Asset("part-002", len(second_data), "https://example.test/part-002"),
        )

        class Response(io.BytesIO):
            def __init__(self, value: bytes, status: int):
                super().__init__(value)
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 120)
            payload = second_data if request.full_url.endswith("part-002") else first_data
            range_header = request.get_header("Range")
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            return Response(payload[offset:], 206)

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            target = directory / "model.litertlm"
            partial = target.with_suffix(".litertlm.partial")
            partial.write_bytes(first_data + second_data[:3])
            with patch("gemma4_e2b_text.download.urllib.request.urlopen", fake_urlopen):
                _download_assets(
                    target,
                    assets,
                    expected_size=len(complete),
                    expected_sha256=hashlib.sha256(complete).hexdigest(),
                )
            self.assertEqual(target.read_bytes(), complete)

    def test_rejects_corrupt_assembly(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            assets, complete = self._fixture(directory)
            target = directory / "model.litertlm"
            with self.assertRaises(ValueError):
                _download_assets(
                    target,
                    assets,
                    expected_size=len(complete),
                    expected_sha256="0" * 64,
                )
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".litertlm.partial").exists())


if __name__ == "__main__":
    unittest.main()
