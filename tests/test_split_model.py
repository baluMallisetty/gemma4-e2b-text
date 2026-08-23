from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "split_model.py"


def load_splitter():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("split_model", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class SplitModelTests(unittest.TestCase):
    def test_creates_two_parts_and_manifest(self):
        module = load_splitter()
        model_data = b"0123456789"
        module.MODEL_SIZE = len(model_data)
        module.PART_SIZE = len(model_data) // 2
        module.MODEL_SHA256 = hashlib.sha256(model_data).hexdigest()
        module.MODEL_FILENAME = "test-model.litertlm"

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            model = directory / module.MODEL_FILENAME
            model.write_bytes(model_data)
            output = directory / "release"
            paths = module.split_model(model, output)

            self.assertEqual(len(paths), 3)
            part_one = output / "test-model.litertlm.part-001"
            part_two = output / "test-model.litertlm.part-002"
            self.assertEqual(part_one.read_bytes() + part_two.read_bytes(), model_data)
            manifest = json.loads((output / "model-manifest.json").read_text())
            self.assertEqual(manifest["model"]["sha256"], module.MODEL_SHA256)
            self.assertEqual(len(manifest["model"]["parts"]), 2)


if __name__ == "__main__":
    unittest.main()

