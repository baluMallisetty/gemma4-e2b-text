from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_weights_wheel.py"


def load_builder():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_weights_wheel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


class WeightsBuilderTests(unittest.TestCase):
    def test_placeholder_wheel_is_valid_shape(self):
        module = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            wheel = module.build("0.1.0.dev0", Path(directory), None, True)
            self.assertTrue(wheel.is_file())
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertIn("gemma4_e2b_text_weights/__init__.py", names)
                self.assertTrue(any(name.endswith(".dist-info/RECORD") for name in names))
                self.assertFalse(any(name.endswith(".litertlm") for name in names))

    def test_placeholder_cannot_use_stable_version(self):
        module = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                module.build("0.1.0", Path(directory), None, True)

    def test_stable_build_rejects_wrong_model(self):
        module = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "wrong.litertlm"
            model.write_bytes(b"not the model")
            with self.assertRaises(ValueError):
                module.build("0.1.0", Path(directory), model, False)


if __name__ == "__main__":
    unittest.main()
