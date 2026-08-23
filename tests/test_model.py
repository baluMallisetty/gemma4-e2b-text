from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gemma4_e2b_text import Gemma4, generate
from gemma4_e2b_text.model import resolve_model_path


class FakeConversation:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def send_message(self, prompt):
        return {"content": [{"type": "text", "text": f"answer:{prompt}"}]}

    def send_message_async(self, prompt):
        yield {"content": [{"type": "text", "text": "answer:"}]}
        yield {"content": [{"type": "text", "text": prompt}]}


class FakeEngine:
    latest_kwargs = None

    def __init__(self, path, **kwargs):
        self.path = path
        FakeEngine.latest_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def create_conversation(self, messages=None):
        self.messages = messages
        return FakeConversation()


class FakeBackend:
    @staticmethod
    def CPU():
        return "cpu-backend"

    @staticmethod
    def GPU():
        return "gpu-backend"

    @staticmethod
    def NPU():
        return "npu-backend"


class FakeMessage:
    @staticmethod
    def system(value):
        return ("system", value)


class ModelTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".litertlm", delete=False)
        handle.close()
        self.model_path = Path(handle.name)
        self.runtime = types.SimpleNamespace(Engine=FakeEngine, Backend=FakeBackend, Message=FakeMessage)
        self.runtime_patch = patch.dict(sys.modules, {"litert_lm": self.runtime})
        self.runtime_patch.start()

    def tearDown(self):
        self.runtime_patch.stop()
        self.model_path.unlink(missing_ok=True)

    def test_generate_function(self):
        result = generate("hello", model_path=self.model_path)
        self.assertEqual(result, "answer:hello")
        self.assertEqual(FakeEngine.latest_kwargs["backend"], "cpu-backend")

    def test_gpu_stream_enables_speculative_decoding(self):
        with Gemma4(self.model_path, backend="gpu") as model:
            self.assertEqual("".join(model.stream("hello")), "answer:hello")
        self.assertTrue(FakeEngine.latest_kwargs["enable_speculative_decoding"])

    def test_stateful_conversation_and_system_message(self):
        with Gemma4(self.model_path) as model:
            with model.conversation(system="Be concise") as conversation:
                self.assertEqual(conversation.send("one"), "answer:one")

    def test_model_must_be_litertlm(self):
        wrong = self.model_path.with_suffix(".bin")
        self.model_path.rename(wrong)
        self.model_path = wrong
        with self.assertRaises(ValueError):
            resolve_model_path(wrong)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            with Gemma4(self.model_path, backend="quantum"):
                pass


if __name__ == "__main__":
    unittest.main()

