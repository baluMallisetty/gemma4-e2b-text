"""Gemma 4 E2B text inference backed by Google's LiteRT-LM runtime."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .download import download_model


MODEL_ENVIRONMENT_VARIABLE = "GEMMA4_E2B_MODEL"


def resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit/environment path or download the release model."""
    candidate = model_path or os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    path = Path(candidate).expanduser() if candidate else download_model()
    if not path.is_file():
        raise FileNotFoundError(f"Gemma 4 model file not found: {path}")
    if path.suffix.lower() != ".litertlm":
        raise ValueError(f"Expected a .litertlm model file, got: {path}")
    return path.resolve()


def _load_runtime() -> Any:
    try:
        import litert_lm
    except ImportError as exc:
        raise RuntimeError(
            "LiteRT-LM is not installed for this platform. Supported targets are "
            "Windows x86-64, Linux x86-64/aarch64, macOS arm64, and Android."
        ) from exc
    return litert_lm


def _backend(runtime: Any, name: str) -> Any:
    normalized = name.strip().upper()
    if normalized not in {"CPU", "GPU", "NPU"}:
        raise ValueError("backend must be one of: cpu, gpu, npu")
    factory = getattr(runtime.Backend, normalized, None)
    if factory is None:
        raise RuntimeError(f"The installed LiteRT-LM does not provide {normalized}.")
    return factory()


def _text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected LiteRT-LM response type: {type(response).__name__}")
    parts: list[str] = []
    for item in response.get("content", []):
        if isinstance(item, dict) and item.get("type", "text") == "text":
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
    if not parts:
        raise RuntimeError("LiteRT-LM returned no text content.")
    return "".join(parts)


class Conversation:
    """A stateful text conversation created by :class:`Gemma4`."""

    def __init__(self, conversation: Any):
        self._conversation = conversation

    def send(self, prompt: str) -> str:
        """Send one message and return the complete text response."""
        return _text(self._conversation.send_message(prompt))

    def stream(self, prompt: str) -> Iterator[str]:
        """Send one message and yield text fragments as they arrive."""
        for chunk in self._conversation.send_message_async(prompt):
            if isinstance(chunk, dict):
                for item in chunk.get("content", []):
                    if isinstance(item, dict) and item.get("type", "text") == "text":
                        value = item.get("text")
                        if isinstance(value, str):
                            yield value


class Gemma4:
    """Context-managed local Gemma 4 E2B engine.

    On first use, the model is downloaded from this project's GitHub Release,
    verified, and cached for offline reuse. The public interface accepts text
    only. The official LiteRT-LM bundle may
    contain optional multimodal components, but no vision or audio backend is
    initialized here.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        backend: str = "cpu",
        cache_dir: str | os.PathLike[str] | None = None,
        speculative_decoding: bool | None = None,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.backend_name = backend
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.speculative_decoding = speculative_decoding
        self._engine_manager: Any | None = None
        self._engine: Any | None = None
        self._runtime: Any | None = None

    def __enter__(self) -> Gemma4:
        runtime = _load_runtime()
        kwargs: dict[str, Any] = {"backend": _backend(runtime, self.backend_name)}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir.resolve())
        enabled = self.speculative_decoding
        if enabled is None:
            enabled = self.backend_name.strip().lower() == "gpu"
        if enabled:
            kwargs["enable_speculative_decoding"] = True
        manager = runtime.Engine(str(self.model_path), **kwargs)
        self._engine_manager = manager
        self._engine = manager.__enter__()
        self._runtime = runtime
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        manager = self._engine_manager
        self._engine = None
        self._engine_manager = None
        self._runtime = None
        if manager is not None:
            return manager.__exit__(exc_type, exc, traceback)
        return None

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("Use Gemma4 as a context manager: `with Gemma4() as model:`")
        return self._engine

    @contextmanager
    def conversation(self, *, system: str | None = None) -> Iterator[Conversation]:
        """Create a stateful conversation, optionally with a system instruction."""
        engine = self._require_engine()
        messages = None
        if system:
            assert self._runtime is not None
            messages = [self._runtime.Message.system(system)]
        with engine.create_conversation(messages=messages) as conversation:
            yield Conversation(conversation)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate one complete response in a fresh conversation."""
        with self.conversation(system=system) as conversation:
            return conversation.send(prompt)

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        """Yield response fragments from a fresh conversation."""
        with self.conversation(system=system) as conversation:
            yield from conversation.stream(prompt)


def generate(
    prompt: str,
    *,
    system: str | None = None,
    backend: str = "cpu",
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    """Convenience function for one local generation."""
    with Gemma4(model_path=model_path, backend=backend) as model:
        return model.generate(prompt, system=system)
