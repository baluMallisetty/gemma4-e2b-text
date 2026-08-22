"""Filesystem access to the bundled Gemma 4 E2B LiteRT-LM model."""

from importlib.resources import files
from pathlib import Path

MODEL_FILENAME = "gemma-4-E2B-it.litertlm"
MODEL_SHA256 = "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
MODEL_SIZE = 2_588_147_712


def model_path() -> Path:
    """Return the installed model path or fail clearly for a placeholder build."""
    resource = files(__package__).joinpath("models", MODEL_FILENAME)
    path = Path(str(resource))
    if not path.is_file():
        raise RuntimeError(
            "This is a placeholder weights release and contains no model. "
            "Install the stable gemma4-e2b-text-weights release."
        )
    return path


__all__ = ["MODEL_FILENAME", "MODEL_SHA256", "MODEL_SIZE", "model_path"]

