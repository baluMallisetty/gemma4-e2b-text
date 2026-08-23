"""Small text-only Python interface for Gemma 4 E2B."""

from .download import default_model_directory, download_model
from .model import Conversation, Gemma4, generate

__all__ = [
    "Conversation",
    "Gemma4",
    "default_model_directory",
    "download_model",
    "generate",
]
__version__ = "0.1.0"
