"""Small text-only Python interface for the bundled Gemma 4 E2B model."""

from .model import Conversation, Gemma4, generate

__all__ = ["Conversation", "Gemma4", "generate"]
__version__ = "0.1.0"

