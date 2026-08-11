"""LLM abstraction. Nothing outside this package imports litellm or instructor (§7)."""

from .base import LLMUnavailable, StructuredLLM
from .prompts import Prompt, PromptError, format_interest_profile, load_prompt

__all__ = [
    "LLMUnavailable",
    "Prompt",
    "PromptError",
    "StructuredLLM",
    "format_interest_profile",
    "load_prompt",
]


def build_llm_client(settings: object, store: object = None) -> StructuredLLM:
    """Construct the real client. Imported lazily so tests never load litellm."""
    from .client import LLMClient

    return LLMClient(settings, store)  # type: ignore[arg-type]
