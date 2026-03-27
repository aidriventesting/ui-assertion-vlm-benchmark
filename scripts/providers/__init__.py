"""
VLM Provider abstraction layer.

Currently OpenAI-only. Logprobs (token-level probabilities) are required
for calibration analysis and confidence-based routing — only OpenAI
exposes these on vision API calls.

Anthropic and Gemini providers exist in this directory but are not
registered here. They can be re-enabled on a separate branch.
"""

from .base import VLMProvider
from .openai_provider import OpenAIProvider

PROVIDERS = {
    "openai": OpenAIProvider,
}


def get_provider(name: str, **kwargs) -> VLMProvider:
    """Get a VLM provider by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)
