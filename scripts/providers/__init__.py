"""
VLM Provider abstraction layer.

Supports multiple VLM providers: OpenAI, Gemini, Anthropic.
"""

from .base import VLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str, **kwargs) -> VLMProvider:
    """Get a VLM provider by name.
    
    Args:
        name: Provider name (openai, gemini, anthropic)
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured VLMProvider instance
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)


def get_all_providers(**kwargs) -> dict[str, VLMProvider]:
    """Get all available VLM providers.
    
    Returns:
        Dict mapping provider name to configured instance
    """
    providers = {}
    for name, cls in PROVIDERS.items():
        try:
            providers[name] = cls(**kwargs)
        except ValueError as e:
            print(f"⚠️  Skipping {name}: {e}")
    return providers
