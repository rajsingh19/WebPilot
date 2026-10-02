"""Grok compatibility alias module pointing to GroqProvider."""

from agent.providers.groq_provider import GroqProvider

# Alias for backward compatibility
GrokProvider = GroqProvider

__all__ = ["GroqProvider", "GrokProvider"]
