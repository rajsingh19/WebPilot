"""Base abstractions and data contracts for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMError(Exception):
    """Raised when an LLM provider fails, returns invalid output, or exhausts retries."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_retryable = is_retryable


class LLMConfigError(LLMError):
    """Raised when an LLM provider encounters a configuration or authentication error (400, 401, 403)."""

    pass


class LLMInfraError(LLMError):
    """Raised when an LLM provider encounters an infrastructure error (429, 500, 502, 503, 504, timeout, or model not found)."""

    pass


@dataclass
class Decision:
    """Represents an action decision produced by an LLM provider."""

    tool_name: str
    tool_args: Dict[str, Any]
    reasoning: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        max_retries: int = 4,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.client = client
        self.max_retries = max_retries

    @abstractmethod
    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Evaluates system instructions, messages, and tools to produce a single Decision.

        Args:
            system: The system instruction prompt.
            messages: A list of conversation messages (e.g. role/content dicts).
            tools: Canonical tool definitions list in JSON schema format.

        Returns:
            Decision dataclass containing the chosen tool, arguments, reasoning, and token usage.

        Raises:
            LLMError: If tool calling fails, args are malformed, or retries are exhausted.
        """
        pass
