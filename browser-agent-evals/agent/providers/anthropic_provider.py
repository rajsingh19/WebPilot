"""Anthropic LLM provider implementation using the Anthropic Python SDK."""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from anthropic import (
    APIConnectionError,
    APIStatusError,
    Anthropic,
    RateLimitError,
)

from agent.providers.base import Decision, LLMError, LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by Anthropic's Claude models."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[Anthropic] = None,
        max_retries: int = 4,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            client=client,
            max_retries=max_retries,
        )
        if self.client is None:
            if not self.api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set in environment or passed to AnthropicProvider."
                )
            self.client = Anthropic(api_key=self.api_key)

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts canonical tools into Anthropic tool format."""
        anthropic_tools: List[Dict[str, Any]] = []
        for tool in tools:
            schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
            anthropic_tools.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": schema,
                }
            )
        return anthropic_tools

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determines if the exception is due to rate limits, connection issues, or 5xx server errors."""
        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500:
            return True
        return False

    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Invokes Anthropic with forced tool choice and returns a Decision."""
        anthropic_tools = self._convert_tools(tools)
        backoff = 1.0
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "AnthropicProvider requesting decision (model=%s, attempt=%d/%d)",
                    self.model_name,
                    attempt,
                    self.max_retries,
                )
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    tools=anthropic_tools,
                    tool_choice={"type": "any"},
                )

                tool_calls: List[Any] = []
                for block in response.content:
                    if getattr(block, "type", "") == "tool_use":
                        tool_calls.append(block)

                if not tool_calls:
                    raise LLMError("Anthropic model returned no tool call.")

                selected_block = tool_calls[0]
                tool_name = selected_block.name
                raw_args = selected_block.input

                if not isinstance(raw_args, dict):
                    raise LLMError(
                        f"Malformed arguments in Anthropic tool call '{tool_name}': {raw_args}"
                    )

                tool_args = dict(raw_args)
                reasoning = str(tool_args.pop("reasoning", "") or "")

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                return Decision(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    reasoning=reasoning,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except LLMError:
                raise
            except Exception as exc:
                if self._is_retryable_error(exc):
                    last_exception = exc
                    jitter = random.uniform(0.1, 0.5 * backoff)
                    sleep_time = backoff + jitter
                    logger.warning(
                        "Anthropic retryable error on attempt %d/%d: %s. Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        exc,
                        sleep_time,
                    )
                    if attempt < self.max_retries:
                        time.sleep(sleep_time)
                        backoff *= 2.0
                    else:
                        raise LLMError(
                            f"Anthropic API failed after {self.max_retries} retries: {exc}"
                        ) from exc
                else:
                    logger.error("Non-retryable Anthropic API error: %s", exc)
                    raise LLMError(f"Anthropic API non-retryable error: {exc}") from exc

        raise LLMError(f"Anthropic provider failed: {last_exception}")
