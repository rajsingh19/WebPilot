"""Grok (xAI) LLM provider implementation using OpenAI SDK."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from agent.providers.base import Decision, LLMError, LLMProvider

logger = logging.getLogger(__name__)


class GrokProvider(LLMProvider):
    """LLM provider backed by xAI's Grok models using the OpenAI-compatible SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        max_retries: int = 3,
        base_url: str = "https://api.x.ai/v1",
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("XAI_API_KEY"),
            client=client,
            max_retries=max_retries,
        )
        self.base_url = base_url
        if self.client is None:
            if not self.api_key:
                raise LLMError(
                    "XAI_API_KEY is not set in environment or passed to GrokProvider."
                )
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts canonical tools into OpenAI/xAI tool format."""
        openai_tools: List[Dict[str, Any]] = []
        for tool in tools:
            schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
        return openai_tools

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determines if the exception is due to rate limits, connection issues, or 5xx server errors."""
        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500:
            return True
        if isinstance(exc, (httpx.RequestError, ConnectionError, TimeoutError)):
            return True
        return False

    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Invokes Grok with tool_choice='required' and returns a Decision."""
        grok_tools = self._convert_tools(tools)
        openai_messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        openai_messages.extend(messages)

        backoff = 1.0
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "GrokProvider requesting decision (model=%s, attempt=%d/%d)",
                    self.model_name,
                    attempt,
                    self.max_retries,
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    tools=grok_tools,
                    tool_choice="required",
                )

                if not response.choices:
                    raise LLMError("Grok returned empty response choices.")

                choice = response.choices[0]
                message = choice.message

                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    raise LLMError("Grok model returned no tool call.")

                first_tc = tool_calls[0]
                tool_name = first_tc.function.name
                raw_args = first_tc.function.arguments

                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            f"Malformed arguments in Grok tool call '{tool_name}': {raw_args}"
                        ) from exc
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                elif raw_args is None:
                    parsed_args = {}
                else:
                    raise LLMError(
                        f"Malformed arguments in Grok tool call '{tool_name}': {raw_args}"
                    )

                tool_args = dict(parsed_args)
                reasoning = str(tool_args.pop("reasoning", "") or "")

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

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
                    logger.warning(
                        "Grok retryable error on attempt %d/%d: %s. Retrying in %.1fs...",
                        attempt,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                    else:
                        raise LLMError(
                            f"Grok API failed after {self.max_retries} retries: {exc}"
                        ) from exc
                else:
                    logger.error("Non-retryable Grok API error: %s", exc)
                    raise LLMError(f"Grok API non-retryable error: {exc}") from exc

        raise LLMError(f"Grok provider failed: {last_exception}")
