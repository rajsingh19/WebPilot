"""Groq LLM provider implementation using OpenAI-compatible SDK."""

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from agent.providers.base import (
    Decision,
    LLMConfigError,
    LLMError,
    LLMInfraError,
    LLMProvider,
)

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider backed by Groq models using the OpenAI-compatible SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        max_retries: int = 3,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
        super().__init__(
            model_name=model_name,
            api_key=resolved_api_key,
            client=client,
            max_retries=max_retries,
        )
        if base_url is None:
            base_url = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        self.base_url = base_url
        if self.client is None:
            if not self.api_key:
                raise LLMError(
                    "GROQ_API_KEY is not set in environment or passed to GroqProvider."
                )
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts canonical tools into OpenAI/Groq tool format."""
        groq_tools: List[Dict[str, Any]] = []
        for tool in tools:
            schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
            groq_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
        return groq_tools

    def _get_status_code(self, exc: Exception) -> Optional[int]:
        """Extracts status code if available."""
        if hasattr(exc, "status_code") and isinstance(exc.status_code, int):
            return exc.status_code
        if hasattr(exc, "code") and isinstance(exc.code, int):
            return exc.code
        return None

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determines if the exception is due to rate limits, connection issues, or 5xx server errors."""
        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
        code = self._get_status_code(exc)
        if code in (429, 500, 502, 503, 504):
            return True
        if isinstance(exc, (httpx.RequestError, ConnectionError, TimeoutError)):
            return True
        return False

    def _is_tool_use_failed(self, exc: Exception) -> bool:
        """Checks if an error has code 'tool_use_failed'."""
        code = getattr(exc, "code", None)
        if code == "tool_use_failed":
            return True
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            if body.get("code") == "tool_use_failed":
                return True
            err = body.get("error")
            if isinstance(err, dict) and err.get("code") == "tool_use_failed":
                return True
        error_obj = getattr(exc, "error", None)
        if isinstance(error_obj, dict) and error_obj.get("code") == "tool_use_failed":
            return True
        if "tool_use_failed" in str(exc):
            return True
        return False

    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Invokes Groq with tool_choice='required' and returns a Decision."""
        groq_tools = self._convert_tools(tools)
        openai_messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        openai_messages.extend(messages)

        backoff = 1.0
        last_exception: Optional[Exception] = None
        tool_use_retries = 0
        max_tool_use_retries = 2
        attempt = 0

        while True:
            attempt += 1
            try:
                logger.debug(
                    "GroqProvider requesting decision (model=%s, attempt=%d)",
                    self.model_name,
                    attempt,
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    tools=groq_tools,
                    tool_choice="required",
                )

                if not response.choices:
                    raise LLMError("Groq returned empty response choices.")

                choice = response.choices[0]
                message = choice.message

                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    raise LLMError("Groq model returned no tool call.")

                first_tc = tool_calls[0]
                tool_name = first_tc.function.name
                raw_args = first_tc.function.arguments

                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            f"Malformed arguments in Groq tool call '{tool_name}': {raw_args}"
                        ) from exc
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                elif raw_args is None:
                    parsed_args = {}
                else:
                    raise LLMError(
                        f"Malformed arguments in Groq tool call '{tool_name}': {raw_args}"
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

            except (LLMConfigError, LLMInfraError):
                raise
            except LLMError:
                raise
            except Exception as exc:
                code = self._get_status_code(exc)

                if self._is_tool_use_failed(exc):
                    if tool_use_retries < max_tool_use_retries:
                        tool_use_retries += 1
                        logger.warning(
                            "Groq tool_use_failed 400 error (retry %d/%d). Retrying...",
                            tool_use_retries,
                            max_tool_use_retries,
                        )
                        time.sleep(0.5)
                        continue
                    else:
                        logger.error(
                            "Groq tool_use_failed exceeded max retries (%d): %s",
                            max_tool_use_retries,
                            exc,
                        )
                        raise LLMError(
                            f"Groq tool_use_failed error after {max_tool_use_retries} retries: {exc}"
                        ) from exc

                if code in (400, 401, 403):
                    logger.error("Non-retryable Groq configuration error (%s): %s", code, exc)
                    raise LLMConfigError(f"Groq configuration error ({code}): {exc}") from exc

                if self._is_retryable_error(exc):
                    last_exception = exc
                    if attempt >= self.max_retries:
                        raise LLMInfraError(
                            f"Groq API failed after {self.max_retries} retries: {exc}",
                            status_code=code,
                            is_retryable=True,
                        ) from exc
                    jitter = random.uniform(0.1, 0.5 * backoff)
                    sleep_time = backoff + jitter
                    logger.warning(
                        "Groq retryable error on attempt %d/%d (code=%s): %s. Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        code or "network",
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    logger.error("Non-retryable Groq API error: %s", exc)
                    raise LLMError(f"Groq API non-retryable error: {exc}") from exc
