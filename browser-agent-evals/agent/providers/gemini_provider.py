"""Gemini LLM provider implementation using google-genai SDK."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors, types
import httpx

from agent.providers.base import Decision, LLMError, LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini models using the google-genai SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[genai.Client] = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            client=client,
            max_retries=max_retries,
        )
        if self.client is None:
            if not self.api_key:
                raise LLMError(
                    "GEMINI_API_KEY is not set in environment or passed to GeminiProvider."
                )
            self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """Converts canonical tools into google-genai Tool declarations."""
        declarations: List[types.FunctionDeclaration] = []
        for tool in tools:
            schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=schema,
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determines if the exception is due to rate limits, connection issues, or 5xx server errors."""
        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429 or (code is not None and code >= 500):
                return True
            return False
        if isinstance(exc, (httpx.RequestError, ConnectionError, TimeoutError)):
            return True
        return False

    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Invokes Gemini with Function Calling mode ANY and returns a Decision."""
        gemini_tools = self._convert_tools(tools)
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY
            )
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            tool_config=tool_config,
        )

        # Build contents from messages
        prompt_parts: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                prompt_parts.append(str(content))
        contents = "\n\n".join(prompt_parts)

        backoff = 1.0
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "GeminiProvider requesting decision (model=%s, attempt=%d/%d)",
                    self.model_name,
                    attempt,
                    self.max_retries,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                function_calls = getattr(response, "function_calls", None) or []
                if not function_calls:
                    raise LLMError("Gemini model returned no tool call.")

                fc = function_calls[0]
                tool_name = getattr(fc, "name", "")
                raw_args = getattr(fc, "args", {})

                if raw_args is None:
                    tool_args: Dict[str, Any] = {}
                elif isinstance(raw_args, dict):
                    tool_args = dict(raw_args)
                else:
                    try:
                        tool_args = dict(raw_args)
                    except Exception as exc:
                        raise LLMError(
                            f"Malformed arguments in Gemini tool call '{tool_name}': {raw_args}"
                        ) from exc

                tool_args = dict(tool_args)
                reasoning = str(tool_args.pop("reasoning", "") or "")

                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0

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
                        "Gemini retryable error on attempt %d/%d: %s. Retrying in %.1fs...",
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
                            f"Gemini API failed after {self.max_retries} retries: {exc}"
                        ) from exc
                else:
                    logger.error("Non-retryable Gemini API error: %s", exc)
                    raise LLMError(f"Gemini API non-retryable error: {exc}") from exc

        raise LLMError(f"Gemini provider failed: {last_exception}")
