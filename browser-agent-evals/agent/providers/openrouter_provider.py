"""OpenRouter LLM provider implementation using OpenAI-compatible SDK."""

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


class OpenRouterProvider(LLMProvider):
    """LLM provider backed by OpenRouter using the OpenAI-compatible SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        max_retries: int = 4,
        base_url: Optional[str] = None,
        provider_order: Optional[List[str]] = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(
            model_name=model_name,
            api_key=resolved_api_key,
            client=client,
            max_retries=max_retries,
        )
        if base_url is None:
            base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.base_url = base_url
        self.provider_order = provider_order
        if self.client is None:
            if not self.api_key:
                raise LLMError(
                    "OPENROUTER_API_KEY is not set in environment or passed to OpenRouterProvider."
                )
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts canonical tools into OpenAI tool format."""
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

    def _extract_upstream_provider(self, response: Any) -> Optional[str]:
        """Extracts OpenRouter upstream provider name from response."""
        if response is None:
            return None
        prov = getattr(response, "provider", None)
        if prov is not None and not type(prov).__name__.startswith("MagicMock"):
            return str(prov)
        model_extra = getattr(response, "model_extra", None)
        if isinstance(model_extra, dict) and "provider" in model_extra:
            extra_prov = model_extra.get("provider")
            if extra_prov is not None and not type(extra_prov).__name__.startswith("MagicMock"):
                return str(extra_prov)
        return None

    def decide(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Decision:
        """Invokes OpenRouter with tool_choice='required' and returns a Decision."""
        openai_tools = self._convert_tools(tools)
        openai_messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        openai_messages.extend(messages)

        backoff = 1.0
        no_tool_retries = 0
        max_no_tool_retries = 2
        attempt = 0
        ignored_upstreams: List[str] = []

        while True:
            attempt += 1

            provider_config: Dict[str, Any] = {
                "require_parameters": True,
                "allow_fallbacks": True,
                "sort": "throughput",
            }
            order_list = self.provider_order
            if order_list is None:
                order_env = os.getenv("OPENROUTER_PROVIDER_ORDER", "").strip()
                if order_env:
                    order_list = [p.strip() for p in order_env.split(",") if p.strip()]
                else:
                    order_list = []
            if order_list:
                provider_config["order"] = list(order_list)

            if ignored_upstreams:
                provider_config["ignore"] = list(ignored_upstreams)

            extra_body = {"provider": provider_config}

            try:
                logger.debug(
                    "OpenRouterProvider requesting decision (model=%s, attempt=%d, ignored=%s)",
                    self.model_name,
                    attempt,
                    ignored_upstreams,
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    tools=openai_tools,
                    tool_choice="required",
                    extra_body=extra_body,
                )

                upstream = self._extract_upstream_provider(response)

                choice = response.choices[0] if (response and response.choices) else None
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                message = getattr(choice, "message", None) if choice else None
                content = getattr(message, "content", None) if message else None
                tool_calls = getattr(message, "tool_calls", None) or []

                is_empty_content = (
                    content is None
                    or (isinstance(content, str) and not content.strip())
                )

                # Transient provider error: finish_reason == "error", empty content with no tools, or empty choices
                is_transient_error = (
                    (not response or not response.choices)
                    or finish_reason == "error"
                    or (is_empty_content and not tool_calls)
                )

                if is_transient_error:
                    if upstream and upstream not in ignored_upstreams:
                        ignored_upstreams.append(upstream)

                    if attempt >= self.max_retries:
                        err_msg = (
                            f"OpenRouter upstream provider error after {self.max_retries} attempts "
                            f"(finish_reason='{finish_reason}', upstream='{upstream}')."
                        )
                        logger.error(err_msg)
                        raise LLMError(err_msg)

                    jitter = random.uniform(0.1, 0.5 * backoff)
                    sleep_time = backoff + jitter
                    logger.warning(
                        "OpenRouter transient provider error on attempt %d/%d (upstream=%s, finish_reason=%s). Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        upstream or "unknown",
                        finish_reason,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                if not tool_calls:
                    if no_tool_retries < max_no_tool_retries:
                        no_tool_retries += 1
                        logger.warning(
                            "OpenRouter model did not call a tool (retry %d/%d). Retrying...",
                            no_tool_retries,
                            max_no_tool_retries,
                        )
                        time.sleep(0.5)
                        continue
                    raise LLMError(f"OpenRouter model did not call a tool after {max_no_tool_retries} retries.")

                first_tc = tool_calls[0]
                tool_name = first_tc.function.name
                raw_args = first_tc.function.arguments

                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            f"Malformed arguments in OpenRouter tool call '{tool_name}': {raw_args}"
                        ) from exc
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                elif raw_args is None:
                    parsed_args = {}
                else:
                    raise LLMError(
                        f"Malformed arguments in OpenRouter tool call '{tool_name}': {raw_args}"
                    )

                tool_args = dict(parsed_args)
                reasoning = str(tool_args.pop("reasoning", "") or "")

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                # Extract OpenRouter usage cost if provided in response
                cost_usd: Optional[float] = None
                if usage:
                    cost_val = getattr(usage, "cost", None)
                    if cost_val is None and hasattr(usage, "model_extra") and isinstance(usage.model_extra, dict):
                        cost_val = usage.model_extra.get("cost")
                    if cost_val is not None:
                        try:
                            cost_usd = float(cost_val)
                        except (ValueError, TypeError):
                            cost_usd = None

                return Decision(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    reasoning=reasoning,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    upstream_provider=upstream,
                )

            except (LLMConfigError, LLMInfraError):
                raise
            except LLMError:
                raise
            except Exception as exc:
                code = self._get_status_code(exc)

                if code in (400, 401, 403):
                    logger.error("Non-retryable OpenRouter configuration error (%s): %s", code, exc)
                    raise LLMConfigError(f"OpenRouter configuration error ({code}): {exc}") from exc

                if self._is_retryable_error(exc):
                    if attempt >= self.max_retries:
                        raise LLMInfraError(
                            f"OpenRouter API failed after {self.max_retries} retries: {exc}",
                            status_code=code,
                            is_retryable=True,
                        ) from exc
                    jitter = random.uniform(0.1, 0.5 * backoff)
                    sleep_time = backoff + jitter
                    logger.warning(
                        "OpenRouter retryable error on attempt %d/%d (code=%s): %s. Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        code or "network",
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    logger.error("Non-retryable OpenRouter API error: %s", exc)
                    raise LLMError(f"OpenRouter API non-retryable error: {exc}") from exc
