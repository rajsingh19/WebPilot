"""LLM integration module providing a unified provider layer and canonical tool schema."""

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent.providers.anthropic_provider import AnthropicProvider
from agent.providers.base import (
    Decision,
    LLMConfigError,
    LLMError,
    LLMInfraError,
    LLMProvider,
)
from agent.providers.gemini_provider import GeminiProvider
from agent.providers.groq_provider import GroqProvider

# Backward compatibility alias
GrokProvider = GroqProvider

load_dotenv()

logger = logging.getLogger(__name__)

# Editable pricing dictionary per million tokens in USD
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-sonnet-5-5": {"input": 3.0, "output": 15.0},  # verify on Anthropic pricing page
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-5-haiku-latest": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
    # Gemini
    "gemini-3.8-flash": {"input": 0.10, "output": 0.40},  # verify on pricing page
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    # Groq
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},  # verify on pricing page
    "grok-beta": {"input": 5.0, "output": 15.0},
    "grok-2-1212": {"input": 2.0, "output": 10.0},
    "grok-2-vision-1212": {"input": 2.0, "output": 10.0},
    # Fallback
    "default": {"input": 3.0, "output": 15.0},
}

SYSTEM_PROMPT = """You control a web browser to achieve the user's goal. Follow these strict rules:
1. You control a web browser to achieve the user's goal. The goal may be in English, Hindi, or Hinglish.
2. Use only element IDs from the latest observation. Never invent IDs.
3. Dismiss popups/cookie banners first if they block the page.
4. NEVER perform irreversible actions (place order, pay, send, delete, submit final form) yourself.
   Instead call ask_user with exactly what will happen (recipient, text, amount, item).
5. If the same action fails twice, try a different approach. If stuck, call finish(success=false) with the reason.
6. Call finish(success=true) only when the goal is truly complete and visible on screen.
7. The observation includes Page text; read error messages from there.
8. Use select_option for dropdowns, never click.
9. All interactive elements are already listed. Do not scroll unless an expected element is missing. For the cart, use the link labelled cart; for checkout, use the checkout button.
10. Use ask_user ONLY before an irreversible action (placing an order, payment, sending). Never ask the user for credentials or details already given in the goal.
11. If the goal is to report something (e.g. an error message), call finish(success=true) with that exact text in summary."""

# Canonical tool definition list (JSON schema) with required "reasoning" on every tool
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "click",
        "description": "Click an interactive element by its ID from the current observation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "id": {
                    "type": "integer",
                    "description": "The sequential element ID from the latest observation.",
                },
            },
            "required": ["id", "reasoning"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input element and optionally press Enter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "id": {
                    "type": "integer",
                    "description": "The element ID to type into.",
                },
                "text": {
                    "type": "string",
                    "description": "The exact text to type into the element.",
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing text.",
                },
            },
            "required": ["id", "text", "reasoning"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option from a dropdown element by value or label.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "id": {
                    "type": "integer",
                    "description": "The sequential element ID of the select element.",
                },
                "value_or_label": {
                    "type": "string",
                    "description": "The value or visible label of the option to select.",
                },
            },
            "required": ["id", "value_or_label", "reasoning"],
        },
    },
    {
        "name": "goto",
        "description": "Navigate directly to a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "url": {
                    "type": "string",
                    "description": "The target URL to navigate to.",
                },
            },
            "required": ["url", "reasoning"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "The scroll direction: 'up' or 'down'.",
                },
            },
            "required": ["direction", "reasoning"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back to the previous page in history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for a specified number of seconds (1 to 5).",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Seconds to wait (1 to 5).",
                },
            },
            "required": ["seconds", "reasoning"],
        },
    },
    {
        "name": "ask_user",
        "description": "Ask the user for clarification or confirmation before executing an irreversible action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "question": {
                    "type": "string",
                    "description": "The specific question or confirmation message for the user.",
                },
            },
            "required": ["question", "reasoning"],
        },
    },
    {
        "name": "finish",
        "description": "Complete the run, indicating whether the goal succeeded or failed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action is taken.",
                },
                "success": {
                    "type": "boolean",
                    "description": "True if the goal is verified complete; False if stuck or failed.",
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of the outcome or reason for failure.",
                },
            },
            "required": ["success", "summary", "reasoning"],
        },
    },
]


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Estimates USD cost based on token counts and model pricing."""
    pricing_map = pricing or MODEL_PRICING
    rates = pricing_map.get(model_name, pricing_map.get("default", {"input": 3.0, "output": 15.0}))
    input_cost = (input_tokens / 1_000_000.0) * rates.get("input", 0.0)
    output_cost = (output_tokens / 1_000_000.0) * rates.get("output", 0.0)
    return input_cost + output_cost


def format_history(history: Optional[List[Any]]) -> str:
    """Formats the last 6 steps into compact text representation."""
    if not history:
        return "No previous actions taken."

    recent = history[-6:]
    lines = []
    for idx, item in enumerate(recent, 1):
        if isinstance(item, str):
            lines.append(f"{idx}. {item}")
        elif isinstance(item, dict):
            action = item.get("action") or item.get("tool_name", "action")
            args = item.get("args") or item.get("tool_args", {})
            result = item.get("result") or item.get("message", "")
            lines.append(f"{idx}. Action: {action}({args}) -> Result: {result}")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            lines.append(f"{idx}. Action: {item[0]} -> Result: {item[1]}")
        else:
            lines.append(f"{idx}. {str(item)}")

    return "\n".join(lines)


def get_provider(
    provider_name: str,
    model_name: str,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    max_retries: int = 3,
) -> LLMProvider:
    """Factory creating an LLMProvider instance based on provider name."""
    normalized = provider_name.strip().lower()
    if normalized == "anthropic":
        return AnthropicProvider(
            model_name=model_name,
            api_key=api_key,
            client=client,
            max_retries=max_retries,
        )
    elif normalized in ("gemini", "google"):
        return GeminiProvider(
            model_name=model_name,
            api_key=api_key,
            client=client,
            max_retries=max_retries,
        )
    elif normalized in ("groq", "grok", "xai"):
        return GroqProvider(
            model_name=model_name,
            api_key=api_key,
            client=client,
            max_retries=max_retries,
        )
    else:
        raise LLMError(
            f"Unsupported provider '{provider_name}'. Supported providers: 'anthropic', 'gemini', 'groq'."
        )


def decide(
    goal: str,
    history: Optional[List[Any]],
    observation_text: str,
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    provider: Optional[str] = None,
) -> Decision:
    """Dispatches to the configured provider to select the next browser action."""
    resolved_provider_name = (
        provider
        or os.getenv("PROVIDER")
        or "anthropic"
    ).strip().lower()

    resolved_model = model_name or os.getenv("MODEL_NAME", "claude-sonnet-5-5")
    if not resolved_model:
        raise LLMError(
            "MODEL_NAME is not set. Please specify MODEL_NAME in environment (.env) or pass it to decide()."
        )

    provider_instance = get_provider(
        provider_name=resolved_provider_name,
        model_name=resolved_model,
        api_key=api_key,
        client=client,
        max_retries=max_retries,
    )

    formatted_history = format_history(history)
    user_prompt = f"""Goal: {goal}

Recent History (last 6 steps):
{formatted_history}

Current Page Observation:
{observation_text}
"""
    messages = [{"role": "user", "content": user_prompt}]

    return provider_instance.decide(
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=TOOLS,
    )



# Provider exhaustion tracking (in-memory)
EXHAUSTED_PROVIDERS: Dict[str, float] = {}  # normalized provider name -> expiry timestamp


def normalize_provider_name(provider_name: str) -> str:
    """Normalizes provider aliases to canonical names ('groq', 'gemini', 'anthropic')."""
    p = (provider_name or "").strip().lower()
    if p in ("groq", "grok", "xai"):
        return "groq"
    if p in ("gemini", "google"):
        return "gemini"
    if p in ("anthropic", "claude"):
        return "anthropic"
    return p


def get_model_for_provider(provider_name: str, fallback_model: Optional[str] = None) -> str:
    """Returns the configured model for a provider from environment or defaults."""
    norm = normalize_provider_name(provider_name)
    if norm == "groq":
        return (
            os.getenv("MODEL_NAME_GROK")
            or os.getenv("MODEL_NAME_GROQ")
            or fallback_model
            or os.getenv("MODEL_NAME")
            or "openai/gpt-oss-120b"
        )
    elif norm == "gemini":
        return (
            os.getenv("MODEL_NAME_GEMINI")
            or os.getenv("MODEL_NAME_GOOGLE")
            or fallback_model
            or os.getenv("MODEL_NAME")
            or "gemini-2.0-flash"
        )
    elif norm == "anthropic":
        return (
            os.getenv("MODEL_NAME_ANTHROPIC")
            or fallback_model
            or os.getenv("MODEL_NAME")
            or "claude-sonnet-5-5"
        )
    return fallback_model or os.getenv("MODEL_NAME") or "default"


def is_provider_exhausted(provider_name: str) -> bool:
    """Returns True if provider is currently in exhaustion cooldown, False otherwise."""
    norm = normalize_provider_name(provider_name)
    expiry = EXHAUSTED_PROVIDERS.get(norm)
    if expiry is None:
        return False
    if time.time() >= expiry:
        EXHAUSTED_PROVIDERS.pop(norm, None)
        return False
    return True


def mark_provider_exhausted(provider_name: str, cooldown_seconds: Optional[float] = None) -> float:
    """Marks a provider as exhausted for cooldown_seconds (default 10 min = 600s)."""
    norm = normalize_provider_name(provider_name)
    if cooldown_seconds is None:
        cooldown_seconds = float(os.getenv("PROVIDER_COOLDOWN_SECONDS", "600"))
    expiry = time.time() + cooldown_seconds
    EXHAUSTED_PROVIDERS[norm] = expiry
    logger.warning("Provider '%s' marked exhausted for %.1fs (cooldown until %s).", norm, cooldown_seconds, expiry)
    return expiry


def reset_provider_exhaustion(provider_name: Optional[str] = None) -> None:
    """Resets exhaustion state for a specific provider or all providers."""
    if provider_name:
        norm = normalize_provider_name(provider_name)
        EXHAUSTED_PROVIDERS.pop(norm, None)
    else:
        EXHAUSTED_PROVIDERS.clear()


def get_provider_priority() -> List[str]:
    """Returns ordered list of canonical provider names from PROVIDER_PRIORITY."""
    raw = os.getenv("PROVIDER_PRIORITY", "grok,gemini")
    return [normalize_provider_name(p.strip()) for p in raw.split(",") if p.strip()]


def is_failover_enabled(default: bool = True) -> bool:
    """Checks if FAILOVER_ENABLED is set in environment."""
    raw = os.getenv("FAILOVER_ENABLED")
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def pick_active_provider(
    priority: Optional[List[str]] = None,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Selects the first non-exhausted provider in priority order.

    Returns (provider, model_name) or (None, None) if all are exhausted.
    """
    priority_list = priority if priority is not None else get_provider_priority()
    if preferred_provider:
        norm_pref = normalize_provider_name(preferred_provider)
        if not is_provider_exhausted(norm_pref):
            model = preferred_model or get_model_for_provider(norm_pref)
            return norm_pref, model

    for prov in priority_list:
        norm_prov = normalize_provider_name(prov)
        if not is_provider_exhausted(norm_prov):
            model = get_model_for_provider(norm_prov)
            return norm_prov, model

    return None, None


def is_rate_limit_error(exc: Exception) -> bool:
    """Determines whether an exception corresponds to a 429, quota, or rate limit error."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, "429", "rate_limit_exceeded"):
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("code") in ("rate_limit_exceeded", "resource_exhausted") or body.get("type") == "tokens":
            return True
        err = body.get("error")
        if isinstance(err, dict):
            if err.get("code") in ("rate_limit_exceeded", "resource_exhausted") or err.get("type") == "tokens":
                return True

    msg = str(exc).lower()
    return any(phrase in msg for phrase in (
        "429",
        "rate limit",
        "rate_limit",
        "quota",
        "tokens per day",
        "tokens per minute",
        "requests per minute",
        "tpd",
        "rpm",
        "tpm",
        "resource_exhausted",
        "exceeded your current quota",
    ))


__all__ = [
    "Decision",
    "LLMError",
    "LLMConfigError",
    "LLMInfraError",
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GroqProvider",
    "GrokProvider",
    "MODEL_PRICING",
    "SYSTEM_PROMPT",
    "TOOLS",
    "estimate_cost",
    "format_history",
    "get_provider",
    "decide",
    "normalize_provider_name",
    "get_model_for_provider",
    "is_provider_exhausted",
    "mark_provider_exhausted",
    "reset_provider_exhaustion",
    "get_provider_priority",
    "is_failover_enabled",
    "pick_active_provider",
    "is_rate_limit_error",
]
