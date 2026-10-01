"""LLM integration module providing a unified provider layer and canonical tool schema."""

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent.providers.anthropic_provider import AnthropicProvider
from agent.providers.base import Decision, LLMError, LLMProvider
from agent.providers.gemini_provider import GeminiProvider
from agent.providers.grok_provider import GrokProvider

load_dotenv()

logger = logging.getLogger(__name__)

# Editable pricing dictionary per million tokens in USD
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-5-haiku-latest": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
    # Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    # Grok
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
6. Call finish(success=true) only when the goal is truly complete and visible on screen."""

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
    elif normalized in ("grok", "xai"):
        return GrokProvider(
            model_name=model_name,
            api_key=api_key,
            client=client,
            max_retries=max_retries,
        )
    else:
        raise LLMError(
            f"Unsupported provider '{provider_name}'. Supported providers: 'anthropic', 'gemini', 'grok'."
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

    resolved_model = model_name or os.getenv("MODEL_NAME")
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


__all__ = [
    "Decision",
    "LLMError",
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GrokProvider",
    "MODEL_PRICING",
    "SYSTEM_PROMPT",
    "TOOLS",
    "estimate_cost",
    "format_history",
    "get_provider",
    "decide",
]
