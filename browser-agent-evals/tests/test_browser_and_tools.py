"""Unit tests for browser observation filtering, select_option tool, Groq 400 tool_use_failed retry, and prompt additions."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from agent.browser import (
    ActionResult,
    BrowserSession,
    InteractiveElement,
    Observation,
    to_prompt_text,
)
from agent.guardrails import StepLimiter, check_action, compute_page_state_hash
from agent.llm import SYSTEM_PROMPT, TOOLS, Decision
from agent.providers.base import LLMConfigError, LLMError
from agent.providers.groq_provider import GroqProvider


class TestSystemPromptAndTools:
    """Tests for system prompt additions and tool schemas."""

    def test_system_prompt_includes_all_required_rules(self):
        assert "The observation includes Page text; read error messages from there." in SYSTEM_PROMPT
        assert "Use select_option for dropdowns, never click." in SYSTEM_PROMPT
        assert (
            "All interactive elements are already listed. Do not scroll unless an expected element is missing. "
            "For the cart, use the link labelled cart; for checkout, use the checkout button."
        ) in SYSTEM_PROMPT
        assert (
            "Use ask_user ONLY before an irreversible action (placing an order, payment, sending). "
            "Never ask the user for credentials or details already given in the goal."
        ) in SYSTEM_PROMPT
        assert (
            "If the goal is to report something (e.g. an error message), "
            "call finish(success=true) with that exact text in summary."
        ) in SYSTEM_PROMPT
        assert (
            "Never call ask_user on the first step or before you have looked at the page. "
            "Use ask_user only right before an irreversible action."
        ) in SYSTEM_PROMPT

    def test_select_option_tool_definition(self):
        select_tool = next((t for t in TOOLS if t["name"] == "select_option"), None)
        assert select_tool is not None
        schema = select_tool["input_schema"]
        assert "id" in schema["properties"]
        assert "value_or_label" in schema["properties"]
        assert "reasoning" in schema["properties"]
        assert set(schema["required"]) == {"id", "value_or_label", "reasoning"}

    def test_select_option_guardrail_safe(self):
        verdict = check_action("select_option", element="Option 1")
        assert verdict.allowed
        assert "select option action allowed" in verdict.reason.lower()


class TestObservationFormatting:
    """Tests for observation to_prompt_text formatting with page text, scroll info, and select options."""

    def test_to_prompt_text_includes_scroll_and_page_text(self):
        obs = Observation(
            url="https://www.saucedemo.com/inventory.html",
            title="Swag Labs",
            elements=[
                InteractiveElement(
                    id=1,
                    tag="select",
                    text="Name (A to Z)",
                    value="az",
                    options=["Name (A to Z)", "Name (Z to A)", "Price (low to high)", "Price (high to low)"],
                ),
                InteractiveElement(
                    id=2,
                    tag="button",
                    text="Add to cart (Sauce Labs Backpack)",
                ),
            ],
            page_text="Products | Sauce Labs Backpack | $29.99",
            scroll_info="Scroll: 0% of page",
        )
        prompt_text = to_prompt_text(obs)
        assert "URL: https://www.saucedemo.com/inventory.html" in prompt_text
        assert "Title: Swag Labs" in prompt_text
        assert "Scroll: 0% of page" in prompt_text
        assert "Page text: Products | Sauce Labs Backpack | $29.99" in prompt_text
        assert "[1] select 'Name (A to Z)' value='az' options=['Name (A to Z)', 'Name (Z to A)', 'Price (low to high)', 'Price (high to low)']" in prompt_text
        assert "[2] button 'Add to cart (Sauce Labs Backpack)'" in prompt_text


class TestBrowserSessionSelectOption:
    """Tests for BrowserSession.select_option execution."""

    def test_select_option_tries_value_then_label(self):
        session = BrowserSession(headless=True)
        mock_page = MagicMock()
        session._page = mock_page

        mock_locator = MagicMock()
        mock_page.locator.return_value = mock_locator
        mock_locator.count.return_value = 1

        # Success on first attempt with value
        res = session.select_option(id=3, value_or_label="lohi")
        assert res.ok
        mock_locator.first.select_option.assert_called_with(value="lohi", timeout=3000)

        # Fallback to label if value fails
        mock_locator.first.select_option.reset_mock()
        mock_locator.first.select_option.side_effect = [Exception("No such value"), None]
        res2 = session.select_option(id=3, value_or_label="Price (low to high)")
        assert res2.ok
        assert mock_locator.first.select_option.call_count == 2
        mock_locator.first.select_option.assert_called_with(label="Price (low to high)", timeout=3000)

    def test_select_option_element_not_found(self):
        session = BrowserSession(headless=True)
        mock_page = MagicMock()
        session._page = mock_page

        mock_locator = MagicMock()
        mock_page.locator.return_value = mock_locator
        mock_locator.count.return_value = 0

        res = session.select_option(id=99, value_or_label="Option 1")
        assert not res.ok
        assert "not found" in res.message


class TestGroqToolUseFailedRetry:
    """Tests for Groq provider 400 error retry on tool_use_failed."""

    def test_tool_use_failed_retries_and_succeeds(self):
        provider = GroqProvider(model_name="openai/gpt-oss-120b", api_key="dummy_key", client=MagicMock())

        # Create a mock 400 error with tool_use_failed code
        err = Exception("Failed tool use")
        err.status_code = 400
        err.code = "tool_use_failed"
        err.body = {"code": "tool_use_failed", "message": "Failed to call tool"}

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_tc = MagicMock()
        mock_tc.function.name = "finish"
        mock_tc.function.arguments = '{"success": true, "summary": "done", "reasoning": "ok"}'
        mock_message.tool_calls = [mock_tc]
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20

        # Attempt 1 fails with tool_use_failed, Attempt 2 succeeds
        provider.client.chat.completions.create.side_effect = [err, mock_response]

        decision = provider.decide(
            system="system prompt",
            messages=[{"role": "user", "content": "goal"}],
            tools=TOOLS,
        )
        assert decision.tool_name == "finish"
        assert decision.tool_args["success"] is True
        assert provider.client.chat.completions.create.call_count == 2

    def test_tool_use_failed_raises_llm_error_after_2_retries(self):
        provider = GroqProvider(model_name="openai/gpt-oss-120b", api_key="dummy_key", client=MagicMock())

        err = Exception("Failed tool call: tool_use_failed")
        err.status_code = 400
        err.code = "tool_use_failed"
        err.body = {"code": "tool_use_failed"}

        # All attempts fail with tool_use_failed
        provider.client.chat.completions.create.side_effect = err

        with pytest.raises(LLMError) as exc_info:
            provider.decide(
                system="system prompt",
                messages=[{"role": "user", "content": "goal"}],
                tools=TOOLS,
            )

        # Total calls = 1 initial + 2 retries = 3 calls
        assert provider.client.chat.completions.create.call_count == 3
        assert "tool_use_failed error after 2 retries" in str(exc_info.value)

    def test_regular_400_raises_llm_config_error_immediately(self):
        provider = GroqProvider(model_name="openai/gpt-oss-120b", api_key="dummy_key", client=MagicMock())

        err = Exception("Invalid API key or model")
        err.status_code = 400
        err.code = "invalid_request_error"
        err.body = {"code": "invalid_request_error"}

        provider.client.chat.completions.create.side_effect = err

        with pytest.raises(LLMConfigError):
            provider.decide(
                system="system prompt",
                messages=[{"role": "user", "content": "goal"}],
                tools=TOOLS,
            )
        assert provider.client.chat.completions.create.call_count == 1


class TestBrowserGoto:
    """Tests for BrowserSession.goto retry on timeout with wait_until=domcontentloaded and 45s timeout."""

    def test_goto_uses_domcontentloaded_and_45s_timeout(self):
        session = BrowserSession(headless=True)
        mock_page = MagicMock()
        session._page = mock_page

        res = session.goto("https://example.com")

        assert res.ok is True
        assert "Navigated to https://example.com" in res.message
        mock_page.goto.assert_called_once_with(
            "https://example.com",
            timeout=45000,
            wait_until="domcontentloaded",
        )

    def test_goto_retries_once_on_timeout_and_succeeds(self):
        session = BrowserSession(headless=True)
        mock_page = MagicMock()
        session._page = mock_page

        # Attempt 1: Timeout error, Attempt 2: Success
        mock_page.goto.side_effect = [
            Exception("Page.goto: Timeout 45000ms exceeded."),
            None,
        ]

        res = session.goto("https://example.com")

        assert res.ok is True
        assert "Navigated to https://example.com" in res.message
        assert mock_page.goto.call_count == 2
        mock_page.goto.assert_called_with(
            "https://example.com",
            timeout=45000,
            wait_until="domcontentloaded",
        )

    def test_goto_fails_after_retry_on_timeout(self):
        session = BrowserSession(headless=True)
        mock_page = MagicMock()
        session._page = mock_page

        # Both attempts time out
        mock_page.goto.side_effect = [
            Exception("Page.goto: Timeout 45000ms exceeded."),
            Exception("Page.goto: Timeout 45000ms exceeded."),
        ]

        res = session.goto("https://example.com")

        assert res.ok is False
        assert "after retry" in res.message
        assert "Timeout 45000ms exceeded" in res.message
        assert mock_page.goto.call_count == 2
