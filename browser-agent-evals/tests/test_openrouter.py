"""Tests for OpenRouter LLM provider implementation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.llm import TOOLS, get_provider, is_price_unverified, normalize_provider_name
from agent.providers.base import Decision, LLMConfigError, LLMError, LLMInfraError
from agent.providers.openrouter_provider import OpenRouterProvider
from openai import APIConnectionError, RateLimitError


def create_mock_completion(
    tool_name: str = "click",
    tool_args: dict = None,
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    cost: float = None,
    has_tool_call: bool = True,
    finish_reason: str = "stop",
    content: str = None,
    provider: str = None,
):
    """Helper creating a mock ChatCompletion response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_choice.finish_reason = finish_reason
    if content is not None:
        mock_message.content = content

    if has_tool_call:
        args_dict = tool_args or {"id": 1, "reasoning": "Clicking item"}
        mock_tc = MagicMock()
        mock_tc.function.name = tool_name
        mock_tc.function.arguments = json.dumps(args_dict)
        mock_message.tool_calls = [mock_tc]
    else:
        mock_message.tool_calls = []

    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    if cost is not None:
        mock_response.usage.cost = cost
    else:
        del mock_response.usage.cost

    if provider is not None:
        mock_response.provider = provider
    else:
        mock_response.provider = None

    return mock_response


class TestOpenRouterProvider:
    """Test suite for OpenRouterProvider."""

    def test_openrouter_init_with_key_and_custom_client(self):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-or-dummy",
            client=mock_client,
        )
        assert provider.model_name == "openai/gpt-oss-120b"
        assert provider.api_key == "sk-or-dummy"
        assert provider.client is mock_client
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_init_missing_key_raises_llm_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(LLMError) as exc_info:
            OpenRouterProvider(model_name="openai/gpt-oss-120b")
        assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)

    def test_decide_success_with_tool_call_and_cost(self):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
        )

        mock_resp = create_mock_completion(
            tool_name="click",
            tool_args={"id": 4, "reasoning": "navigate to cart"},
            prompt_tokens=100,
            completion_tokens=25,
            cost=0.00015,
        )
        mock_client.chat.completions.create.return_value = mock_resp

        decision = provider.decide(
            system="System instruction",
            messages=[{"role": "user", "content": "Click the cart"}],
            tools=TOOLS,
        )

        assert isinstance(decision, Decision)
        assert decision.tool_name == "click"
        assert decision.tool_args == {"id": 4}
        assert decision.reasoning == "navigate to cart"
        assert decision.input_tokens == 100
        assert decision.output_tokens == 25
        assert decision.cost_usd == 0.00015

        # Verify tool_choice="required" was used
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"
        assert call_kwargs["model"] == "openai/gpt-oss-120b"

    @patch("time.sleep", return_value=None)
    def test_decide_no_tool_call_retries_and_succeeds(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
        )

        no_tool_resp = create_mock_completion(has_tool_call=False)
        valid_resp = create_mock_completion(
            tool_name="finish",
            tool_args={"success": True, "summary": "Done", "reasoning": "Goal met"},
        )
        mock_client.chat.completions.create.side_effect = [no_tool_resp, valid_resp]

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Done"}],
            tools=TOOLS,
        )

        assert decision.tool_name == "finish"
        assert decision.tool_args["success"] is True
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_decide_no_tool_call_fails_after_2_retries(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
        )

        no_tool_resp = create_mock_completion(has_tool_call=False)
        # Always return no tool calls
        mock_client.chat.completions.create.return_value = no_tool_resp

        with pytest.raises(LLMError) as exc_info:
            provider.decide(
                system="System",
                messages=[{"role": "user", "content": "Do something"}],
                tools=TOOLS,
            )

        # 1 initial + 2 retries = 3 calls
        assert mock_client.chat.completions.create.call_count == 3
        assert "did not call a tool after 2 retries" in str(exc_info.value)

    @patch("time.sleep", return_value=None)
    def test_decide_429_retries_and_succeeds(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_429 = Exception("Rate limit reached")
        err_429.status_code = 429
        valid_resp = create_mock_completion(
            tool_name="type_text",
            tool_args={"id": 2, "text": "standard_user", "press_enter": False, "reasoning": "Type user"},
        )
        mock_client.chat.completions.create.side_effect = [err_429, valid_resp]

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Login"}],
            tools=TOOLS,
        )

        assert decision.tool_name == "type_text"
        assert decision.tool_args["text"] == "standard_user"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_decide_429_exhausts_retries_and_raises_infra_error(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_429 = Exception("Rate limit exceeded")
        err_429.status_code = 429
        mock_client.chat.completions.create.side_effect = err_429

        with pytest.raises(LLMInfraError) as exc_info:
            provider.decide(
                system="System",
                messages=[{"role": "user", "content": "Login"}],
                tools=TOOLS,
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.is_retryable is True
        assert mock_client.chat.completions.create.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("time.sleep", return_value=None)
    def test_decide_503_retries_and_succeeds(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_503 = Exception("Service Unavailable")
        err_503.status_code = 503
        valid_resp = create_mock_completion(
            tool_name="click",
            tool_args={"id": 1, "reasoning": "retry after 503"},
        )
        mock_client.chat.completions.create.side_effect = [err_503, valid_resp]

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Action"}],
            tools=TOOLS,
        )

        assert decision.tool_name == "click"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_decide_503_exhausts_retries_and_raises_infra_error(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_503 = Exception("Service Unavailable")
        err_503.status_code = 503
        mock_client.chat.completions.create.side_effect = err_503

        with pytest.raises(LLMInfraError) as exc_info:
            provider.decide(
                system="System",
                messages=[{"role": "user", "content": "Action"}],
                tools=TOOLS,
            )

        assert exc_info.value.status_code == 503
        assert exc_info.value.is_retryable is True
        assert mock_client.chat.completions.create.call_count == 4
        assert mock_sleep.call_count == 3

    def test_decide_401_raises_config_error_immediately(self):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_401 = Exception("Invalid API Key")
        err_401.status_code = 401
        mock_client.chat.completions.create.side_effect = err_401

        with pytest.raises(LLMConfigError):
            provider.decide(
                system="System",
                messages=[{"role": "user", "content": "Action"}],
                tools=TOOLS,
            )

        assert mock_client.chat.completions.create.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_decide_finish_reason_error_retries_with_ignore_and_succeeds(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        # Attempt 1: CoreWeave returns finish_reason == "error"
        err_resp = create_mock_completion(
            has_tool_call=False,
            finish_reason="error",
            content="",
            provider="CoreWeave",
        )
        # Attempt 2: Novita returns valid tool call
        ok_resp = create_mock_completion(
            tool_name="click",
            tool_args={"id": 2, "reasoning": "Click button"},
            finish_reason="stop",
            provider="Novita",
        )
        mock_client.chat.completions.create.side_effect = [err_resp, ok_resp]

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Click"}],
            tools=TOOLS,
        )

        assert decision.tool_name == "click"
        assert decision.upstream_provider == "Novita"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

        first_call_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs

        assert "ignore" not in first_call_kwargs["extra_body"]["provider"]
        assert second_call_kwargs["extra_body"]["provider"]["ignore"] == ["CoreWeave"]
        assert second_call_kwargs["extra_body"]["provider"]["require_parameters"] is True
        assert second_call_kwargs["extra_body"]["provider"]["allow_fallbacks"] is True
        assert second_call_kwargs["extra_body"]["provider"]["sort"] == "throughput"

    @patch("time.sleep", return_value=None)
    def test_decide_finish_reason_error_exhausts_retries_raises_llm_error(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        err_resp = create_mock_completion(
            has_tool_call=False,
            finish_reason="error",
            content="",
            provider="CoreWeave",
        )
        mock_client.chat.completions.create.return_value = err_resp

        with pytest.raises(LLMError) as exc_info:
            provider.decide(
                system="System",
                messages=[{"role": "user", "content": "Action"}],
                tools=TOOLS,
            )

        assert "OpenRouter upstream provider error after 4 attempts" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("time.sleep", return_value=None)
    def test_decide_empty_content_and_no_tool_calls_treated_as_transient_error(self, mock_sleep):
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
            max_retries=4,
        )

        # Attempt 1: empty content, no tool calls, finish_reason "stop"
        empty_resp = create_mock_completion(
            has_tool_call=False,
            finish_reason="stop",
            content="",
            provider="CoreWeave",
        )
        # Attempt 2: success
        ok_resp = create_mock_completion(
            tool_name="finish",
            tool_args={"success": True, "reasoning": "Done"},
            provider="SambaNova",
        )
        mock_client.chat.completions.create.side_effect = [empty_resp, ok_resp]

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Finish"}],
            tools=TOOLS,
        )

        assert decision.tool_name == "finish"
        assert decision.upstream_provider == "SambaNova"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert second_call_kwargs["extra_body"]["provider"]["ignore"] == ["CoreWeave"]

    def test_decide_openrouter_provider_order_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "Novita, Together, SambaNova")
        mock_client = MagicMock()
        provider = OpenRouterProvider(
            model_name="openai/gpt-oss-120b",
            api_key="sk-dummy",
            client=mock_client,
        )

        mock_resp = create_mock_completion(provider="Novita")
        mock_client.chat.completions.create.return_value = mock_resp

        decision = provider.decide(
            system="System",
            messages=[{"role": "user", "content": "Action"}],
            tools=TOOLS,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        provider_cfg = call_kwargs["extra_body"]["provider"]
        assert provider_cfg["order"] == ["Novita", "Together", "SambaNova"]
        assert provider_cfg["require_parameters"] is True
        assert provider_cfg["allow_fallbacks"] is True
        assert provider_cfg["sort"] == "throughput"
        assert decision.upstream_provider == "Novita"


class TestOpenRouterRegistrationAndPricing:
    """Test suite for OpenRouter registration, pricing, and factory routing."""

    def test_get_provider_openrouter(self):
        mock_client = MagicMock()
        prov = get_provider(
            provider_name="openrouter",
            model_name="openai/gpt-oss-120b",
            api_key="sk-test",
            client=mock_client,
        )
        assert isinstance(prov, OpenRouterProvider)
        assert prov.model_name == "openai/gpt-oss-120b"

    def test_normalize_provider_name_openrouter(self):
        assert normalize_provider_name("openrouter") == "openrouter"
        assert normalize_provider_name("OpenRouter") == "openrouter"
        assert normalize_provider_name(" OPENROUTER ") == "openrouter"

    def test_pricing_unverified_without_explicit_rate(self):
        # OpenRouter models are unverified unless usage cost returned or rate provided
        assert is_price_unverified("meta-llama/llama-3.3-70b-instruct", "openrouter") is True
        assert is_price_unverified(None, "openrouter") is True


class TestOpenRouterTraceAndReport:
    """Test suite verifying upstream provider recording in traces and evaluation reports."""

    @patch("agent.agent.decide")
    def test_upstream_provider_recorded_in_trace_steps_and_result(self, mock_decide, tmp_path):
        from agent.agent import Agent
        from agent.browser import BrowserSession, InteractiveElement, Observation

        session = MagicMock(spec=BrowserSession)
        session._page = MagicMock()
        session.observe.return_value = Observation(
            url="https://example.com",
            title="Example",
            elements=[InteractiveElement(id=1, tag="button", text="Submit")],
            screenshot_path=None,
        )

        mock_decide.return_value = Decision(
            tool_name="finish",
            tool_args={"success": True, "summary": "Finished successfully."},
            reasoning="Goal accomplished.",
            input_tokens=40,
            output_tokens=15,
            upstream_provider="Novita",
        )

        agent = Agent(
            provider="openrouter",
            browser_session=session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Test goal", start_url="https://example.com")

        assert result.upstream_provider == "Novita"
        trace_file = Path(result.trace_file)
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

        assert trace_data["upstream_provider"] == "Novita"
        assert len(trace_data["steps"]) >= 1
        for step in trace_data["steps"]:
            assert step["upstream_provider"] == "Novita"

    def test_eval_reports_show_upstream_provider(self, tmp_path):
        from evals.run_evals import compute_eval_summary, save_reports

        results = [
            {
                "test_id": "test_login",
                "test_name": "login_flow",
                "category": "functional",
                "passed": True,
                "actual_status": "success",
                "steps": 2,
                "cost_usd": 0.001,
                "duration_s": 3.5,
                "failure_category": "",
                "checker_detail": "Logged in",
                "provider": "openrouter",
                "model": "openai/gpt-oss-120b",
                "upstream_provider": "Novita",
            }
        ]

        summary = compute_eval_summary(results)
        json_file, md_file = save_reports(
            results=results,
            summary=summary,
            provider="openrouter",
            model_name="openai/gpt-oss-120b",
            git_commit="test_commit",
            output_dir=str(tmp_path),
        )

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["results"][0]["upstream_provider"] == "Novita"

        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Upstream Provider" in md_text
        assert "Novita" in md_text
