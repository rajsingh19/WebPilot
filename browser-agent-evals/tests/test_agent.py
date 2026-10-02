"""Unit tests for Agent orchestrator, lifecycle, safety enforcement, and tracing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.agent import Agent, RunResult
from agent.browser import BrowserSession, InteractiveElement, Observation
from agent.llm import Decision, LLMError


@pytest.fixture
def mock_browser_session():
    """Mock BrowserSession providing predictable observation states."""
    session = MagicMock(spec=BrowserSession)
    session._page = MagicMock()
    # Default initial observation
    session.observe.return_value = Observation(
        url="https://example.com",
        title="Example Domain",
        elements=[
            InteractiveElement(id=1, tag="button", text="Click Me"),
            InteractiveElement(id=2, tag="a", text="Pay Now"),
            InteractiveElement(id=3, tag="input", type="text", text="Username"),
        ],
        screenshot_path="/tmp/fake.png",
    )
    session.click.return_value = MagicMock(ok=True, message="Clicked")
    session.type_text.return_value = MagicMock(ok=True, message="Typed")
    session.goto.return_value = MagicMock(ok=True, message="Navigated")
    session.wait.return_value = MagicMock(ok=True, message="Waited")
    return session


class TestAgentExecution:
    """Tests covering agent decision flow, guardrails, and run outcomes."""

    @patch("agent.agent.decide")
    def test_run_finish_success(self, mock_decide, mock_browser_session, tmp_path):
        mock_decide.return_value = Decision(
            tool_name="finish",
            tool_args={"success": True, "summary": "Goal completed successfully."},
            reasoning="Everything is done.",
            input_tokens=100,
            output_tokens=25,
        )

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Test goal", start_url="https://example.com")

        assert result.status == "success"
        assert result.steps == 1
        assert "completed successfully" in result.summary
        assert Path(result.trace_file).exists()

        with open(result.trace_file) as f:
            trace_json = json.load(f)
            assert trace_json["status"] == "success"
            assert trace_json["steps"][0]["action"] == "finish"

    @patch("agent.agent.decide")
    def test_run_ask_user_needs_confirmation(self, mock_decide, mock_browser_session, tmp_path):
        mock_decide.return_value = Decision(
            tool_name="ask_user",
            tool_args={"question": "Should I pay $50 for this order?"},
            reasoning="Action is irreversible.",
            input_tokens=100,
            output_tokens=25,
        )

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        # Without confirm_callback, agent must pause
        result = agent.run(goal="Buy item", start_url="https://example.com")

        assert result.status == "needs_confirmation"
        assert result.question == "Should I pay $50 for this order?"
        assert "paused for user confirmation" in result.summary

    @patch("agent.agent.decide")
    def test_run_ask_user_with_callback_confirmed(self, mock_decide, mock_browser_session, tmp_path):
        mock_decide.side_effect = [
            Decision(
                tool_name="ask_user",
                tool_args={"question": "Confirm order?"},
                reasoning="Ask user first.",
                input_tokens=50,
                output_tokens=10,
            ),
            Decision(
                tool_name="finish",
                tool_args={"success": True, "summary": "Finished after confirmation."},
                reasoning="Proceeded after confirm.",
                input_tokens=50,
                output_tokens=10,
            ),
        ]

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        confirm_cb = MagicMock(return_value=True)
        result = agent.run(
            goal="Buy item",
            start_url="https://example.com",
            confirm_callback=confirm_cb,
        )

        confirm_cb.assert_called_once_with("Confirm order?")
        assert result.status == "success"
        assert result.steps == 2

    @patch("agent.agent.decide")
    def test_run_guardrails_blocked_after_two_attempts(self, mock_decide, mock_browser_session, tmp_path):
        # Element id 2 has label "Pay Now", which is risky
        mock_decide.side_effect = [
            Decision(
                tool_name="click",
                tool_args={"id": 2},
                reasoning="Clicking pay.",
                input_tokens=50,
                output_tokens=10,
            ),
            Decision(
                tool_name="click",
                tool_args={"id": 2},
                reasoning="Retrying pay.",
                input_tokens=50,
                output_tokens=10,
            ),
        ]

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Checkout", start_url="https://example.com")

        assert result.status == "blocked"
        assert "blocked 2 times" in result.summary
        # Browser session click should never have been executed
        mock_browser_session.click.assert_not_called()

    @patch("agent.agent.decide")
    def test_run_hallucinated_element_id_continues(self, mock_decide, mock_browser_session, tmp_path):
        # Element 99 does not exist in observation
        mock_decide.side_effect = [
            Decision(
                tool_name="click",
                tool_args={"id": 99},
                reasoning="Clicking mystery element.",
                input_tokens=40,
                output_tokens=10,
            ),
            Decision(
                tool_name="finish",
                tool_args={"success": False, "summary": "Giving up."},
                reasoning="Element missing.",
                input_tokens=40,
                output_tokens=10,
            ),
        ]

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Test", start_url="https://example.com")

        assert result.status == "failed"
        assert result.steps == 2
        with open(result.trace_file) as f:
            data = json.load(f)
            assert data["hallucinated_ids_count"] == 1

    @patch("agent.agent.decide")
    def test_run_loop_detection(self, mock_decide, mock_browser_session, tmp_path):
        # Repeat identical action 3 times
        repeat_decision = Decision(
            tool_name="click",
            tool_args={"id": 1},
            reasoning="Clicking button.",
            input_tokens=30,
            output_tokens=10,
        )
        mock_decide.side_effect = [repeat_decision, repeat_decision, repeat_decision]

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Looping", start_url="https://example.com")

        assert result.status == "loop"
        assert "loop detected" in result.summary.lower()

    @patch("agent.agent.decide")
    def test_run_llm_error_handled(self, mock_decide, mock_browser_session, tmp_path):
        mock_decide.side_effect = LLMError("API rate limit exhausted.")

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Test error", start_url="https://example.com")

        assert result.status in ("error", "infra_error")
        assert "API rate limit exhausted" in result.summary

    @patch("agent.agent.decide")
    def test_run_fallback_success(self, mock_decide, mock_browser_session, tmp_path):
        # First call (primary) raises LLMError, second call (fallback) succeeds
        mock_decide.side_effect = [
            LLMError("Primary 503 unavailable"),
            Decision(
                tool_name="finish",
                tool_args={"success": True, "summary": "Done via fallback."},
                reasoning="",
                input_tokens=10,
                output_tokens=5,
            ),
        ]
        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            fallback_provider="groq",
            fallback_model_name="openai/gpt-oss-120b",
        )
        result = agent.run(goal="Test fallback", start_url="https://example.com")
        assert result.status == "success"
        assert result.fallback_used is True
        assert result.model_used == "openai/gpt-oss-120b"
        assert "Primary 503" in result.fallback_reason

    def test_browser_session_kept_open_after_run(self, mock_browser_session, tmp_path):
        with patch("agent.agent.decide") as mock_decide:
            mock_decide.return_value = Decision(
                tool_name="finish",
                tool_args={"success": True, "summary": "Done"},
                reasoning="",
                input_tokens=10,
                output_tokens=5,
            )
            agent = Agent(
                browser_session=mock_browser_session,
                traces_root=str(tmp_path),
            )
            agent.run(goal="Test live session", start_url="https://example.com")

            # BrowserSession.close() must NOT have been called automatically
            mock_browser_session.close.assert_not_called()

    @patch("agent.agent.decide")
    def test_run_per_test_max_steps_overrides_default_12(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        mock_browser_session.observe.side_effect = [
            Observation(
                url=f"https://example.com/page_{i}",
                title=f"Page {i}",
                elements=[InteractiveElement(id=i, tag="div", text=f"Item {i}")],
                screenshot_path=None,
            )
            for i in range(10)
        ]
        mock_decide.side_effect = [
            Decision(
                tool_name="scroll",
                tool_args={"direction": "down"},
                reasoning=f"Step {i}",
                input_tokens=10,
                output_tokens=5,
            )
            for i in range(10)
        ]
        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            max_steps=12,
        )
        assert agent.max_steps == 12

        # Override default 12 with max_steps=5
        result = agent.run(goal="Scroll test", start_url="https://example.com", max_steps=5)
        assert result.status == "failed"
        assert result.steps == 5
        assert "5/5 steps executed" in result.summary

    @patch("agent.agent.decide")
    def test_blocked_action_not_executed_and_reports_status_blocked(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        # Two consecutive attempts to navigate to blocked URL
        mock_decide.side_effect = [
            Decision(
                tool_name="goto",
                tool_args={"url": "https://example.com/checkout-complete"},
                reasoning="Attempt direct jump to complete.",
                input_tokens=40,
                output_tokens=10,
            ),
            Decision(
                tool_name="goto",
                tool_args={"url": "https://example.com/checkout-complete"},
                reasoning="Retry direct jump to complete.",
                input_tokens=40,
                output_tokens=10,
            ),
        ]
        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Finish order", start_url="https://example.com")

        # 1. Action is reported with status 'blocked'
        assert result.status == "blocked"
        # 2. Confirmed not auto-converted to needs_confirmation
        assert result.status != "needs_confirmation"
        assert "blocked 2 times" in result.summary

        # 3. Destination URL is NOT executed on the browser
        for call_args in mock_browser_session.goto.call_args_list:
            called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
            assert "checkout-complete" not in called_url

    @patch("agent.agent.decide")
    def test_finish_on_checkout_step_two_converts_to_needs_confirmation(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        """Test with 11 steps ending in finish at checkout-step-two.html.
        Verifies finish is intercepted and converted to needs_confirmation
        with question 'Order is ready to place. Confirm?' and recorded in trace."""
        trace_file = Path("traces/run_20261002_121647_eab476/trace.json")
        if trace_file.exists():
            recorded_data = json.loads(trace_file.read_text(encoding="utf-8"))
            recorded_steps = recorded_data["steps"]
        else:
            recorded_steps = [
                {"step": 1, "url": "https://www.saucedemo.com/", "action": "type_text", "args": {"id": 1, "text": "standard_user", "press_enter": False}, "reasoning": "Enter username"},
                {"step": 2, "url": "https://www.saucedemo.com/", "action": "type_text", "args": {"id": 2, "text": "secret_sauce", "press_enter": False}, "reasoning": "Enter password"},
                {"step": 3, "url": "https://www.saucedemo.com/", "action": "click", "args": {"id": 3}, "reasoning": "Click login"},
                {"step": 4, "url": "https://www.saucedemo.com/inventory.html", "action": "click", "args": {"id": 6}, "reasoning": "Add backpack"},
                {"step": 5, "url": "https://www.saucedemo.com/inventory.html", "action": "click", "args": {"id": 2}, "reasoning": "Navigate to cart"},
                {"step": 6, "url": "https://www.saucedemo.com/cart.html", "action": "click", "args": {"id": 6}, "reasoning": "Checkout"},
                {"step": 7, "url": "https://www.saucedemo.com/checkout-step-one.html", "action": "type_text", "args": {"id": 3, "text": "John", "press_enter": False}, "reasoning": "Enter first name"},
                {"step": 8, "url": "https://www.saucedemo.com/checkout-step-one.html", "action": "type_text", "args": {"id": 4, "text": "Doe", "press_enter": False}, "reasoning": "Enter last name"},
                {"step": 9, "url": "https://www.saucedemo.com/checkout-step-one.html", "action": "type_text", "args": {"id": 5, "text": "12345", "press_enter": False}, "reasoning": "Enter zip"},
                {"step": 10, "url": "https://www.saucedemo.com/checkout-step-one.html", "action": "click", "args": {"id": 7}, "reasoning": "Proceed to overview"},
                {"step": 11, "url": "https://www.saucedemo.com/checkout-step-two.html", "action": "finish", "args": {"success": True, "summary": "Reached checkout overview page with Backpack in cart, details entered. Order not placed."}, "reasoning": "The overview page after checkout is displayed and the order has not been placed. Goal achieved."},
            ]

        assert len(recorded_steps) == 11
        assert recorded_steps[-1]["url"] == "https://www.saucedemo.com/checkout-step-two.html"
        assert recorded_steps[-1]["action"] == "finish"

        # Build observations and decision mocks for all 11 steps
        observations = []
        decisions = []
        for s in recorded_steps:
            target_id = s["args"].get("id", 1)
            observations.append(
                Observation(
                    url=s["url"],
                    title="Swag Labs",
                    elements=[
                        InteractiveElement(id=target_id, tag="input" if s["action"] == "type_text" else "button", text="Element"),
                        InteractiveElement(id=99, tag="div", text="Other"),
                    ],
                    screenshot_path=None,
                )
            )
            decisions.append(
                Decision(
                    tool_name=s["action"],
                    tool_args=s["args"],
                    reasoning=s["reasoning"],
                    input_tokens=100,
                    output_tokens=20,
                )
            )

        mock_browser_session.observe.side_effect = observations
        mock_decide.side_effect = decisions

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            max_steps=12,
        )

        goal = "Login, add the Sauce Labs Backpack to the cart, go through checkout with name John Doe and zip 12345 until the final overview page. Do NOT place the order. Ask me to confirm before the order is placed."
        result = agent.run(goal=goal, start_url="https://www.saucedemo.com")

        # Verify agent status, question, and conversion flag
        assert result.status == "needs_confirmation"
        assert result.question == "Order is ready to place. Confirm?"
        assert result.finish_converted_to_confirmation is True
        assert result.steps == 11
        assert result.final_url == "https://www.saucedemo.com/checkout-step-two.html"
        assert "Order is ready to place. Confirm?" in result.summary

        # Verify trace.json contents
        assert result.trace_file is not None
        assert Path(result.trace_file).exists()
        with open(result.trace_file, "r", encoding="utf-8") as f:
            trace_json = json.load(f)

        assert trace_json["status"] == "needs_confirmation"
        assert trace_json["question"] == "Order is ready to place. Confirm?"
        assert trace_json["finish_converted_to_confirmation"] is True
        assert trace_json["total_steps"] == 11
        assert len(trace_json["steps"]) == 11
        assert trace_json["steps"][-1]["action"] == "finish"
        assert trace_json["steps"][-1]["url"] == "https://www.saucedemo.com/checkout-step-two.html"
        assert trace_json["steps"][-1]["result"] == "finish_converted_to_confirmation"

    @pytest.mark.parametrize(
        "url_pattern",
        [
            "https://example.com/checkout-step-two",
            "https://example.com/payment",
            "https://example.com/checkout/payment.html",
            "https://example.com/order/review",
            "https://example.com/cart/review-order",
        ],
    )
    @patch("agent.agent.decide")
    def test_finish_on_payment_and_review_urls_converts_to_needs_confirmation(
        self, mock_decide, mock_browser_session, tmp_path, url_pattern
    ):
        mock_browser_session.observe.return_value = Observation(
            url=url_pattern,
            title="Checkout",
            elements=[InteractiveElement(id=1, tag="button", text="Place Order")],
            screenshot_path=None,
        )
        mock_decide.return_value = Decision(
            tool_name="finish",
            tool_args={"success": True, "summary": "Finished order."},
            reasoning="Done.",
            input_tokens=50,
            output_tokens=10,
        )

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
        )
        result = agent.run(goal="Buy item", start_url=url_pattern)

        assert result.status == "needs_confirmation"
        assert result.question == "Order is ready to place. Confirm?"
        assert result.finish_converted_to_confirmation is True
        assert "Order is ready to place. Confirm?" in result.summary

        with open(result.trace_file, "r", encoding="utf-8") as f:
            trace_json = json.load(f)
        assert trace_json["status"] == "needs_confirmation"
        assert trace_json["question"] == "Order is ready to place. Confirm?"
        assert trace_json["finish_converted_to_confirmation"] is True
