"""Tests for provider failover, exhaustion cooldown, and eval reporting."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import Agent, RunResult
from agent.browser import BrowserSession, InteractiveElement, Observation
from agent.llm import (
    Decision,
    LLMError,
    get_model_for_provider,
    get_provider_priority,
    is_price_unverified,
    is_provider_exhausted,
    is_provider_unavailable_error,
    mark_provider_exhausted,
    normalize_provider_name,
    pick_active_provider,
    reset_provider_exhaustion,
)
from agent.providers.base import LLMError
from agent.providers.gemini_provider import GeminiProvider
from agent.providers.groq_provider import GroqProvider
from evals.compare import compare_runs, compute_group_metrics, load_result_file
from evals.run_evals import compute_eval_summary, run_single_eval, save_reports


@pytest.fixture
def mock_browser_session():
    """Mock BrowserSession returning a stable page observation."""
    session = MagicMock(spec=BrowserSession)
    session._page = MagicMock()
    session.observe.return_value = Observation(
        url="https://example.com",
        title="Example Domain",
        elements=[InteractiveElement(id=1, tag="button", text="Submit")],
        screenshot_path=None,
    )
    session.click.return_value = MagicMock(success=True, message="Clicked 1")
    session.scroll.return_value = MagicMock(success=True, message="Scrolled down")
    session.goto.return_value = MagicMock(success=True, message="Navigated")
    return session


class TestProviderFailoverExecution:
    """Tests for failover logic during agent execution."""

    @patch("agent.agent.decide")
    def test_primary_429_at_step_1_restarts_on_backup(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        """Primary provider 429 at step 1 causes run to restart on backup provider."""
        # 1st call on primary (groq) raises 429 rate limit
        # 2nd call on backup (gemini) succeeds with finish
        mock_decide.side_effect = [
            LLMError("429 rate_limit_exceeded: tokens per day limit reached"),
            Decision(
                tool_name="finish",
                tool_args={"success": True, "summary": "Goal reached via backup provider."},
                reasoning="Completed on backup provider.",
                input_tokens=50,
                output_tokens=20,
            ),
        ]

        agent = Agent(
            provider="groq",
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            failover_enabled=True,
        )

        result = agent.run(goal="Perform search", start_url="https://example.com")

        # Primary was marked exhausted
        assert is_provider_exhausted("groq") is True

        # Run succeeded on backup provider
        assert result.status == "success"
        assert result.failover_happened is True
        assert result.provider_used == "gemini"
        assert result.model_used == get_model_for_provider("gemini")

        # Verify trace.json contents
        trace_file = Path(result.trace_file)
        assert trace_file.exists()
        with open(trace_file, "r", encoding="utf-8") as tf:
            trace_data = json.load(tf)

        assert trace_data["failover_happened"] is True
        assert trace_data["provider_used"] == "gemini"
        assert trace_data["status"] == "success"

    @patch("agent.agent.decide")
    def test_429_mid_run_status_error_no_switch(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        """429 at step > 1 does NOT switch; ends run with status error and failure_category rate_limited."""
        # Step 1 succeeds with scroll
        # Step 2 raises 429 rate limit
        mock_decide.side_effect = [
            Decision(
                tool_name="scroll",
                tool_args={"direction": "down"},
                reasoning="Scroll to view more.",
                input_tokens=30,
                output_tokens=10,
            ),
            LLMError("429 Too Many Requests: Rate limit exceeded"),
        ]

        agent = Agent(
            provider="groq",
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            failover_enabled=True,
        )

        result = agent.run(goal="Scroll and submit", start_url="https://example.com")

        # Provider marked exhausted
        assert is_provider_exhausted("groq") is True

        # Status must be error, category rate_limited, no switch
        assert result.status == "error"
        assert result.failure_category == "rate_limited"
        assert result.failover_happened is False
        assert result.provider_used == "groq"
        assert result.steps == 1

        # Check trace.json
        trace_file = Path(result.trace_file)
        with open(trace_file, "r", encoding="utf-8") as tf:
            trace_data = json.load(tf)

        assert trace_data["status"] == "error"
        assert trace_data["failure_category"] == "rate_limited"
        assert trace_data["failover_happened"] is False
        assert trace_data["provider_used"] == "groq"

    def test_all_providers_exhausted_before_run(self, mock_browser_session, tmp_path):
        """When all providers are exhausted before a run, returns error and rate_limited."""
        for p in get_provider_priority():
            mark_provider_exhausted(p)

        agent = Agent(
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            failover_enabled=True,
        )
        result = agent.run(goal="Test all exhausted", start_url="https://example.com")

        assert result.status == "error"
        assert result.failure_category == "rate_limited"
        assert "All providers exhausted" in result.summary


class TestEvalFailoverAndInvalidRun:
    """Tests for eval suite runner handling rate limits, invalid_run marking, and reports."""

    def test_rate_limited_runs_excluded_from_denominator(self):
        """Rate limited runs have failure_category rate_limited and are excluded from pass-rate denominator."""
        results = [
            {
                "test_id": "test_1",
                "category": "functional",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "passed": True,
                "actual_status": "success",
                "steps": 2,
                "cost_usd": 0.001,
                "duration_s": 2.0,
                "failure_category": "",
            },
            {
                "test_id": "test_2",
                "category": "functional",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "passed": False,
                "actual_status": "failed",
                "steps": 5,
                "cost_usd": 0.002,
                "duration_s": 3.0,
                "failure_category": "wrong_element",
            },
            {
                "test_id": "test_3",
                "category": "functional",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "passed": False,
                "actual_status": "error",
                "steps": 1,
                "cost_usd": 0.0005,
                "duration_s": 1.0,
                "failure_category": "rate_limited",
            },
        ]

        summary = compute_eval_summary(results)

        assert summary["total_runs"] == 3
        assert summary["rate_limited_count"] == 1
        # Denominator should exclude rate-limited run: eval_total = 3 - 1 = 2
        assert summary["eval_total"] == 2
        assert summary["passed_runs"] == 1
        # Pass rate is 1 / 2 = 50.0%
        assert summary["overall_pass_rate_pct"] == 50.0

    def test_mixed_providers_report_header_and_per_provider_metrics(self, tmp_path):
        """Mixed provider runs produce 'MIXED PROVIDERS' header and distinct per-provider tables."""
        results = [
            {
                "test_id": "test_1",
                "category": "functional",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "passed": True,
                "actual_status": "success",
                "steps": 2,
                "cost_usd": 0.001,
                "duration_s": 2.0,
                "failure_category": "",
                "checker_detail": "Passed",
            },
            {
                "test_id": "test_2",
                "category": "functional",
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "passed": False,
                "actual_status": "failed",
                "steps": 4,
                "cost_usd": 0.0015,
                "duration_s": 2.5,
                "failure_category": "wrong_element",
                "checker_detail": "Wrong button",
            },
        ]

        summary = compute_eval_summary(results)
        assert summary["is_mixed_providers"] is True
        assert "groq" in summary["by_provider"]
        assert "gemini" in summary["by_provider"]
        assert summary["by_provider"]["groq"]["pass_rate_pct"] == 100.0
        assert summary["by_provider"]["gemini"]["pass_rate_pct"] == 0.0

        json_path, md_path = save_reports(
            results=results,
            summary=summary,
            provider="groq",
            model_name="openai/gpt-oss-120b",
            git_commit="deadbeef",
            output_dir=str(tmp_path),
        )

        assert json_path.exists()
        assert md_path.exists()

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        assert "# Evaluation Report: MIXED PROVIDERS" in md_content
        assert "### Performance by Provider" in md_content
        assert "`groq`" in md_content
        assert "`gemini`" in md_content

    def test_all_providers_exhausted_aborts_suite_and_marks_invalid_run(self, tmp_path):
        """When 3 consecutive rate-limited runs occur and all providers are exhausted, results are marked invalid_run."""
        results = [
            {
                "test_id": f"test_{i}",
                "category": "functional",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "passed": False,
                "actual_status": "error",
                "steps": 0,
                "cost_usd": 0.0,
                "duration_s": 0.5,
                "failure_category": "rate_limited",
                "checker_detail": "Rate limited",
            }
            for i in range(3)
        ]

        summary = compute_eval_summary(results)
        json_path, md_path = save_reports(
            results=results,
            summary=summary,
            provider="groq",
            model_name="openai/gpt-oss-120b",
            git_commit="abcdef1",
            output_dir=str(tmp_path),
            is_invalid_run=True,
        )

        assert json_path.name.startswith("invalid_run_")
        assert md_path.name.startswith("invalid_run_")

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["invalid_run"] is True
        assert payload["status"] == "invalid_run"

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "INVALID RUN" in md_content


class TestCompareGroupingByProviderModel:
    """Tests for evals/compare.py grouping by provider/model without merging."""

    def test_compare_groups_by_provider_model_never_merges_mixed_runs(self, tmp_path):
        """Compare tool splits mixed runs in a single dataset into separate rows for each (provider, model)."""
        # Dataset 1: Baseline with single provider
        data1 = {
            "timestamp": "2026-10-02T10:00:00Z",
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "git_commit": "1111111",
            "results": [
                {
                    "test_id": "test_1",
                    "category": "functional",
                    "provider": "anthropic",
                    "model": "claude-sonnet-5-5",
                    "passed": True,
                    "actual_status": "success",
                    "steps": 3,
                    "cost_usd": 0.01,
                    "duration_s": 2.0,
                    "failure_category": "",
                }
            ],
            "summary": {
                "total_runs": 1,
                "overall_pass_rate_pct": 100.0,
                "avg_cost_usd": 0.01,
                "avg_duration_s": 2.0,
                "safety_failures_count": 0,
                "by_category": {"functional": {"passed": 1, "total": 1, "pass_rate_pct": 100.0}},
            },
        }

        # Dataset 2: Mixed runs with groq and gemini
        data2 = {
            "timestamp": "2026-10-02T11:00:00Z",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "git_commit": "2222222",
            "results": [
                {
                    "test_id": "test_1",
                    "category": "functional",
                    "provider": "groq",
                    "model": "openai/gpt-oss-120b",
                    "passed": True,
                    "actual_status": "success",
                    "steps": 2,
                    "cost_usd": 0.001,
                    "duration_s": 1.5,
                    "failure_category": "",
                },
                {
                    "test_id": "test_2",
                    "category": "functional",
                    "provider": "gemini",
                    "model": "gemini-2.0-flash",
                    "passed": False,
                    "actual_status": "failed",
                    "steps": 4,
                    "cost_usd": 0.0005,
                    "duration_s": 2.0,
                    "failure_category": "wrong_element",
                },
            ],
            "summary": {
                "total_runs": 2,
                "overall_pass_rate_pct": 50.0,
                "is_mixed_providers": True,
            },
        }

        file1 = tmp_path / "baseline.json"
        file2 = tmp_path / "mixed.json"

        with open(file1, "w", encoding="utf-8") as f:
            json.dump(data1, f)
        with open(file2, "w", encoding="utf-8") as f:
            json.dump(data2, f)

        # Verify compute_group_metrics computes correct isolated metrics
        groq_items = [r for r in data2["results"] if r["provider"] == "groq"]
        gemini_items = [r for r in data2["results"] if r["provider"] == "gemini"]

        groq_metrics = compute_group_metrics(groq_items)
        assert groq_metrics["overall_pass_rate_pct"] == 100.0
        assert groq_metrics["total_runs"] == 1

        gemini_metrics = compute_group_metrics(gemini_items)
        assert gemini_metrics["overall_pass_rate_pct"] == 0.0
        assert gemini_metrics["total_runs"] == 1

        # Run compare_runs - must succeed and execute without crashing
        compare_runs([str(file1), str(file2)])


class TestProviderUnavailableAndPricing:
    """Tests for provider unavailable handling, retry backoff, and unverified pricing display."""

    def test_provider_unavailable_runs_excluded_from_denominator(self):
        """Runs with failure_category provider_unavailable are excluded from pass-rate denominator."""
        results = [
            {
                "test_id": "test_1",
                "category": "functional",
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "passed": True,
                "actual_status": "success",
                "steps": 3,
                "cost_usd": 0.0001,
                "duration_s": 2.0,
                "failure_category": "",
            },
            {
                "test_id": "test_2",
                "category": "functional",
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "passed": False,
                "actual_status": "error",
                "steps": 0,
                "cost_usd": 0.0,
                "duration_s": 1.0,
                "failure_category": "provider_unavailable",
            },
            {
                "test_id": "test_3",
                "category": "functional",
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "passed": False,
                "actual_status": "failed",
                "steps": 4,
                "cost_usd": 0.0002,
                "duration_s": 3.0,
                "failure_category": "wrong_element",
            },
        ]

        summary = compute_eval_summary(results)

        assert summary["total_runs"] == 3
        assert summary["provider_unavailable_count"] == 1
        # Denominator excludes provider_unavailable: eval_total = 3 - 1 = 2
        assert summary["eval_total"] == 2
        assert summary["passed_runs"] == 1
        # Pass rate is 1 / 2 = 50.0%
        assert summary["overall_pass_rate_pct"] == 50.0

        # Check compare tool's group metrics as well
        metrics = compute_group_metrics(results)
        assert metrics["total_runs"] == 3
        assert metrics["provider_unavailable_count"] == 1
        assert metrics["eval_total"] == 2
        assert metrics["overall_pass_rate_pct"] == 50.0

    def test_unverified_pricing_eval_report_header(self, tmp_path):
        """When model pricing is unverified, report header displays 'cost: estimated/unverified'."""
        assert is_price_unverified("gemini-3.8-flash", "gemini") is True

        results = [
            {
                "test_id": "test_1",
                "category": "functional",
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "passed": True,
                "actual_status": "success",
                "steps": 3,
                "cost_usd": 0.0001,
                "duration_s": 2.0,
                "failure_category": "",
                "checker_detail": "Passed",
            }
        ]

        summary = compute_eval_summary(results)
        json_path, md_path = save_reports(
            results=results,
            summary=summary,
            provider="gemini",
            model_name="gemini-3.8-flash",
            git_commit="abcdef1",
            output_dir=str(tmp_path),
        )

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        assert "cost: estimated/unverified" in md_content
        assert "- **Cost**: `cost: estimated/unverified`" in md_content
        assert "| Average Cost (USD) | cost: estimated/unverified |" in md_content

    @patch("agent.agent.decide")
    def test_503_raises_and_marked_provider_unavailable(
        self, mock_decide, mock_browser_session, tmp_path
    ):
        """When 503 is returned after retries, Agent sets failure_category to provider_unavailable."""
        mock_decide.side_effect = LLMError("503 The model is overloaded. Please try again later.")

        agent = Agent(
            provider="gemini",
            model_name="gemini-3.8-flash",
            browser_session=mock_browser_session,
            traces_root=str(tmp_path),
            failover_enabled=False,
        )

        result = agent.run(goal="Test 503", start_url="https://example.com")

        assert result.status == "error"
        assert result.failure_category == "provider_unavailable"
        assert "unavailable" in result.summary.lower()

        trace_file = Path(result.trace_file)
        with open(trace_file, "r", encoding="utf-8") as tf:
            trace_data = json.load(tf)
        assert trace_data["failure_category"] == "provider_unavailable"

    def test_grok_to_groq_normalization(self):
        """Provider name 'grok' is normalized to 'groq' everywhere."""
        assert normalize_provider_name("grok") == "groq"
        assert normalize_provider_name("groq") == "groq"
        assert normalize_provider_name("GROK") == "groq"
        assert normalize_provider_name("xai") == "groq"

        agent = Agent(provider="grok")
        assert agent.provider == "groq"

    def test_gemini_and_groq_default_4_retries(self):
        """Providers have max_retries default set to 4."""
        gemini = GeminiProvider(model_name="gemini-3.8-flash", api_key="dummy")
        assert gemini.max_retries == 4

        groq = GroqProvider(model_name="openai/gpt-oss-120b", api_key="dummy")
        assert groq.max_retries == 4
