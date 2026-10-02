"""Unit tests for evaluation runner, failure categorization, reporting, and comparison."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from evals.compare import compare_runs, load_result_file
from evals.run_evals import (
    categorize_failure,
    compute_eval_summary,
    matches_expected_status,
    save_reports,
)


class TestEvalRunnerLogic:
    """Tests for status matching and failure categorization."""

    def test_matches_expected_status(self):
        assert matches_expected_status("success", "success")
        assert matches_expected_status("needs_confirmation", "needs_confirmation or blocked")
        assert matches_expected_status("blocked", "needs_confirmation or blocked")
        assert not matches_expected_status("failed", "success")
        assert not matches_expected_status("success", "needs_confirmation")

    def test_categorize_failure_safety_violation(self):
        cat = categorize_failure(
            test_category="safety",
            run_status="success",
            summary="Bought item",
            check_detail="SAFETY VIOLATION: Final order was completed",
        )
        assert cat == "safety_violation"

    def test_categorize_failure_loop(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="loop",
            summary="Action repeated 3 times",
            check_detail="Loop occurred",
        )
        assert cat == "loop"

    def test_categorize_failure_timeout(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="timeout",
            summary="Time limit reached",
            check_detail="Timed out",
        )
        assert cat == "timeout"

    def test_categorize_failure_hallucinated_id(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="failed",
            summary="Failed due to invalid element id",
            check_detail="Did not complete",
            trace_data={"hallucinated_ids_count": 2},
        )
        assert cat == "hallucinated_id"

    def test_categorize_failure_api_error(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="error",
            summary="Anthropic API error: Rate limit exceeded",
            check_detail="Error",
        )
        assert cat == "api_error"

    def test_categorize_failure_language(self):
        cat = categorize_failure(
            test_category="language",
            run_status="failed",
            summary="Could not understand prompt",
            check_detail="Login page not reached",
        )
        assert cat == "language_misread"

    def test_categorize_failure_wrong_element(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="failed",
            summary="Clicked button",
            check_detail="Expected cart badge 2, but found '1'",
        )
        assert cat == "wrong_element"


class TestReportingAndSummary:
    """Tests for metrics aggregation and report generation."""

    def test_compute_eval_summary(self):
        mock_results = [
            {
                "test_id": "t1",
                "category": "functional",
                "passed": True,
                "steps": 4,
                "cost_usd": 0.01,
                "duration_s": 5.0,
            },
            {
                "test_id": "t2",
                "category": "functional",
                "passed": False,
                "steps": 6,
                "cost_usd": 0.02,
                "duration_s": 10.0,
            },
            {
                "test_id": "t3",
                "category": "safety",
                "passed": True,
                "steps": 5,
                "cost_usd": 0.015,
                "duration_s": 6.0,
            },
            {
                "test_id": "t4",
                "category": "safety",
                "passed": False,
                "steps": 5,
                "cost_usd": 0.015,
                "duration_s": 6.0,
            },
        ]

        summary = compute_eval_summary(mock_results)
        assert summary["total_runs"] == 4
        assert summary["passed_runs"] == 2
        assert summary["overall_pass_rate_pct"] == 50.0
        assert summary["by_category"]["functional"]["pass_rate_pct"] == 50.0
        assert summary["by_category"]["safety"]["pass_rate_pct"] == 50.0
        assert summary["safety_failures_count"] == 1
        assert summary["avg_steps"] == 5.0

    def test_save_reports(self, tmp_path):
        results = [
            {
                "test_id": "test_01",
                "test_name": "login_en",
                "category": "functional",
                "passed": True,
                "actual_status": "success",
                "steps": 3,
                "cost_usd": 0.005,
                "duration_s": 4.2,
                "failure_category": "",
                "checker_detail": "Logged in",
            }
        ]
        summary = compute_eval_summary(results)

        json_path, md_path = save_reports(
            results=results,
            summary=summary,
            provider="anthropic",
            model_name="claude-sonnet-5-5",
            git_commit="abc1234",
            output_dir=str(tmp_path),
        )

        assert json_path.exists()
        assert md_path.exists()

        with open(json_path) as f:
            data = json.load(f)
            assert data["provider"] == "anthropic"
            assert data["git_commit"] == "abc1234"

        with open(md_path) as f:
            content = f.read()
            assert "Evaluation Report: anthropic / claude-sonnet-5-5" in content
            assert "`abc1234`" in content
            assert "| `test_01` | functional | PASS |" in content


class TestCompareTool:
    """Tests for evals.compare functionality."""

    def test_compare_runs(self, tmp_path):
        # Create two sample result files
        data1 = {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "git_commit": "abc1111",
            "summary": {
                "total_runs": 10,
                "passed_runs": 9,
                "overall_pass_rate_pct": 90.0,
                "by_category": {
                    "functional": {"passed": 6, "total": 6, "pass_rate_pct": 100.0},
                    "safety": {"passed": 2, "total": 2, "pass_rate_pct": 100.0},
                },
                "avg_cost_usd": 0.012,
                "avg_duration_s": 5.4,
                "safety_failures_count": 0,
            },
            "results": [],
        }

        data2 = {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "git_commit": "abc2222",
            "summary": {
                "total_runs": 10,
                "passed_runs": 8,
                "overall_pass_rate_pct": 80.0,
                "by_category": {
                    "functional": {"passed": 5, "total": 6, "pass_rate_pct": 83.3},
                    "safety": {"passed": 1, "total": 2, "pass_rate_pct": 50.0},
                },
                "avg_cost_usd": 0.002,
                "avg_duration_s": 3.8,
                "safety_failures_count": 1,
            },
            "results": [],
        }

        f1 = tmp_path / "run1.json"
        f2 = tmp_path / "run2.json"
        with open(f1, "w") as f:
            json.dump(data1, f)
        with open(f2, "w") as f:
            json.dump(data2, f)

        # Should execute cleanly without error
        compare_runs([str(f1), str(f2)])

    @patch("evals.run_evals.get_checker")
    @patch("evals.run_evals.Agent")
    @patch("evals.run_evals.BrowserSession")
    def test_run_single_eval_passes_max_steps_and_prints(
        self, mock_session_cls, mock_agent_cls, mock_get_checker, capsys
    ):
        from agent.agent import RunResult
        from evals.checks import CheckResult
        from evals.run_evals import run_single_eval

        mock_session = MagicMock()
        mock_session._page = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.run.return_value = RunResult(
            status="success",
            steps=3,
            total_tokens=100,
            estimated_cost_usd=0.001,
            duration_s=1.5,
            final_url="https://example.com/done",
            trace_dir="/tmp/trace",
            summary="Success",
        )
        mock_checker = MagicMock(return_value=CheckResult(passed=True, detail="All good"))
        mock_get_checker.return_value = mock_checker

        test_case = {
            "id": "custom_steps_test",
            "name": "Custom Steps Test",
            "category": "functional",
            "goal": "Test goal",
            "start_url": "https://example.com",
            "max_steps": 25,
            "check": "login_en",
            "expected_status": "success",
        }

        record = run_single_eval(
            test_case=test_case,
            provider="groq",
            model_name="test-model",
            headed=False,
        )

        captured = capsys.readouterr()
        assert "max_steps used: 25" in captured.out
        mock_agent.run.assert_called_once_with(
            goal="Test goal",
            start_url="https://example.com",
            max_steps=25,
        )
        assert record["passed"] is True
