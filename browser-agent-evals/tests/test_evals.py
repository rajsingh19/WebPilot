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

    def test_categorize_failure_navigation_timeout_in_check_detail_classified_as_slow_page(self):
        cat = categorize_failure(
            test_category="functional",
            run_status="failed",
            summary="element not found",
            check_detail="Navigation timeout on goto: Timeout 30000ms exceeded",
        )
        assert cat == "slow_page"

    def test_categorize_failure_navigation_timeout_in_trace_classified_as_slow_page_not_wrong_element(self):
        trace_data = {
            "navigation_error": "Failed to navigate to https://the-internet.herokuapp.com/dropdown: Page.goto: Timeout 30000ms exceeded.",
            "steps": [
                {
                    "step": 1,
                    "action": "finish",
                    "result": "Failed: No dropdown element ID available in observation.",
                }
            ],
        }
        cat = categorize_failure(
            test_category="functional",
            run_status="failed",
            summary="Failed: No dropdown element ID available in observation.",
            check_detail="Expected dropdown value '2', but found ''",
            trace_data=trace_data,
        )
        assert cat == "slow_page"

    def test_categorize_failure_goto_step_timeout_in_trace_classified_as_slow_page(self):
        trace_data = {
            "steps": [
                {
                    "step": 1,
                    "action": "goto",
                    "args": {"url": "https://example.com"},
                    "result": "Page.goto: Timeout 30000ms exceeded",
                }
            ]
        }
        cat = categorize_failure(
            test_category="functional",
            run_status="failed",
            summary="Wrong button clicked",
            check_detail="Wrong element selected",
            trace_data=trace_data,
        )
        assert cat == "slow_page"


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


class TestBudgetAndEstimates:
    """Tests for --max-llm-calls, budget skipping, LLM call logging, and daily request limit estimation."""

    def test_compute_eval_summary_excludes_skipped_budget(self):
        """Tests marked skipped_budget are excluded from the pass-rate denominator."""
        results = [
            {
                "test_id": "test_1",
                "category": "functional",
                "passed": True,
                "actual_status": "success",
                "steps": 2,
                "llm_calls": 2,
                "cost_usd": 0.001,
                "duration_s": 2.0,
                "failure_category": "",
            },
            {
                "test_id": "test_2",
                "category": "functional",
                "passed": False,
                "actual_status": "failed",
                "steps": 4,
                "llm_calls": 4,
                "cost_usd": 0.002,
                "duration_s": 3.0,
                "failure_category": "wrong_element",
            },
            {
                "test_id": "test_3",
                "category": "functional",
                "passed": False,
                "actual_status": "skipped_budget",
                "steps": 0,
                "llm_calls": 0,
                "cost_usd": 0.0,
                "duration_s": 0.0,
                "failure_category": "skipped_budget",
            },
            {
                "test_id": "test_4",
                "category": "functional",
                "passed": False,
                "actual_status": "skipped_budget",
                "steps": 0,
                "llm_calls": 0,
                "cost_usd": 0.0,
                "duration_s": 0.0,
                "failure_category": "skipped_budget",
            },
        ]

        summary = compute_eval_summary(results)

        assert summary["total_runs"] == 4
        assert summary["skipped_budget_count"] == 2
        # eval_total should exclude the 2 skipped runs: 4 - 2 = 2
        assert summary["eval_total"] == 2
        assert summary["passed_runs"] == 1
        # Pass rate is 1 / 2 = 50.0%
        assert summary["overall_pass_rate_pct"] == 50.0
        assert summary["total_llm_calls"] == 6

    @patch("evals.run_evals.save_reports")
    @patch("evals.run_evals.print_rich_results")
    @patch("evals.run_evals.run_single_eval")
    def test_max_llm_calls_stops_cleanly_marks_remaining_and_saves_valid_report(
        self, mock_run_single_eval, mock_print_rich, mock_save_reports, capsys, monkeypatch, tmp_path
    ):
        """Suite stops cleanly when max_llm_calls is reached, remaining are marked skipped_budget, valid report is saved."""
        from evals.run_evals import main

        # Simulate run 1 using 3 LLM calls
        mock_run_single_eval.return_value = {
            "test_id": "test_1",
            "category": "functional",
            "passed": True,
            "actual_status": "success",
            "steps": 3,
            "llm_calls": 3,
            "cost_usd": 0.001,
            "duration_s": 1.5,
            "failure_category": "",
            "checker_detail": "Passed",
            "summary": "Completed",
            "final_url": "https://example.com",
            "trace_file": None,
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
        }
        mock_save_reports.return_value = (Path("test.json"), Path("test.md"))

        test_data = [
            {"id": "test_1", "category": "functional", "goal": "g1", "start_url": "u1", "check": "c1", "expected_status": "success"},
            {"id": "test_2", "category": "functional", "goal": "g2", "start_url": "u2", "check": "c2", "expected_status": "success"},
            {"id": "test_3", "category": "functional", "goal": "g3", "start_url": "u3", "check": "c3", "expected_status": "success"},
        ]
        tests_file = tmp_path / "tests.json"
        with open(tests_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_evals.py",
                "--tests-file",
                str(tests_file),
                "--max-llm-calls",
                "3",
                "--provider",
                "groq",
            ],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0

        # Only 1 test actually ran
        assert mock_run_single_eval.call_count == 1

        # Check save_reports was called with is_invalid_run=False
        _, kwargs = mock_save_reports.call_args
        assert kwargs["is_invalid_run"] is False

        # Results passed to save_reports should have 3 records (1 ran, 2 skipped_budget)
        results = kwargs["results"]
        assert len(results) == 3
        assert results[0]["actual_status"] == "success"
        assert results[1]["actual_status"] == "skipped_budget"
        assert results[1]["failure_category"] == "skipped_budget"
        assert results[2]["actual_status"] == "skipped_budget"
        assert results[2]["failure_category"] == "skipped_budget"

        captured = capsys.readouterr()
        # Verify LLM calls per run and running total printed
        assert "LLM calls used: 3 (run) | 3 / 3 (running total)" in captured.out
        # Verify clean stop message
        assert "Budget limit reached (3 >= 3 LLM calls). Stopping suite cleanly." in captured.out

    @patch("evals.run_evals.save_reports")
    @patch("evals.run_evals.print_rich_results")
    @patch("evals.run_evals.run_single_eval")
    def test_estimate_and_daily_limit_warning(
        self, mock_run_single_eval, mock_print_rich, mock_save_reports, capsys, monkeypatch, tmp_path
    ):
        """Prints estimate before starting and warns if estimate exceeds DAILY_REQUEST_LIMIT_<PROVIDER>."""
        from evals.run_evals import main

        mock_run_single_eval.return_value = {
            "test_id": "test_1",
            "category": "functional",
            "passed": True,
            "actual_status": "success",
            "steps": 2,
            "llm_calls": 2,
            "cost_usd": 0.001,
            "duration_s": 1.0,
            "failure_category": "",
            "checker_detail": "Passed",
            "summary": "Completed",
            "final_url": "https://example.com",
            "trace_file": None,
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
        }
        mock_save_reports.return_value = (Path("test.json"), Path("test.md"))

        test_data = [
            {"id": "t1", "category": "functional", "goal": "g", "start_url": "u", "check": "c", "expected_status": "success"},
            {"id": "t2", "category": "functional", "goal": "g", "start_url": "u", "check": "c", "expected_status": "success"},
        ]
        tests_file = tmp_path / "tests.json"
        with open(tests_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        monkeypatch.setenv("DAILY_REQUEST_LIMIT_GROQ", "10")
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_evals.py",
                "--tests-file",
                str(tests_file),
                "--provider",
                "groq",
                "--repeat",
                "1",
            ],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        out_text = " ".join(captured.out.split())
        # 2 tests x 1 repeat x 8 avg_steps = 16 calls
        assert "Estimated LLM calls: 2 tests x 1 repeat x ~8 avg_steps = ~16 calls" in out_text
        # 16 > 10, so warning must be printed
        assert "WARNING: Estimated LLM calls (~16) exceeds known daily limit of 10 for groq (DAILY_REQUEST_LIMIT_GROQ)" in out_text


