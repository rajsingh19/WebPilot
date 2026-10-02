"""Evaluation test runner executing test suites against live browser sessions."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.agent import Agent, RunResult
from agent.browser import BrowserSession
from agent.llm import pick_active_provider
from evals.checks import CheckResult, get_checker

logger = logging.getLogger(__name__)
console = Console()

FAILURE_CATEGORIES = [
    "wrong_element",
    "popup_blocked",
    "loop",
    "timeout",
    "slow_page",
    "hallucinated_id",
    "language_misread",
    "safety_violation",
    "rate_limited",
    "api_error",
    "other",
]


def get_git_commit_hash() -> str:
    """Retrieves current short git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def matches_expected_status(actual: str, expected: str) -> bool:
    """Validates if actual run status matches expected test definition."""
    actual_clean = (actual or "").strip().lower()
    expected_clean = (expected or "").strip().lower()

    if expected_clean in ("needs_confirmation or blocked", "needs_confirmation | blocked", "needs_confirmation/blocked"):
        return actual_clean in ("needs_confirmation", "blocked")
    return actual_clean == expected_clean


def categorize_failure(
    test_category: str,
    run_status: str,
    summary: str,
    check_detail: str,
    trace_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Classifies a failed evaluation run into a canonical failure category."""
    detail_lower = (check_detail or "").lower()
    summary_lower = (summary or "").lower()
    status_lower = (run_status or "").lower()

    # Rule 1: SAFETY VIOLATION in detail or summary always maps to safety_violation
    if "safety violation" in detail_lower or "safety violation" in summary_lower:
        return "safety_violation"

    # Rule 2: Loop detection
    if status_lower == "loop":
        return "loop"

    # Rule 3: Timeout / Wall time
    if status_lower == "timeout" or "time limit" in summary_lower:
        return "timeout"

    # Rule 4: Hallucinated Element IDs
    if trace_data and trace_data.get("hallucinated_ids_count", 0) > 0:
        return "hallucinated_id"
    if "invalid element id" in summary_lower or "hallucinated" in summary_lower:
        return "hallucinated_id"

    # Rule 5: Rate limit or API / LLM Errors
    if status_lower == "error":
        if "rate limited" in summary_lower or "rate_limited" in summary_lower or "due to rate limits" in summary_lower:
            if "api error" in summary_lower and "rate limited" not in summary_lower:
                return "api_error"
            return "rate_limited"
        if any(term in summary_lower for term in ("api", "rate limit", "connection", "auth", "llm")):
            return "api_error"

    # Rule 6: Popups or cookie overlays
    if any(term in summary_lower for term in ("popup", "cookie", "overlay", "banner")):
        return "popup_blocked"

    # Rule 7: Slow page or network lag
    if any(term in summary_lower for term in ("slow", "network idle", "page load timeout")):
        return "slow_page"

    # Rule 8: Language / Hinglish misinterpretation
    if test_category == "language":
        return "language_misread"

    # Rule 9: Element selection / wrong element
    if any(term in detail_lower for term in ("element", "badge", "cart", "price", "not found", "wrong")):
        return "wrong_element"
    if any(term in summary_lower for term in ("element", "click", "selector", "button", "input")):
        return "wrong_element"

    return "other"


def run_single_eval(
    test_case: Dict[str, Any],
    provider: str,
    model_name: str,
    headed: bool = False,
    failover_enabled: bool = False,
) -> Dict[str, Any]:
    """Executes a single evaluation test case against a fresh BrowserSession."""
    test_id = test_case["id"]
    test_name = test_case.get("name", test_id)
    category = test_case.get("category", "functional")
    goal = test_case["goal"]
    start_url = test_case["start_url"]
    max_steps = int(test_case.get("max_steps", 12))
    check_name = test_case["check"]
    expected_status = test_case["expected_status"]

    session = BrowserSession(headless=not headed)
    run_result: Optional[RunResult] = None
    check_result: Optional[CheckResult] = None
    trace_data: Optional[Dict[str, Any]] = None

    try:
        session.start()
        agent = Agent(
            provider=provider,
            model_name=model_name,
            browser_session=session,
            headless=not headed,
            max_steps=max_steps,
            failover_enabled=failover_enabled,
        )

        print(f"max_steps used: {max_steps}")
        run_result = agent.run(goal=goal, start_url=start_url, max_steps=max_steps)

        # Run verification checker on the LIVE page BEFORE closing the browser
        checker_fn = get_checker(check_name)
        check_result = checker_fn(session._page, run_result)

        # Load trace data if generated
        if run_result.trace_file and Path(run_result.trace_file).exists():
            try:
                with open(run_result.trace_file, "r", encoding="utf-8") as tf:
                    trace_data = json.load(tf)
            except Exception as e:
                logger.debug("Failed reading trace file %s: %s", run_result.trace_file, e)

    except Exception as exc:
        logger.exception("Unexpected exception in eval run %s: %s", test_id, exc)
        if run_result is None:
            run_result = RunResult(
                status="error",
                steps=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                duration_s=0.0,
                final_url=start_url,
                trace_dir="",
                summary=f"Evaluation execution crashed: {exc}",
            )
        if check_result is None:
            check_result = CheckResult(passed=False, detail=f"Crashed before verification: {exc}")
    finally:
        session.close()

    # Determine overall evaluation pass condition
    status_matched = matches_expected_status(run_result.status, expected_status)
    passed = bool(check_result.passed and status_matched)

    failure_cat = getattr(run_result, "failure_category", None) or ""
    if not passed and not failure_cat:
        failure_cat = categorize_failure(
            test_category=category,
            run_status=run_result.status,
            summary=run_result.summary,
            check_detail=check_result.detail,
            trace_data=trace_data,
        )

    provider_used = (
        getattr(run_result, "provider_used", None)
        or (trace_data.get("provider_used") if trace_data else None)
        or (trace_data.get("provider") if trace_data else None)
        or provider
    )
    model_used = (
        getattr(run_result, "model_used", None)
        or (trace_data.get("model_used") if trace_data else None)
        or (trace_data.get("model") if trace_data else None)
        or model_name
    )
    failover_happened = (
        getattr(run_result, "failover_happened", False)
        or (trace_data.get("failover_happened", False) if trace_data else False)
    )

    return {
        "test_id": test_id,
        "test_name": test_name,
        "category": category,
        "passed": passed,
        "checker_passed": check_result.passed,
        "status_matched": status_matched,
        "expected_status": expected_status,
        "actual_status": run_result.status,
        "steps": run_result.steps,
        "total_tokens": run_result.total_tokens,
        "cost_usd": run_result.estimated_cost_usd,
        "duration_s": run_result.duration_s,
        "failure_category": failure_cat,
        "checker_detail": check_result.detail,
        "summary": run_result.summary,
        "final_url": run_result.final_url,
        "trace_file": run_result.trace_file,
        "provider": provider_used,
        "model": model_used,
        "provider_used": provider_used,
        "model_used": model_used,
        "failover_happened": failover_happened,
        "fallback_used": getattr(run_result, "fallback_used", False) or (trace_data.get("fallback_used", False) if trace_data else False),
    }


def compute_eval_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates metrics and pass rates across categories."""
    total = len(results)
    infra_errors = sum(1 for r in results if r.get("actual_status") == "infra_error")
    rate_limited_count = sum(1 for r in results if r.get("failure_category") == "rate_limited")
    eval_total = total - infra_errors - rate_limited_count

    passed = sum(
        1 for r in results
        if r["passed"] and r.get("actual_status") != "infra_error" and r.get("failure_category") != "rate_limited"
    )
    overall_rate = (passed / eval_total * 100.0) if eval_total > 0 else 0.0

    categories = sorted(list(set(r["category"] for r in results)))
    category_metrics: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_infra = sum(1 for r in cat_items if r.get("actual_status") == "infra_error")
        cat_rate_limited = sum(1 for r in cat_items if r.get("failure_category") == "rate_limited")
        cat_eval_total = len(cat_items) - cat_infra - cat_rate_limited
        cat_passed = sum(
            1 for r in cat_items
            if r["passed"] and r.get("actual_status") != "infra_error" and r.get("failure_category") != "rate_limited"
        )
        cat_rate = (cat_passed / cat_eval_total * 100.0) if cat_eval_total > 0 else 0.0
        category_metrics[cat] = {
            "total": len(cat_items),
            "passed": cat_passed,
            "infra_errors": cat_infra,
            "rate_limited": cat_rate_limited,
            "pass_rate_pct": round(cat_rate, 1),
        }

    # Group by model_used
    models = sorted(list(set(r.get("model_used") or r.get("model", "unknown") for r in results)))
    by_model: Dict[str, Dict[str, Any]] = {}
    for m in models:
        m_items = [r for r in results if (r.get("model_used") or r.get("model", "unknown")) == m]
        m_total = len(m_items)
        m_infra = sum(1 for r in m_items if r.get("actual_status") == "infra_error")
        m_rate_limited = sum(1 for r in m_items if r.get("failure_category") == "rate_limited")
        m_eval_total = m_total - m_infra - m_rate_limited
        m_passed = sum(
            1 for r in m_items
            if r["passed"] and r.get("actual_status") != "infra_error" and r.get("failure_category") != "rate_limited"
        )
        m_rate = (m_passed / m_eval_total * 100.0) if m_eval_total > 0 else 0.0
        m_avg_cost = (sum(r.get("cost_usd", 0.0) for r in m_items) / m_total) if m_total > 0 else 0.0
        m_avg_time = (sum(r.get("duration_s", 0.0) for r in m_items) / m_total) if m_total > 0 else 0.0
        by_model[m] = {
            "total": m_total,
            "infra_errors": m_infra,
            "rate_limited": m_rate_limited,
            "passed": m_passed,
            "eval_total": m_eval_total,
            "pass_rate_pct": round(m_rate, 1),
            "avg_cost_usd": round(m_avg_cost, 4),
            "avg_duration_s": round(m_avg_time, 2),
        }

    # Group by provider_used
    providers = sorted(list(set(r.get("provider") or r.get("provider_used", "unknown") for r in results)))
    by_provider: Dict[str, Dict[str, Any]] = {}
    for p in providers:
        p_items = [r for r in results if (r.get("provider") or r.get("provider_used", "unknown")) == p]
        p_total = len(p_items)
        p_infra = sum(1 for r in p_items if r.get("actual_status") == "infra_error")
        p_rate_limited = sum(1 for r in p_items if r.get("failure_category") == "rate_limited")
        p_eval_total = p_total - p_infra - p_rate_limited
        p_passed = sum(
            1 for r in p_items
            if r["passed"] and r.get("actual_status") != "infra_error" and r.get("failure_category") != "rate_limited"
        )
        p_rate = (p_passed / p_eval_total * 100.0) if p_eval_total > 0 else 0.0
        p_avg_cost = (sum(r.get("cost_usd", 0.0) for r in p_items) / p_total) if p_total > 0 else 0.0
        p_avg_time = (sum(r.get("duration_s", 0.0) for r in p_items) / p_total) if p_total > 0 else 0.0
        by_provider[p] = {
            "total": p_total,
            "infra_errors": p_infra,
            "rate_limited": p_rate_limited,
            "passed": p_passed,
            "eval_total": p_eval_total,
            "pass_rate_pct": round(p_rate, 1),
            "avg_cost_usd": round(p_avg_cost, 4),
            "avg_duration_s": round(p_avg_time, 2),
        }

    avg_steps = (sum(r.get("steps", 0) for r in results) / total) if total > 0 else 0.0
    avg_cost = (sum(r.get("cost_usd", 0.0) for r in results) / total) if total > 0 else 0.0
    avg_duration = (sum(r.get("duration_s", 0.0) for r in results) / total) if total > 0 else 0.0

    safety_failures = [
        r for r in results if r["category"] == "safety" and not r["passed"]
    ]

    return {
        "total_runs": total,
        "eval_total": eval_total,
        "infra_errors_count": infra_errors,
        "rate_limited_count": rate_limited_count,
        "passed_runs": passed,
        "overall_pass_rate_pct": round(overall_rate, 1),
        "by_category": category_metrics,
        "by_model": by_model,
        "by_provider": by_provider,
        "is_mixed_providers": len(providers) > 1,
        "avg_steps": round(avg_steps, 1),
        "avg_cost_usd": round(avg_cost, 4),
        "avg_duration_s": round(avg_duration, 2),
        "safety_failures_count": len(safety_failures),
    }


def save_reports(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    provider: str,
    model_name: str,
    git_commit: str,
    output_dir: str = "results",
    is_invalid_run: bool = False,
) -> Tuple[Path, Path]:
    """Generates JSON and Markdown report artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_model = model_name.replace("/", "_").replace(":", "_")
    prefix = "invalid_run_" if is_invalid_run else ""
    base_name = f"{prefix}{timestamp_str}_{provider}_{clean_model}"

    json_file = out_path / f"{base_name}.json"
    md_file = out_path / f"{base_name}.md"

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model_name,
        "git_commit": git_commit,
        "invalid_run": is_invalid_run,
        "status": "invalid_run" if is_invalid_run else "completed",
        "summary": summary,
        "results": results,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Generate Markdown Report
    is_mixed = summary.get("is_mixed_providers", False)
    header_title = "MIXED PROVIDERS" if is_mixed else f"{provider} / {model_name}"

    md_lines = [
        f"# Evaluation Report: {header_title}",
        "",
    ]
    if is_invalid_run:
        md_lines.extend([
            "> [!WARNING]",
            "> **INVALID RUN**: Suite aborted due to 3 consecutive rate limits with no backup provider available.",
            "",
        ])
    md_lines.extend([
        f"- **Timestamp**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Provider**: `{provider}`" + (" (MIXED)" if is_mixed else ""),
        f"- **Model**: `{model_name}`",
        f"- **Git Commit**: `{git_commit}`",
        f"- **Overall Pass Rate**: **{summary['overall_pass_rate_pct']}%** ({summary['passed_runs']}/{summary.get('eval_total', summary['total_runs'])})",
        f"- **Rate Limited Runs (excluded from denominator)**: **{summary.get('rate_limited_count', 0)}**",
        f"- **Infrastructure Errors**: **{summary.get('infra_errors_count', 0)}**",
        f"- **Safety Failures**: **{summary['safety_failures_count']}**",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Average Steps | {summary['avg_steps']} |",
        f"| Average Cost (USD) | ${summary['avg_cost_usd']:.4f} |",
        f"| Average Wall Time | {summary['avg_duration_s']:.2f}s |",
        "",
    ])

    # Per-provider section
    md_lines.extend([
        "### Performance by Provider",
        "",
        "| Provider | Runs | Rate Limited | Infra Errors | Passed / Eval Total | Pass Rate | Avg Cost ($) | Avg Time (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for p_name, p_data in summary.get("by_provider", {}).items():
        md_lines.append(
            f"| `{p_name}` | {p_data['total']} | {p_data.get('rate_limited', 0)} | {p_data.get('infra_errors', 0)} | "
            f"{p_data['passed']}/{p_data['eval_total']} | {p_data['pass_rate_pct']}% | "
            f"${p_data['avg_cost_usd']:.4f} | {p_data['avg_duration_s']:.2f}s |"
        )

    md_lines.extend([
        "",
        "### Performance by Model Used",
        "",
        "| Model Used | Runs | Infra Errors | Passed / Eval Total | Success Rate | Avg Cost ($) | Avg Time (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for m_name, m_data in summary.get("by_model", {}).items():
        eval_tot = m_data.get("eval_total", m_data["total"] - m_data.get("infra_errors", 0))
        md_lines.append(
            f"| `{m_name}` | {m_data['total']} | {m_data.get('infra_errors', 0)} | "
            f"{m_data['passed']}/{eval_tot} | {m_data['pass_rate_pct']}% | "
            f"${m_data['avg_cost_usd']:.4f} | {m_data['avg_duration_s']:.2f}s |"
        )

    md_lines.extend(
        [
            "",
            "### Pass Rate by Category",
            "",
            "| Category | Passed / Total | Pass Rate |",
            "| :--- | :--- | :--- |",
        ]
    )

    for cat, data in summary["by_category"].items():
        md_lines.append(f"| {cat.title()} | {data['passed']}/{data['total']} | {data['pass_rate_pct']}% |")

    if is_mixed:
        md_lines.extend(
            [
                "",
                "## Detailed Results",
                "",
                "| Test ID | Category | Provider | Model | Result | Actual Status | Steps | Cost ($) | Time (s) | Failure Category | Detail |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in results:
            res_badge = "PASS" if r["passed"] else "**FAIL**"
            fail_cat = r["failure_category"] or "-"
            detail_snippet = r["checker_detail"].replace("|", "\\|")
            md_lines.append(
                f"| `{r['test_id']}` | {r['category']} | `{r.get('provider', '-')}` | `{r.get('model', '-')}` | {res_badge} | `{r['actual_status']}` | "
                f"{r['steps']} | ${r['cost_usd']:.4f} | {r['duration_s']}s | {fail_cat} | {detail_snippet} |"
            )
    else:
        md_lines.extend(
            [
                "",
                "## Detailed Results",
                "",
                "| Test ID | Category | Result | Actual Status | Steps | Cost ($) | Time (s) | Failure Category | Detail |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in results:
            res_badge = "PASS" if r["passed"] else "**FAIL**"
            fail_cat = r["failure_category"] or "-"
            detail_snippet = r["checker_detail"].replace("|", "\\|")
            md_lines.append(
                f"| `{r['test_id']}` | {r['category']} | {res_badge} | `{r['actual_status']}` | "
                f"{r['steps']} | ${r['cost_usd']:.4f} | {r['duration_s']}s | {fail_cat} | {detail_snippet} |"
            )

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_file, md_file


def print_rich_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    provider: str,
    model_name: str,
    git_commit: str,
) -> None:
    """Renders evaluation results as formatted Rich tables in stdout."""
    is_mixed = summary.get("is_mixed_providers", False)
    title_suffix = " - MIXED PROVIDERS" if is_mixed else ""
    console.print(
        Panel.fit(
            f"[bold cyan]WebPilot Evaluation Suite{title_suffix}[/bold cyan]\n"
            f"[yellow]Provider:[/yellow] {provider}{' (MIXED)' if is_mixed else ''} | [yellow]Model:[/yellow] {model_name} | [yellow]Commit:[/yellow] {git_commit}\n"
            f"[yellow]Runs:[/yellow] {summary['total_runs']} | [yellow]Rate Limited:[/yellow] {summary.get('rate_limited_count', 0)} | [yellow]Infra Errors:[/yellow] {summary.get('infra_errors_count', 0)} | [yellow]Overall Pass Rate:[/yellow] [bold green]{summary['overall_pass_rate_pct']}%[/bold green]",
            border_style="cyan",
        )
    )

    table = Table(title="Test Execution Details", border_style="blue")
    table.add_column("Test ID", style="cyan")
    table.add_column("Category")
    if is_mixed:
        table.add_column("Provider", style="dim")
    table.add_column("Result")
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Failure Category", style="magenta")

    for r in results:
        res_str = "[green]PASS[/green]" if r["passed"] else "[bold red]FAIL[/bold red]"
        fail_str = r["failure_category"] if r["failure_category"] else "-"
        row = [
            r["test_id"],
            r["category"],
        ]
        if is_mixed:
            row.append(r.get("provider", "-"))
        row.extend([
            res_str,
            r["actual_status"],
            str(r["steps"]),
            f"${r['cost_usd']:.4f}",
            f"{r['duration_s']}s",
            fail_str,
        ])
        table.add_row(*row)

    console.print(table)

    provider_table = Table(title="Performance by Provider", border_style="cyan")
    provider_table.add_column("Provider", style="bold")
    provider_table.add_column("Runs", justify="center")
    provider_table.add_column("Rate Limited", justify="center")
    provider_table.add_column("Infra Errors", justify="center")
    provider_table.add_column("Passed / Eval Total", justify="center")
    provider_table.add_column("Pass Rate", justify="right")
    provider_table.add_column("Avg Cost", justify="right")
    provider_table.add_column("Avg Time", justify="right")

    for p_name, p_data in summary.get("by_provider", {}).items():
        provider_table.add_row(
            p_name,
            str(p_data["total"]),
            str(p_data.get("rate_limited", 0)),
            str(p_data.get("infra_errors", 0)),
            f"{p_data['passed']}/{p_data['eval_total']}",
            f"{p_data['pass_rate_pct']}%",
            f"${p_data['avg_cost_usd']:.4f}",
            f"{p_data['avg_duration_s']:.2f}s",
        )
    console.print(provider_table)

    model_table = Table(title="Performance by Model Used", border_style="magenta")
    model_table.add_column("Model Used", style="bold")
    model_table.add_column("Runs", justify="center")
    model_table.add_column("Infra Errors", justify="center")
    model_table.add_column("Passed / Eval Total", justify="center")
    model_table.add_column("Success Rate", justify="right")
    model_table.add_column("Avg Cost", justify="right")
    model_table.add_column("Avg Time", justify="right")

    for m_name, m_data in summary.get("by_model", {}).items():
        eval_tot = m_data.get("eval_total", m_data["total"] - m_data.get("infra_errors", 0))
        model_table.add_row(
            m_name,
            str(m_data["total"]),
            str(m_data.get("infra_errors", 0)),
            f"{m_data['passed']}/{eval_tot}",
            f"{m_data['pass_rate_pct']}%",
            f"${m_data['avg_cost_usd']:.4f}",
            f"{m_data['avg_duration_s']:.2f}s",
        )
    console.print(model_table)

    summary_table = Table(title="Category & Overall Performance", border_style="green")
    summary_table.add_column("Category", style="bold")
    summary_table.add_column("Passed / Total", justify="center")
    summary_table.add_column("Pass Rate", justify="right")

    for cat, data in summary["by_category"].items():
        summary_table.add_row(
            cat.title(),
            f"{data['passed']}/{data['total']}",
            f"{data['pass_rate_pct']}%",
        )

    summary_table.add_section()
    summary_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{summary['passed_runs']}/{summary.get('eval_total', summary['total_runs'])}[/bold]",
        f"[bold green]{summary['overall_pass_rate_pct']}%[/bold green]",
    )

    console.print(summary_table)

    metrics_panel = (
        f"[bold]Average Steps:[/] {summary['avg_steps']}  |  "
        f"[bold]Average Cost:[/] ${summary['avg_cost_usd']:.4f}  |  "
        f"[bold]Average Duration:[/] {summary['avg_duration_s']}s  |  "
        f"[bold]Rate Limited:[/] {summary.get('rate_limited_count', 0)}  |  "
        f"[bold]Safety Failures:[/] [{'red' if summary['safety_failures_count'] > 0 else 'green'}]{summary['safety_failures_count']}[/]"
    )
    console.print(Panel(metrics_panel, border_style="yellow"))


def main() -> None:
    """CLI entrypoint for running evaluation tests."""
    parser = argparse.ArgumentParser(description="WebPilot Evaluation Runner")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (anthropic, gemini, groq)")
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    parser.add_argument("--tests", type=str, default=None, help="Comma-separated test IDs to run")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions per test case")
    parser.add_argument("--headed", action="store_true", help="Run browser sessions in headed mode")
    parser.add_argument("--tests-file", type=str, default="evals/tests.json", help="Path to tests.json")
    parser.add_argument("--failover", action="store_true", help="Enable provider failover (default false for evals)")

    args = parser.parse_args()

    provider = args.provider or os.getenv("PROVIDER", "anthropic")
    model_name = args.model or os.getenv("MODEL_NAME", "claude-sonnet-5-5")
    git_commit = get_git_commit_hash()

    tests_path = Path(args.tests_file)
    if not tests_path.exists():
        console.print(f"[bold red]Tests definition file '{tests_path}' not found.[/bold red]")
        sys.exit(1)

    with open(tests_path, "r", encoding="utf-8") as f:
        all_tests: List[Dict[str, Any]] = json.load(f)

    valid_test_ids = [t["id"] for t in all_tests]
    selected_ids = [t.strip() for t in args.tests.split(",")] if args.tests else None
    if selected_ids:
        unknown_ids = [tid for tid in selected_ids if tid not in valid_test_ids]
        if unknown_ids:
            console.print(
                f"[bold red]Error: Unknown test ID(s): {', '.join(unknown_ids)}[/bold red]\n"
                f"[yellow]Valid test IDs are:[/yellow]\n  • " + "\n  • ".join(valid_test_ids)
            )
            sys.exit(1)
        tests_to_run = [t for t in all_tests if t["id"] in selected_ids]
    else:
        tests_to_run = all_tests

    console.print(
        f"[cyan]Loaded {len(tests_to_run)} test definition(s). Running with repeat={args.repeat}...[/cyan]"
    )

    results: List[Dict[str, Any]] = []
    consecutive_rate_limits = 0
    aborted_due_to_exhaustion = False

    for test in tests_to_run:
        if aborted_due_to_exhaustion:
            break
        for rep in range(1, args.repeat + 1):
            rep_label = f" (run {rep}/{args.repeat})" if args.repeat > 1 else ""
            console.print(f"[dim]Executing: {test['id']}{rep_label}...[/dim]")
            eval_record = run_single_eval(
                test_case=test,
                provider=provider,
                model_name=model_name,
                headed=args.headed,
                failover_enabled=args.failover,
            )
            results.append(eval_record)

            if eval_record.get("failure_category") == "rate_limited":
                consecutive_rate_limits += 1
                next_prov, _ = pick_active_provider() if args.failover else (None, None)
                if consecutive_rate_limits >= 3 and next_prov is None:
                    console.print(
                        "[bold red]ABORTING SUITE: 3 consecutive runs were rate limited and no backup provider is available.[/bold red]"
                    )
                    aborted_due_to_exhaustion = True
                    break
            else:
                consecutive_rate_limits = 0

            # Pause 1-2s between runs to respect rate limits
            time.sleep(1.5)

    summary = compute_eval_summary(results)
    json_path, md_path = save_reports(
        results=results,
        summary=summary,
        provider=provider,
        model_name=model_name,
        git_commit=git_commit,
        is_invalid_run=aborted_due_to_exhaustion,
    )

    print_rich_results(
        results=results,
        summary=summary,
        provider=provider,
        model_name=model_name,
        git_commit=git_commit,
    )

    console.print(f"[green]Saved evaluation artifacts:[/green]\n  • JSON: {json_path}\n  • Markdown: {md_path}")

    if aborted_due_to_exhaustion:
        console.print("[bold red]Run marked as 'invalid_run' due to provider exhaustion.[/bold red]")
        sys.exit(1)

    # Exit code 1 if any safety-category test fails
    if summary["safety_failures_count"] > 0:
        console.print(
            f"[bold red]FAILURE: {summary['safety_failures_count']} safety test(s) failed. Exiting with code 1.[/bold red]"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
