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
from agent.llm import (
    get_model_for_provider,
    is_price_unverified,
    normalize_provider_name,
    pick_active_provider,
)
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
    "provider_unavailable",
    "skipped_budget",
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

    # Rule 1: Skipped budget
    if status_lower == "skipped_budget" or "skipped_budget" in summary_lower:
        return "skipped_budget"

    # Rule 2: SAFETY VIOLATION in detail or summary always maps to safety_violation
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

    # Rule 5: Rate limit, provider unavailable, or API / LLM Errors
    if status_lower == "error":
        if "provider_unavailable" in summary_lower or any(p in summary_lower for p in ("503", "500", "502", "504", "unavailable", "high demand", "spikes in demand")):
            return "provider_unavailable"
        if "rate limited" in summary_lower or "rate_limited" in summary_lower or "due to rate limits" in summary_lower:
            if "api error" in summary_lower and "rate limited" not in summary_lower:
                return "api_error"
            return "rate_limited"
        if any(term in summary_lower for term in ("api", "rate limit", "connection", "auth", "llm")):
            return "api_error"

    # Rule 6: Popups or cookie overlays
    if any(term in summary_lower for term in ("popup", "cookie", "overlay", "banner")):
        return "popup_blocked"

    # Rule 7: Slow page, network lag, or navigation timeout on goto
    is_nav_timeout = (
        any(term in detail_lower for term in ("navigation timeout", "timeout 30000ms", "timeout 45000ms", "page.goto"))
        or any(term in summary_lower for term in ("navigation timeout", "timeout 30000ms", "timeout 45000ms", "page.goto"))
        or ("navigation" in detail_lower and "timeout" in detail_lower)
        or ("goto" in detail_lower and "timeout" in detail_lower)
        or ("navigation" in summary_lower and "timeout" in summary_lower)
        or ("goto" in summary_lower and "timeout" in summary_lower)
    )
    if not is_nav_timeout and trace_data:
        nav_err = str(trace_data.get("navigation_error") or "").lower()
        if any(term in nav_err for term in ("timeout", "30000ms", "45000ms", "navigation timeout")):
            is_nav_timeout = True
        if not is_nav_timeout:
            for st in trace_data.get("steps", []):
                act = str(st.get("action", "")).lower()
                res = str(st.get("result", "")).lower()
                reasoning = str(st.get("reasoning", "")).lower()
                if (act == "goto" and "timeout" in res) or any(
                    term in res for term in ("navigation timeout", "timeout 30000ms", "timeout 45000ms", "page.goto")
                ):
                    is_nav_timeout = True
                    break
                if any(term in reasoning for term in ("navigation timeout", "timeout 30000ms", "timeout 45000ms")):
                    is_nav_timeout = True
                    break

    if is_nav_timeout or any(term in summary_lower for term in ("slow", "network idle", "page load timeout")):
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
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes a single evaluation test case against a fresh BrowserSession."""
    test_id = test_case["id"]
    test_name = test_case.get("name", test_id)
    category = test_case.get("category", "functional")
    goal = test_case["goal"]
    start_url = test_case["start_url"]
    base_max_steps = int(test_case.get("max_steps", 12))
    max_steps = max_steps_override if max_steps_override is not None else base_max_steps
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

    upstream_provider = (
        getattr(run_result, "upstream_provider", None)
        or (trace_data.get("upstream_provider") if trace_data else None)
    )
    if not upstream_provider and trace_data and "steps" in trace_data:
        for st in reversed(trace_data.get("steps", [])):
            if st.get("upstream_provider"):
                upstream_provider = st.get("upstream_provider")
                break

    llm_calls = (
        getattr(run_result, "llm_calls", 0)
        or (trace_data.get("llm_calls") if trace_data else 0)
        or run_result.steps
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
        "llm_calls": llm_calls,
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
        "upstream_provider": upstream_provider,
        "failover_happened": failover_happened,
        "fallback_used": getattr(run_result, "fallback_used", False) or (trace_data.get("fallback_used", False) if trace_data else False),
        "has_real_cost": getattr(run_result, "has_real_cost", False) or (trace_data.get("has_real_cost", False) if trace_data else False),
    }


def compute_eval_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates metrics and pass rates across categories."""
    total = len(results)
    infra_errors = sum(1 for r in results if r.get("actual_status") == "infra_error")
    rate_limited_count = sum(1 for r in results if r.get("failure_category") == "rate_limited")
    provider_unavailable_count = sum(1 for r in results if r.get("failure_category") == "provider_unavailable")
    skipped_budget_count = sum(
        1 for r in results
        if r.get("failure_category") == "skipped_budget" or r.get("actual_status") == "skipped_budget"
    )
    eval_total = total - infra_errors - rate_limited_count - provider_unavailable_count - skipped_budget_count

    passed = sum(
        1 for r in results
        if r["passed"]
        and r.get("actual_status") not in ("infra_error", "skipped_budget")
        and r.get("failure_category") not in ("rate_limited", "provider_unavailable", "skipped_budget")
    )
    overall_rate = (passed / eval_total * 100.0) if eval_total > 0 else 0.0

    categories = sorted(list(set(r["category"] for r in results)))
    category_metrics: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_infra = sum(1 for r in cat_items if r.get("actual_status") == "infra_error")
        cat_rate_limited = sum(1 for r in cat_items if r.get("failure_category") == "rate_limited")
        cat_unavailable = sum(1 for r in cat_items if r.get("failure_category") == "provider_unavailable")
        cat_skipped_budget = sum(
            1 for r in cat_items
            if r.get("failure_category") == "skipped_budget" or r.get("actual_status") == "skipped_budget"
        )
        cat_eval_total = len(cat_items) - cat_infra - cat_rate_limited - cat_unavailable - cat_skipped_budget
        cat_passed = sum(
            1 for r in cat_items
            if r["passed"]
            and r.get("actual_status") not in ("infra_error", "skipped_budget")
            and r.get("failure_category") not in ("rate_limited", "provider_unavailable", "skipped_budget")
        )
        cat_rate = (cat_passed / cat_eval_total * 100.0) if cat_eval_total > 0 else 0.0
        category_metrics[cat] = {
            "total": len(cat_items),
            "passed": cat_passed,
            "infra_errors": cat_infra,
            "rate_limited": cat_rate_limited,
            "provider_unavailable": cat_unavailable,
            "skipped_budget": cat_skipped_budget,
            "eval_total": cat_eval_total,
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
        m_unavailable = sum(1 for r in m_items if r.get("failure_category") == "provider_unavailable")
        m_skipped_budget = sum(
            1 for r in m_items
            if r.get("failure_category") == "skipped_budget" or r.get("actual_status") == "skipped_budget"
        )
        m_eval_total = m_total - m_infra - m_rate_limited - m_unavailable - m_skipped_budget
        m_passed = sum(
            1 for r in m_items
            if r["passed"]
            and r.get("actual_status") not in ("infra_error", "skipped_budget")
            and r.get("failure_category") not in ("rate_limited", "provider_unavailable", "skipped_budget")
        )
        m_rate = (m_passed / m_eval_total * 100.0) if m_eval_total > 0 else 0.0
        m_avg_cost = (sum(r.get("cost_usd", 0.0) for r in m_items) / m_total) if m_total > 0 else 0.0
        m_avg_time = (sum(r.get("duration_s", 0.0) for r in m_items) / m_total) if m_total > 0 else 0.0
        by_model[m] = {
            "total": m_total,
            "infra_errors": m_infra,
            "rate_limited": m_rate_limited,
            "provider_unavailable": m_unavailable,
            "skipped_budget": m_skipped_budget,
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
        p_unavailable = sum(1 for r in p_items if r.get("failure_category") == "provider_unavailable")
        p_skipped_budget = sum(
            1 for r in p_items
            if r.get("failure_category") == "skipped_budget" or r.get("actual_status") == "skipped_budget"
        )
        p_eval_total = p_total - p_infra - p_rate_limited - p_unavailable - p_skipped_budget
        p_passed = sum(
            1 for r in p_items
            if r["passed"]
            and r.get("actual_status") not in ("infra_error", "skipped_budget")
            and r.get("failure_category") not in ("rate_limited", "provider_unavailable", "skipped_budget")
        )
        p_rate = (p_passed / p_eval_total * 100.0) if p_eval_total > 0 else 0.0
        p_avg_cost = (sum(r.get("cost_usd", 0.0) for r in p_items) / p_total) if p_total > 0 else 0.0
        p_avg_time = (sum(r.get("duration_s", 0.0) for r in p_items) / p_total) if p_total > 0 else 0.0
        by_provider[p] = {
            "total": p_total,
            "infra_errors": p_infra,
            "rate_limited": p_rate_limited,
            "provider_unavailable": p_unavailable,
            "skipped_budget": p_skipped_budget,
            "passed": p_passed,
            "eval_total": p_eval_total,
            "pass_rate_pct": round(p_rate, 1),
            "avg_cost_usd": round(p_avg_cost, 4),
            "avg_duration_s": round(p_avg_time, 2),
        }

    avg_steps = (sum(r.get("steps", 0) for r in results) / total) if total > 0 else 0.0
    avg_cost = (sum(r.get("cost_usd", 0.0) for r in results) / total) if total > 0 else 0.0
    avg_duration = (sum(r.get("duration_s", 0.0) for r in results) / total) if total > 0 else 0.0
    total_llm_calls = sum(r.get("llm_calls", r.get("steps", 0)) for r in results)

    safety_failures = [
        r for r in results if r["category"] == "safety" and not r["passed"]
    ]

    return {
        "total_runs": total,
        "eval_total": eval_total,
        "infra_errors_count": infra_errors,
        "rate_limited_count": rate_limited_count,
        "provider_unavailable_count": provider_unavailable_count,
        "skipped_budget_count": skipped_budget_count,
        "total_llm_calls": total_llm_calls,
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

    has_unverified = is_price_unverified(model_name, provider) or any(
        (not r.get("has_real_cost")) and is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider"))
        for r in results
    )
    is_openrouter = normalize_provider_name(provider) == "openrouter" or any(
        normalize_provider_name(r.get("provider_used") or r.get("provider")) == "openrouter"
        for r in results
    )
    if has_unverified:
        cost_display = "cost: unverified" if is_openrouter else "cost: estimated/unverified"
    else:
        cost_display = f"${summary['avg_cost_usd']:.4f}"

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
    elif summary.get("skipped_budget_count", 0) > 0:
        md_lines.extend([
            "> [!NOTE]",
            f"> **PARTIAL REPORT**: Suite stopped cleanly after reaching `--max-llm-calls` budget. {summary['skipped_budget_count']} test(s) skipped.",
            "",
        ])
    md_lines.extend([
        f"- **Timestamp**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Provider**: `{provider}`" + (" (MIXED)" if is_mixed else ""),
        f"- **Model**: `{model_name}`",
    ])
    upstreams = sorted(list(set(r.get("upstream_provider") for r in results if r.get("upstream_provider"))))
    if upstreams:
        md_lines.append(f"- **Upstream Provider**: {', '.join(f'`{u}`' for u in upstreams)}")
    md_lines.extend([
        f"- **Cost**: `{cost_display}`",
        f"- **Git Commit**: `{git_commit}`",
        f"- **Overall Pass Rate**: **{summary['overall_pass_rate_pct']}%** ({summary['passed_runs']}/{summary.get('eval_total', summary['total_runs'])})",
        f"- **Total LLM Calls**: **{summary.get('total_llm_calls', 0)}**",
        f"- **Rate Limited Runs (excluded from denominator)**: **{summary.get('rate_limited_count', 0)}**",
        f"- **Provider Unavailable Runs (excluded from denominator)**: **{summary.get('provider_unavailable_count', 0)}**",
        f"- **Skipped due to Budget limit (excluded from denominator)**: **{summary.get('skipped_budget_count', 0)}**",
        f"- **Infrastructure Errors**: **{summary.get('infra_errors_count', 0)}**",
        f"- **Safety Failures**: **{summary['safety_failures_count']}**",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Total LLM Calls | {summary.get('total_llm_calls', 0)} |",
        f"| Skipped (Budget) | {summary.get('skipped_budget_count', 0)} |",
        f"| Average Steps | {summary['avg_steps']} |",
        f"| Average Cost (USD) | {cost_display} |",
        f"| Average Wall Time | {summary['avg_duration_s']:.2f}s |",
        "",
    ])

    # Per-provider section
    md_lines.extend([
        "### Performance by Provider",
        "",
        "| Provider | Runs | Rate Limited | Provider Unavailable | Skipped (Budget) | Infra Errors | Passed / Eval Total | Pass Rate | Avg Cost ($) | Avg Time (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for p_name, p_data in summary.get("by_provider", {}).items():
        p_norm = normalize_provider_name(p_name)
        p_has_real = any(r.get("has_real_cost") for r in results if normalize_provider_name(r.get("provider_used") or r.get("provider")) == p_norm)
        p_unverified = (not p_has_real) and is_price_unverified(None, p_name)
        if p_unverified:
            p_cost_str = "cost: unverified" if p_norm == "openrouter" else "cost: estimated/unverified"
        else:
            p_cost_str = f"${p_data['avg_cost_usd']:.4f}"
        md_lines.append(
            f"| `{p_name}` | {p_data['total']} | {p_data.get('rate_limited', 0)} | {p_data.get('provider_unavailable', 0)} | {p_data.get('skipped_budget', 0)} | {p_data.get('infra_errors', 0)} | "
            f"{p_data['passed']}/{p_data['eval_total']} | {p_data['pass_rate_pct']}% | "
            f"{p_cost_str} | {p_data['avg_duration_s']:.2f}s |"
        )

    md_lines.extend([
        "",
        "### Performance by Model Used",
        "",
        "| Model Used | Runs | Provider Unavailable | Skipped (Budget) | Infra Errors | Passed / Eval Total | Success Rate | Avg Cost ($) | Avg Time (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for m_name, m_data in summary.get("by_model", {}).items():
        eval_tot = m_data.get("eval_total", m_data["total"] - m_data.get("infra_errors", 0))
        m_has_real = any(r.get("has_real_cost") for r in results if (r.get("model_used") or r.get("model")) == m_name)
        m_unverified = (not m_has_real) and is_price_unverified(m_name)
        if m_unverified:
            m_cost_str = "cost: unverified" if is_openrouter else "cost: estimated/unverified"
        else:
            m_cost_str = f"${m_data['avg_cost_usd']:.4f}"
        md_lines.append(
            f"| `{m_name}` | {m_data['total']} | {m_data.get('provider_unavailable', 0)} | {m_data.get('skipped_budget', 0)} | {m_data.get('infra_errors', 0)} | "
            f"{m_data['passed']}/{eval_tot} | {m_data['pass_rate_pct']}% | "
            f"{m_cost_str} | {m_data['avg_duration_s']:.2f}s |"
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

    has_upstream = any(r.get("upstream_provider") for r in results)
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
            if r.get("has_real_cost"):
                cost_str = f"${r['cost_usd']:.4f}"
            elif is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider")):
                r_prov = normalize_provider_name(r.get("provider_used") or r.get("provider"))
                cost_str = "unverified" if r_prov == "openrouter" else "estimated/unverified"
            else:
                cost_str = f"${r['cost_usd']:.4f}"
            prov_label = r.get('provider_used') or r.get('provider', '-')
            if r.get('upstream_provider'):
                prov_label = f"{prov_label} ({r['upstream_provider']})"
            md_lines.append(
                f"| `{r['test_id']}` | {r['category']} | `{prov_label}` | `{r.get('model_used') or r.get('model', '-')}` | {res_badge} | `{r['actual_status']}` | "
                f"{r['steps']} | {cost_str} | {r['duration_s']}s | {fail_cat} | {detail_snippet} |"
            )
    else:
        if has_upstream:
            md_lines.extend(
                [
                    "",
                    "## Detailed Results",
                    "",
                    "| Test ID | Category | Result | Actual Status | Steps | Cost ($) | Time (s) | Upstream | Failure Category | Detail |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for r in results:
                res_badge = "PASS" if r["passed"] else "**FAIL**"
                fail_cat = r["failure_category"] or "-"
                detail_snippet = r["checker_detail"].replace("|", "\\|")
                if r.get("has_real_cost"):
                    cost_str = f"${r['cost_usd']:.4f}"
                elif is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider")):
                    r_prov = normalize_provider_name(r.get("provider_used") or r.get("provider"))
                    cost_str = "unverified" if r_prov == "openrouter" else "estimated/unverified"
                else:
                    cost_str = f"${r['cost_usd']:.4f}"
                md_lines.append(
                    f"| `{r['test_id']}` | {r['category']} | {res_badge} | `{r['actual_status']}` | "
                    f"{r['steps']} | {cost_str} | {r['duration_s']}s | `{r.get('upstream_provider') or '-'}` | {fail_cat} | {detail_snippet} |"
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
                if r.get("has_real_cost"):
                    cost_str = f"${r['cost_usd']:.4f}"
                elif is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider")):
                    r_prov = normalize_provider_name(r.get("provider_used") or r.get("provider"))
                    cost_str = "unverified" if r_prov == "openrouter" else "estimated/unverified"
                else:
                    cost_str = f"${r['cost_usd']:.4f}"
                md_lines.append(
                    f"| `{r['test_id']}` | {r['category']} | {res_badge} | `{r['actual_status']}` | "
                    f"{r['steps']} | {cost_str} | {r['duration_s']}s | {fail_cat} | {detail_snippet} |"
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
    has_unverified = is_price_unverified(model_name, provider) or any(
        (not r.get("has_real_cost")) and is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider"))
        for r in results
    )
    is_openrouter = normalize_provider_name(provider) == "openrouter" or any(
        normalize_provider_name(r.get("provider_used") or r.get("provider")) == "openrouter"
        for r in results
    )
    if has_unverified:
        cost_display = "cost: unverified" if is_openrouter else "cost: estimated/unverified"
    else:
        cost_display = f"${summary['avg_cost_usd']:.4f}"

    console.print(
        Panel.fit(
            f"[bold cyan]WebPilot Evaluation Suite{title_suffix}[/bold cyan]\n"
            f"[yellow]Provider:[/yellow] {provider}{' (MIXED)' if is_mixed else ''} | [yellow]Model:[/yellow] {model_name} | [yellow]Commit:[/yellow] {git_commit}\n"
            f"[yellow]Cost:[/yellow] {cost_display} | [yellow]Runs:[/yellow] {summary['total_runs']} | [yellow]LLM Calls:[/yellow] {summary.get('total_llm_calls', 0)} | [yellow]Skipped (Budget):[/yellow] {summary.get('skipped_budget_count', 0)} | [yellow]Rate Limited:[/yellow] {summary.get('rate_limited_count', 0)} | [yellow]Provider Unavailable:[/yellow] {summary.get('provider_unavailable_count', 0)} | [yellow]Infra Errors:[/yellow] {summary.get('infra_errors_count', 0)} | [yellow]Overall Pass Rate:[/yellow] [bold green]{summary['overall_pass_rate_pct']}%[/bold green]",
            border_style="cyan",
        )
    )

    table = Table(title="Test Execution Details", border_style="blue")
    table.add_column("Test ID", style="cyan")
    table.add_column("Category")
    if is_mixed:
        table.add_column("Provider", style="dim")
    has_upstream = any(r.get("upstream_provider") for r in results)
    if has_upstream:
        table.add_column("Upstream", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Result")
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Failure Category", style="magenta")

    for r in results:
        res_str = "[green]PASS[/green]" if r["passed"] else "[bold red]FAIL[/bold red]"
        fail_str = r["failure_category"] if r["failure_category"] else "-"
        if r.get("has_real_cost"):
            r_cost = f"${r['cost_usd']:.4f}"
        elif is_price_unverified(r.get("model_used") or r.get("model"), r.get("provider_used") or r.get("provider")):
            r_prov = normalize_provider_name(r.get("provider_used") or r.get("provider"))
            r_cost = "unverified" if r_prov == "openrouter" else "cost: estimated/unverified"
        else:
            r_cost = f"${r['cost_usd']:.4f}"
        row = [
            r["test_id"],
            r["category"],
        ]
        if is_mixed:
            row.append(r.get("provider_used") or r.get("provider", "-"))
        if has_upstream:
            row.append(str(r.get("upstream_provider") or "-"))
        row.append(str(r.get("model_used") or r.get("model", "-")))
        row.extend([
            res_str,
            r["actual_status"],
            str(r["steps"]),
            str(r.get("llm_calls", r["steps"])),
            r_cost,
            f"{r['duration_s']}s",
            fail_str,
        ])
        table.add_row(*row)

    console.print(table)

    provider_table = Table(title="Performance by Provider", border_style="cyan")
    provider_table.add_column("Provider", style="bold")
    provider_table.add_column("Runs", justify="center")
    provider_table.add_column("Rate Limited", justify="center")
    provider_table.add_column("Provider Unavailable", justify="center")
    provider_table.add_column("Skipped (Budget)", justify="center")
    provider_table.add_column("Infra Errors", justify="center")
    provider_table.add_column("Passed / Eval Total", justify="center")
    provider_table.add_column("Pass Rate", justify="right")
    provider_table.add_column("Avg Cost", justify="right")
    provider_table.add_column("Avg Time", justify="right")

    for p_name, p_data in summary.get("by_provider", {}).items():
        p_norm = normalize_provider_name(p_name)
        p_has_real = any(r.get("has_real_cost") for r in results if normalize_provider_name(r.get("provider_used") or r.get("provider")) == p_norm)
        p_unverified = (not p_has_real) and is_price_unverified(None, p_name)
        if p_unverified:
            p_cost_str = "cost: unverified" if p_norm == "openrouter" else "cost: estimated/unverified"
        else:
            p_cost_str = f"${p_data['avg_cost_usd']:.4f}"
        provider_table.add_row(
            p_name,
            str(p_data["total"]),
            str(p_data.get("rate_limited", 0)),
            str(p_data.get("provider_unavailable", 0)),
            str(p_data.get("skipped_budget", 0)),
            str(p_data.get("infra_errors", 0)),
            f"{p_data['passed']}/{p_data['eval_total']}",
            f"{p_data['pass_rate_pct']}%",
            p_cost_str,
            f"{p_data['avg_duration_s']:.2f}s",
        )
    console.print(provider_table)

    model_table = Table(title="Performance by Model Used", border_style="magenta")
    model_table.add_column("Model Used", style="bold")
    model_table.add_column("Runs", justify="center")
    model_table.add_column("Provider Unavailable", justify="center")
    model_table.add_column("Skipped (Budget)", justify="center")
    model_table.add_column("Infra Errors", justify="center")
    model_table.add_column("Passed / Eval Total", justify="center")
    model_table.add_column("Success Rate", justify="right")
    model_table.add_column("Avg Cost", justify="right")
    model_table.add_column("Avg Time", justify="right")

    for m_name, m_data in summary.get("by_model", {}).items():
        eval_tot = m_data.get("eval_total", m_data["total"] - m_data.get("infra_errors", 0))
        m_has_real = any(r.get("has_real_cost") for r in results if (r.get("model_used") or r.get("model")) == m_name)
        m_unverified = (not m_has_real) and is_price_unverified(m_name)
        if m_unverified:
            m_cost_str = "cost: unverified" if is_openrouter else "cost: estimated/unverified"
        else:
            m_cost_str = f"${m_data['avg_cost_usd']:.4f}"
        model_table.add_row(
            m_name,
            str(m_data["total"]),
            str(m_data.get("provider_unavailable", 0)),
            str(m_data.get("skipped_budget", 0)),
            str(m_data.get("infra_errors", 0)),
            f"{m_data['passed']}/{eval_tot}",
            f"{m_data['pass_rate_pct']}%",
            m_cost_str,
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
        f"[bold]Total LLM Calls:[/] {summary.get('total_llm_calls', 0)}  |  "
        f"[bold]Average Cost:[/] {cost_display}  |  "
        f"[bold]Average Duration:[/] {summary['avg_duration_s']}s  |  "
        f"[bold]Skipped (Budget):[/bold] {summary.get('skipped_budget_count', 0)}  |  "
        f"[bold]Rate Limited:[/] {summary.get('rate_limited_count', 0)}  |  "
        f"[bold]Provider Unavailable:[/] {summary.get('provider_unavailable_count', 0)}  |  "
        f"[bold]Safety Failures:[/] [{'red' if summary['safety_failures_count'] > 0 else 'green'}]{summary['safety_failures_count']}[/]"
    )
    console.print(Panel(metrics_panel, border_style="yellow"))


def main() -> None:
    """CLI entrypoint for running evaluation tests."""
    parser = argparse.ArgumentParser(description="WebPilot Evaluation Runner")
    parser.add_argument("--provider", choices=["anthropic", "gemini", "groq", "openrouter"], default=None, help="LLM Provider (anthropic, gemini, groq, openrouter)")
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    parser.add_argument("--tests", type=str, default=None, help="Comma-separated test IDs to run")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions per test case")
    parser.add_argument("--headed", action="store_true", help="Run browser sessions in headed mode")
    parser.add_argument("--tests-file", type=str, default="evals/tests.json", help="Path to tests.json")
    parser.add_argument("--failover", action="store_true", help="Enable provider failover (default false for evals)")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="Maximum total LLM calls allowed before cleanly stopping suite")

    args = parser.parse_args()

    raw_provider = args.provider or os.getenv("PROVIDER", "anthropic")
    provider = normalize_provider_name(raw_provider)
    model_name = args.model or get_model_for_provider(provider)
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

    # Requirement 3: Print estimate tests x repeat x avg_steps(~8) calls and check daily limit
    num_tests = len(tests_to_run)
    estimated_calls = num_tests * args.repeat * 8
    console.print(
        f"[cyan]Estimated LLM calls: {num_tests} tests x {args.repeat} repeat x ~8 avg_steps = ~{estimated_calls} calls[/cyan]"
    )

    daily_limit_env = f"DAILY_REQUEST_LIMIT_{provider.upper()}"
    daily_limit_val = os.getenv(daily_limit_env)
    if daily_limit_val:
        try:
            daily_limit = int(daily_limit_val.strip())
            if estimated_calls > daily_limit:
                console.print(
                    f"[bold yellow]WARNING: Estimated LLM calls (~{estimated_calls}) exceeds known daily limit of {daily_limit} for {provider} ({daily_limit_env})[/bold yellow]"
                )
        except ValueError:
            pass

    console.print(
        f"[cyan]Loaded {num_tests} test definition(s). Running with repeat={args.repeat}...[/cyan]"
    )

    planned_runs = []
    for test in tests_to_run:
        for rep in range(1, args.repeat + 1):
            planned_runs.append((test, rep))

    results: List[Dict[str, Any]] = []
    consecutive_rate_limits = 0
    aborted_due_to_exhaustion = False
    stopped_due_to_budget = False
    total_llm_calls = 0

    idx = 0
    while idx < len(planned_runs):
        if aborted_due_to_exhaustion:
            break

        test, rep = planned_runs[idx]

        # Check if budget reached before starting run
        if args.max_llm_calls is not None and total_llm_calls >= args.max_llm_calls:
            console.print(
                f"[bold yellow]Budget limit reached ({total_llm_calls} >= {args.max_llm_calls} LLM calls). Stopping suite cleanly.[/bold yellow]"
            )
            stopped_due_to_budget = True
            break

        rep_label = f" (run {rep}/{args.repeat})" if args.repeat > 1 else ""
        console.print(f"[dim]Executing: {test['id']}{rep_label}...[/dim]")

        test_max_steps = int(test.get("max_steps", 12))
        effective_max_steps = test_max_steps
        if args.max_llm_calls is not None:
            remaining_budget = args.max_llm_calls - total_llm_calls
            if remaining_budget < test_max_steps:
                effective_max_steps = max(1, remaining_budget)

        eval_record = run_single_eval(
            test_case=test,
            provider=provider,
            model_name=model_name,
            headed=args.headed,
            failover_enabled=args.failover,
            max_steps_override=effective_max_steps if args.max_llm_calls is not None else None,
        )
        results.append(eval_record)
        idx += 1

        run_calls = eval_record.get("llm_calls", eval_record.get("steps", 0))
        total_llm_calls += run_calls
        budget_str = f" / {args.max_llm_calls}" if args.max_llm_calls is not None else ""
        console.print(
            f"[cyan]LLM calls used: {run_calls} (run) | {total_llm_calls}{budget_str} (running total)[/cyan]"
        )

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

        # Check if budget reached after this run
        if args.max_llm_calls is not None and total_llm_calls >= args.max_llm_calls:
            console.print(
                f"[bold yellow]Budget limit reached ({total_llm_calls} >= {args.max_llm_calls} LLM calls). Stopping suite cleanly.[/bold yellow]"
            )
            stopped_due_to_budget = True
            break

        time.sleep(1.5)

    # Requirement 1: Mark remaining tests skipped_budget and save valid partial report
    if stopped_due_to_budget:
        for rem_test, rem_rep in planned_runs[idx:]:
            skipped_record = {
                "test_id": rem_test["id"],
                "test_name": rem_test.get("name", rem_test["id"]),
                "category": rem_test.get("category", "functional"),
                "passed": False,
                "checker_passed": False,
                "status_matched": False,
                "expected_status": rem_test.get("expected_status", "success"),
                "actual_status": "skipped_budget",
                "steps": 0,
                "llm_calls": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "duration_s": 0.0,
                "failure_category": "skipped_budget",
                "checker_detail": "Skipped due to --max-llm-calls budget limit",
                "summary": "Skipped due to --max-llm-calls budget limit reached",
                "final_url": rem_test.get("start_url", ""),
                "trace_file": None,
                "provider": provider,
                "model": model_name,
                "provider_used": provider,
                "model_used": model_name,
                "failover_happened": False,
                "fallback_used": False,
            }
            results.append(skipped_record)

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
