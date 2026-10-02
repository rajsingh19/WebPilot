"""Comparison tool comparing performance across evaluation run results."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_result_file(file_path: str) -> Dict[str, Any]:
    """Loads and validates an evaluation result JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "summary" not in data or "results" not in data:
        raise ValueError(f"Invalid evaluation result schema in: {file_path}")

    return data


def format_category_rate(summary: Dict[str, Any], category: str) -> str:
    """Formats pass rate string for a given category."""
    by_cat = summary.get("by_category", {})
    if category in by_cat:
        cat_info = by_cat[category]
        rate = cat_info.get("pass_rate_pct", 0.0)
        passed = cat_info.get("passed", 0)
        total = cat_info.get("total", 0)
        color = "green" if rate == 100.0 else ("yellow" if rate >= 50.0 else "red")
        return f"[{color}]{rate}%[/] ({passed}/{total})"
    return "[dim]N/A[/dim]"


def compute_group_metrics(group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes summary metrics for a subset of results belonging to a specific (provider, model)."""
    total = len(group_results)
    infra = sum(1 for r in group_results if r.get("actual_status") == "infra_error")
    rate_limited = sum(1 for r in group_results if r.get("failure_category") == "rate_limited")
    provider_unavailable = sum(1 for r in group_results if r.get("failure_category") == "provider_unavailable")
    eval_total = total - infra - rate_limited - provider_unavailable
    passed = sum(
        1 for r in group_results
        if r.get("passed")
        and r.get("actual_status") != "infra_error"
        and r.get("failure_category") not in ("rate_limited", "provider_unavailable")
    )
    overall_pct = (passed / eval_total * 100.0) if eval_total > 0 else 0.0

    categories = ["functional", "safety", "language", "robustness"]
    by_category: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        cat_items = [r for r in group_results if r.get("category") == cat]
        c_infra = sum(1 for r in cat_items if r.get("actual_status") == "infra_error")
        c_rl = sum(1 for r in cat_items if r.get("failure_category") == "rate_limited")
        c_un = sum(1 for r in cat_items if r.get("failure_category") == "provider_unavailable")
        c_eval = len(cat_items) - c_infra - c_rl - c_un
        c_passed = sum(
            1 for r in cat_items
            if r.get("passed")
            and r.get("actual_status") != "infra_error"
            and r.get("failure_category") not in ("rate_limited", "provider_unavailable")
        )
        c_rate = (c_passed / c_eval * 100.0) if c_eval > 0 else 0.0
        by_category[cat] = {
            "total": len(cat_items),
            "eval_total": c_eval,
            "passed": c_passed,
            "pass_rate_pct": round(c_rate, 1),
        }

    avg_cost = sum(r.get("cost_usd", 0.0) for r in group_results) / total if total > 0 else 0.0
    avg_duration = sum(r.get("duration_s", 0.0) for r in group_results) / total if total > 0 else 0.0
    safety_fails = sum(1 for r in group_results if r.get("category") == "safety" and not r.get("passed"))

    return {
        "total_runs": total,
        "eval_total": eval_total,
        "infra_errors_count": infra,
        "rate_limited_count": rate_limited,
        "provider_unavailable_count": provider_unavailable,
        "passed_runs": passed,
        "overall_pass_rate_pct": round(overall_pct, 1),
        "by_category": by_category,
        "avg_cost_usd": avg_cost,
        "avg_duration_s": avg_duration,
        "safety_failures_count": safety_fails,
    }


def compare_runs(files: List[str]) -> None:
    """Compares metrics across multiple evaluation run result JSON files."""
    if len(files) < 2:
        console.print("[bold red]Error: evals.compare requires at least 2 result JSON files to compare.[/bold red]")
        sys.exit(1)

    runs_data: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = load_result_file(fp)
            runs_data.append({"path": fp, "data": data})
        except Exception as exc:
            console.print(f"[bold red]Failed loading '{fp}': {exc}[/bold red]")
            sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]WebPilot Evaluation Comparison[/bold cyan]\n"
            f"[yellow]Comparing {len(runs_data)} evaluation result datasets[/yellow]",
            border_style="cyan",
        )
    )

    table = Table(title="Model & Provider Performance Comparison", border_style="blue")
    table.add_column("Provider / Model", style="bold cyan")
    table.add_column("Commit", style="dim")
    table.add_column("Runs", justify="right")
    table.add_column("Overall %", justify="right")
    table.add_column("Functional", justify="center")
    table.add_column("Safety", justify="center")
    table.add_column("Language", justify="center")
    table.add_column("Robustness", justify="center")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Avg Time", justify="right")
    table.add_column("Safety Fails", justify="right")

    for item in runs_data:
        d = item["data"]
        commit = str(d.get("git_commit", "-"))[:7]
        results = d.get("results", [])

        # Group results by (provider, model) so mixed runs are never merged into one row
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in results:
            prov = r.get("provider") or r.get("provider_used") or d.get("provider", "unknown")
            mdl = r.get("model") or r.get("model_used") or d.get("model", "unknown")
            key = (prov, mdl)
            groups.setdefault(key, []).append(r)

        if not groups:
            # Fallback if results array is empty
            key = (d.get("provider", "unknown"), d.get("model", "unknown"))
            groups[key] = []

        for (prov, mdl), group_items in groups.items():
            if len(groups) == 1 and group_items and "summary" in d and d.get("provider") == prov and d.get("model") == mdl:
                summary = d["summary"]
            else:
                summary = compute_group_metrics(group_items)

            overall_pct = summary.get("overall_pass_rate_pct", 0.0)
            overall_color = "green" if overall_pct >= 80.0 else ("yellow" if overall_pct >= 50.0 else "red")
            overall_str = f"[{overall_color}]{overall_pct}%[/]"

            safety_fails = summary.get("safety_failures_count", 0)
            safety_color = "red" if safety_fails > 0 else "green"

            table.add_row(
                f"{prov} / {mdl}",
                commit,
                str(summary.get("total_runs", len(group_items))),
                overall_str,
                format_category_rate(summary, "functional"),
                format_category_rate(summary, "safety"),
                format_category_rate(summary, "language"),
                format_category_rate(summary, "robustness"),
                f"${summary.get('avg_cost_usd', 0.0):.4f}",
                f"{summary.get('avg_duration_s', 0.0):.1f}s",
                f"[{safety_color}]{safety_fails}[/]",
            )

    console.print(table)


def main() -> None:
    """CLI entrypoint for evals.compare."""
    parser = argparse.ArgumentParser(
        description="Compare 2 or more evaluation result JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to 2+ evaluation result JSON files (e.g. results/run1.json results/run2.json)",
    )

    args = parser.parse_args()
    compare_runs(args.files)


if __name__ == "__main__":
    main()
