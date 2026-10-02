"""Browser agent orchestrator coordinating browser sessions, LLM decisions, and guardrails."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from agent.browser import BrowserSession, Observation, to_prompt_text
from agent.guardrails import (
    LoopDetector,
    StepLimiter,
    check_action,
    compute_page_state_hash,
    element_label,
)
from agent.llm import (
    LLMConfigError,
    LLMError,
    LLMInfraError,
    decide,
    estimate_cost,
    get_model_for_provider,
    get_provider_priority,
    is_failover_enabled,
    is_price_unverified,
    is_provider_exhausted,
    is_provider_unavailable_error,
    is_rate_limit_error,
    mark_provider_exhausted,
    normalize_provider_name,
    pick_active_provider,
)

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Structured result returned at the conclusion of an agent execution run."""

    status: str
    steps: int
    total_tokens: int
    estimated_cost_usd: float
    duration_s: float
    final_url: str
    trace_dir: str
    summary: str
    question: Optional[str] = None
    trace_file: Optional[str] = None
    primary_model: Optional[str] = None
    model_used: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    per_provider: Optional[Dict[str, Any]] = None
    provider_used: Optional[str] = None
    failover_happened: bool = False
    failure_category: Optional[str] = None
    llm_calls: int = 0
    has_real_cost: bool = False
    upstream_provider: Optional[str] = None
    navigation_error: Optional[str] = None


@dataclass
class StepRecord:
    """Record of a single step executed during an agent run."""

    step: int
    url: str
    screenshot_path: Optional[str]
    elements_count: int
    page_state_hash: str
    reasoning: str
    action: str
    args: Dict[str, Any]
    result: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    upstream_provider: Optional[str] = None


class Agent:
    """Autonomous web agent executing goals in a browser with safety guardrails."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        browser_session: Optional[BrowserSession] = None,
        headless: bool = True,
        max_steps: int = 12,
        max_wall_time: float = 120.0,
        traces_root: str = "traces",
        failover_enabled: Optional[bool] = None,
    ) -> None:
        self.provider = normalize_provider_name(provider) if provider else None
        self.model_name = model_name

        fb_prov = fallback_provider if fallback_provider is not None else os.getenv("FALLBACK_PROVIDER", "")
        self.fallback_provider: Optional[str] = normalize_provider_name(fb_prov.strip()) if fb_prov.strip() else None

        fb_model = fallback_model_name if fallback_model_name is not None else os.getenv("FALLBACK_MODEL_NAME", "")
        self.fallback_model_name: Optional[str] = fb_model.strip() if fb_model else None

        self.browser_session = browser_session
        self.headless = headless
        self.max_steps = max_steps
        self.max_wall_time = max_wall_time
        self.traces_root = Path(traces_root)
        self._owned_session: bool = False
        self.failover_enabled = failover_enabled if failover_enabled is not None else is_failover_enabled(default=True)

    def _get_or_create_session(self) -> BrowserSession:
        """Retrieves the active session or starts an owned session."""
        if self.browser_session is not None:
            if self.browser_session._page is None:
                self.browser_session.start()
            return self.browser_session

        session = BrowserSession(headless=self.headless)
        session.start()
        self.browser_session = session
        self._owned_session = True
        return session

    def close(self) -> None:
        """Closes the browser session if it was instantiated by the agent."""
        if self.browser_session is not None:
            self.browser_session.close()

    def run(
        self,
        goal: str,
        start_url: str,
        confirm_callback: Optional[Callable[[str], bool]] = None,
        max_steps: Optional[int] = None,
    ) -> RunResult:
        """Executes the agent loop to achieve the given goal starting from start_url.

        Loop order:
            observe -> llm.decide -> guardrails.check_action -> execute -> record trace -> repeat.

        Never raises an exception; all errors are caught and returned as RunResult(status="error").
        The browser session is kept open so callers and evaluators can inspect live state.
        """
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_trace_dir = self.traces_root / run_id
        run_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_json_path = run_trace_dir / "trace.json"

        start_time = time.monotonic()
        total_input_tokens = 0
        total_output_tokens = 0
        step_records: List[StepRecord] = []
        history_entries: List[Dict[str, Any]] = []
        hallucinated_ids = 0
        blocked_attempts = 0

        failover_happened = False
        failure_category: Optional[str] = None

        if self.failover_enabled:
            chosen_prov, chosen_model = pick_active_provider(
                preferred_provider=self.provider,
                preferred_model=self.model_name,
            )
            if not chosen_prov:
                logger.error("All providers in priority list are currently exhausted.")
                final_status = "error"
                failure_category = "rate_limited"
                final_summary = "All providers exhausted due to rate limits."
                trace_data = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "goal": goal,
                    "start_url": start_url,
                    "final_url": start_url,
                    "provider": self.provider or "none",
                    "model": self.model_name or "none",
                    "provider_used": self.provider or "none",
                    "model_used": self.model_name or "none",
                    "failover_happened": False,
                    "status": final_status,
                    "failure_category": failure_category,
                    "summary": final_summary,
                    "duration_s": 0.0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "hallucinated_ids_count": 0,
                    "steps": [],
                }
                try:
                    with open(trace_json_path, "w", encoding="utf-8") as f:
                        json.dump(trace_data, f, indent=2)
                except Exception as exc:
                    logger.error("Failed to write trace file: %s", exc)

                return RunResult(
                    status=final_status,
                    steps=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    duration_s=0.0,
                    final_url=start_url,
                    trace_dir=str(run_trace_dir),
                    summary=final_summary,
                    trace_file=str(trace_json_path),
                    primary_model=self.model_name,
                    model_used=self.model_name or "none",
                    provider_used=self.provider or "none",
                    failover_happened=False,
                    failure_category=failure_category,
                )

            priority_first = get_provider_priority()[0] if get_provider_priority() else "groq"
            configured_primary = self.provider or priority_first
            if normalize_provider_name(chosen_prov) != normalize_provider_name(configured_primary):
                failover_happened = True

            primary_provider = chosen_prov
            primary_model = chosen_model
        else:
            primary_provider = self.provider or os.getenv("PROVIDER", "groq")
            primary_model = self.model_name or get_model_for_provider(primary_provider)

        current_provider = primary_provider
        current_model = primary_model
        fallback_used = False
        fallback_reason: Optional[str] = None

        provider_usage: Dict[str, Dict[str, Any]] = {
            primary_provider: {
                "model": primary_model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        }
        if self.fallback_provider and self.fallback_model_name:
            provider_usage[self.fallback_provider] = {
                "model": self.fallback_model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        effective_max_steps = max_steps if max_steps is not None else self.max_steps
        loop_detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        step_limiter = StepLimiter(max_steps=effective_max_steps, max_wall_time=self.max_wall_time)

        final_status = "failed"
        final_summary = "Run terminated without conclusion."
        asked_question: Optional[str] = None
        current_url = start_url
        total_llm_calls = 0
        has_real_cost = False
        latest_upstream_provider: Optional[str] = None

        try:
            session = self._get_or_create_session()
            navigation_error: Optional[str] = None
            if start_url:
                logger.info("Navigating to start URL: %s", start_url)
                nav_res = session.goto(start_url)
                if not nav_res.ok:
                    navigation_error = nav_res.message
                    logger.warning("Failed navigating to start_url: %s", nav_res.message)

            step_no = 0
            while True:
                limit_verdict = step_limiter.check()
                if limit_verdict.exceeded:
                    if "Time limit" in limit_verdict.reason:
                        final_status = "timeout"
                    else:
                        final_status = "failed"
                    final_summary = limit_verdict.reason
                    break

                step_no += 1

                # 1. Observe current page
                screenshot_filename = f"step_{step_no}.png"
                screenshot_file_path = run_trace_dir / screenshot_filename
                obs = session.observe(screenshot_path=str(screenshot_file_path))
                current_url = obs.url or current_url

                # Compute page state hash for loop detection
                page_state_hash = compute_page_state_hash(obs.url, obs.elements, obs.scroll_info)

                # Format prompt representation
                observation_prompt_text = to_prompt_text(obs)

                # 2. LLM Decision
                llm_start = time.monotonic()
                decision = None
                total_llm_calls += 1
                try:
                    decision = decide(
                        goal=goal,
                        history=history_entries,
                        observation_text=observation_prompt_text,
                        model_name=current_model,
                        provider=current_provider,
                    )
                except LLMConfigError as exc:
                    final_status = "error"
                    final_summary = f"LLM configuration error: {exc}"
                    logger.error("LLM config error (fallback not attempted): %s", exc)
                    break
                except (LLMInfraError, LLMError, Exception) as exc:
                    # Check for 429 / quota / rate limit errors
                    if is_rate_limit_error(exc):
                        mark_provider_exhausted(current_provider)
                        is_step_1 = (step_no == 1 and len(step_records) == 0)

                        if is_step_1 and self.failover_enabled:
                            next_prov, next_model = pick_active_provider()
                            if next_prov is not None:
                                logger.warning(
                                    "Provider '%s' rate limited at step 1. Restarting run on backup provider '%s' (%s).",
                                    current_provider,
                                    next_prov,
                                    next_model,
                                )
                                current_provider = next_prov
                                current_model = next_model
                                failover_happened = True

                                # Restart run on the next provider
                                step_no = 0
                                step_limiter.reset()
                                loop_detector = LoopDetector(tuple_threshold=3, state_threshold=4)
                                step_records.clear()
                                history_entries.clear()
                                if start_url:
                                    session.goto(start_url)
                                continue

                        # Mid-run rate limit or no backup provider available
                        final_status = "error"
                        failure_category = "rate_limited"
                        final_summary = f"Provider '{current_provider}' rate limited: {exc}"
                        logger.error(
                            "Provider '%s' rate limited (step %d, failover=%s): %s",
                            current_provider,
                            step_no,
                            self.failover_enabled,
                            exc,
                        )
                        break

                    can_fallback = (
                        not fallback_used
                        and bool(self.fallback_provider)
                        and bool(self.fallback_model_name)
                    )
                    if can_fallback:
                        logger.warning(
                            "Primary provider '%s' failed with '%s'. Switching to fallback provider '%s' (%s).",
                            current_provider,
                            exc,
                            self.fallback_provider,
                            self.fallback_model_name,
                        )
                        fallback_used = True
                        fallback_reason = str(exc)
                        current_provider = self.fallback_provider
                        current_model = self.fallback_model_name

                        try:
                            total_llm_calls += 1
                            decision = decide(
                                goal=goal,
                                history=history_entries,
                                observation_text=observation_prompt_text,
                                model_name=current_model,
                                provider=current_provider,
                            )
                        except LLMConfigError as fb_exc:
                            final_status = "error"
                            final_summary = f"Fallback configuration error: {fb_exc}"
                            logger.error("Fallback provider config error: %s", fb_exc)
                            break
                        except (LLMInfraError, LLMError, Exception) as fb_exc:
                            final_status = "infra_error"
                            final_summary = (
                                f"Both primary and fallback providers failed. Primary: {fallback_reason}. Fallback: {fb_exc}"
                            )
                            logger.error("Both providers failed: %s", final_summary)
                            break
                    else:
                        if is_provider_unavailable_error(exc):
                            final_status = "error"
                            failure_category = "provider_unavailable"
                            final_summary = f"Provider '{current_provider}' unavailable: {exc}"
                            logger.error(
                                "Provider '%s' unavailable (step %d): %s",
                                current_provider,
                                step_no,
                                exc,
                            )
                        elif fallback_used:
                            final_status = "infra_error"
                            final_summary = f"Fallback provider failed: {exc}"
                        elif isinstance(exc, LLMInfraError) and (bool(self.fallback_provider)):
                            final_status = "infra_error"
                            final_summary = f"Infrastructure error: {exc}"
                        else:
                            final_status = "error"
                            final_summary = f"LLM error: {exc}"
                        logger.error("LLM decision error: %s", exc)
                        break

                llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
                total_input_tokens += decision.input_tokens
                total_output_tokens += decision.output_tokens

                if current_provider not in provider_usage:
                    provider_usage[current_provider] = {
                        "model": current_model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                provider_usage[current_provider]["input_tokens"] += decision.input_tokens
                provider_usage[current_provider]["output_tokens"] += decision.output_tokens
                if getattr(decision, "cost_usd", None) is not None:
                    step_cost = float(decision.cost_usd)
                    has_real_cost = True
                else:
                    if is_price_unverified(current_model, current_provider):
                        step_cost = 0.0
                    else:
                        step_cost = estimate_cost(current_model, decision.input_tokens, decision.output_tokens)
                provider_usage[current_provider]["cost_usd"] = round(
                    provider_usage[current_provider]["cost_usd"] + step_cost, 6
                )

                step_upstream = getattr(decision, "upstream_provider", None)
                if step_upstream:
                    latest_upstream_provider = step_upstream

                action_name = decision.tool_name
                action_args = decision.tool_args
                action_reasoning = decision.reasoning

                # 3. Handle finish action
                if action_name == "finish":
                    is_success = bool(action_args.get("success", False))
                    final_status = "success" if is_success else "failed"
                    final_summary = str(action_args.get("summary", "Task completed via finish tool."))
                    step_records.append(
                        StepRecord(
                            step=step_no,
                            url=obs.url,
                            screenshot_path=str(screenshot_file_path),
                            elements_count=len(obs.elements),
                            page_state_hash=page_state_hash,
                            reasoning=action_reasoning,
                            action=action_name,
                            args=action_args,
                            result=final_summary,
                            input_tokens=decision.input_tokens,
                            output_tokens=decision.output_tokens,
                            latency_ms=llm_latency_ms,
                            upstream_provider=step_upstream,
                        )
                    )
                    break

                # 4. Handle ask_user action
                if action_name == "ask_user":
                    question = str(action_args.get("question", ""))
                    asked_question = question
                    if confirm_callback is None:
                        final_status = "needs_confirmation"
                        final_summary = f"Agent paused for user confirmation: {question}"
                        step_records.append(
                            StepRecord(
                                step=step_no,
                                url=obs.url,
                                screenshot_path=str(screenshot_file_path),
                                elements_count=len(obs.elements),
                                page_state_hash=page_state_hash,
                                reasoning=action_reasoning,
                                action=action_name,
                                args=action_args,
                                result=final_summary,
                                input_tokens=decision.input_tokens,
                                output_tokens=decision.output_tokens,
                                latency_ms=llm_latency_ms,
                                upstream_provider=step_upstream,
                            )
                        )
                        break
                    else:
                        confirmed = confirm_callback(question)
                        if not confirmed:
                            final_status = "blocked"
                            final_summary = f"User denied confirmation for: {question}"
                            step_records.append(
                                StepRecord(
                                    step=step_no,
                                    url=obs.url,
                                    screenshot_path=str(screenshot_file_path),
                                    elements_count=len(obs.elements),
                                    page_state_hash=page_state_hash,
                                    reasoning=action_reasoning,
                                    action=action_name,
                                    args=action_args,
                                    result=final_summary,
                                    input_tokens=decision.input_tokens,
                                    output_tokens=decision.output_tokens,
                                    latency_ms=llm_latency_ms,
                                    upstream_provider=step_upstream,
                                )
                            )
                            break
                        else:
                            action_result_msg = f"User confirmed question: '{question}'."
                            history_entries.append(
                                {
                                    "action": f"ask_user({question})",
                                    "result": "User confirmed. Proceeding.",
                                }
                            )
                            step_records.append(
                                StepRecord(
                                    step=step_no,
                                    url=obs.url,
                                    screenshot_path=str(screenshot_file_path),
                                    elements_count=len(obs.elements),
                                    page_state_hash=page_state_hash,
                                    reasoning=action_reasoning,
                                    action=action_name,
                                    args=action_args,
                                    result=action_result_msg,
                                    input_tokens=decision.input_tokens,
                                    output_tokens=decision.output_tokens,
                                    latency_ms=llm_latency_ms,
                                    upstream_provider=step_upstream,
                                )
                            )
                            step_limiter.step()
                            continue

                # 5. Check for hallucinated element ID
                target_element = None
                if action_name in ("click", "type_text"):
                    target_id = action_args.get("id")
                    if target_id is not None:
                        matched = [el for el in obs.elements if el.id == int(target_id)]
                        if not matched:
                            hallucinated_ids += 1
                            err_msg = f"Invalid element id [{target_id}] does not exist on page."
                            logger.warning(err_msg)
                            history_entries.append(
                                {
                                    "action": f"{action_name}({action_args})",
                                    "result": f"Failed: {err_msg}",
                                }
                            )
                            step_records.append(
                                StepRecord(
                                    step=step_no,
                                    url=obs.url,
                                    screenshot_path=str(screenshot_file_path),
                                    elements_count=len(obs.elements),
                                    page_state_hash=page_state_hash,
                                    reasoning=action_reasoning,
                                    action=action_name,
                                    args=action_args,
                                    result=err_msg,
                                    input_tokens=decision.input_tokens,
                                    output_tokens=decision.output_tokens,
                                    latency_ms=llm_latency_ms,
                                    upstream_provider=step_upstream,
                                )
                            )
                            step_limiter.step()
                            continue
                        target_element = matched[0]

                # 6. Safety Guardrail Check
                target_label = element_label(target_element) if target_element else ""
                press_enter_flag = bool(action_args.get("press_enter", False))
                dest_url = str(action_args.get("url", "")) if action_name == "goto" else obs.url
                guard_verdict = check_action(
                    action=action_name,
                    element=target_label,
                    url=dest_url or obs.url,
                    press_enter=press_enter_flag,
                )

                if not guard_verdict.allowed:
                    blocked_attempts += 1
                    logger.warning("Guardrail blocked action: %s", guard_verdict.reason)
                    history_entries.append(
                        {
                            "action": f"{action_name}({action_args})",
                            "result": (
                                f"BLOCKED: {guard_verdict.reason}. "
                                "You must call ask_user instead before performing this action."
                            ),
                        }
                    )
                    step_records.append(
                        StepRecord(
                            step=step_no,
                            url=obs.url,
                            screenshot_path=str(screenshot_file_path),
                            elements_count=len(obs.elements),
                            page_state_hash=page_state_hash,
                            reasoning=action_reasoning,
                            action=action_name,
                            args=action_args,
                            result=f"BLOCKED: {guard_verdict.reason}",
                            input_tokens=decision.input_tokens,
                            output_tokens=decision.output_tokens,
                            latency_ms=llm_latency_ms,
                            upstream_provider=step_upstream,
                        )
                    )

                    if blocked_attempts >= 2:
                        final_status = "blocked"
                        final_summary = (
                            f"Execution stopped: action blocked 2 times by guardrails ({guard_verdict.reason})."
                        )
                        break
                    step_limiter.step()
                    continue

                # 7. Check Loop Detection with page_state_hash
                loop_verdict = loop_detector.record(
                    url=obs.url,
                    action=action_name,
                    args=action_args,
                    page_state_hash=page_state_hash,
                )
                if loop_verdict.is_loop:
                    final_status = "loop"
                    final_summary = loop_verdict.reason
                    step_records.append(
                        StepRecord(
                            step=step_no,
                            url=obs.url,
                            screenshot_path=str(screenshot_file_path),
                            elements_count=len(obs.elements),
                            page_state_hash=page_state_hash,
                            reasoning=action_reasoning,
                            action=action_name,
                            args=action_args,
                            result=f"LOOP DETECTED: {loop_verdict.reason}",
                            input_tokens=decision.input_tokens,
                            output_tokens=decision.output_tokens,
                            latency_ms=llm_latency_ms,
                            upstream_provider=step_upstream,
                        )
                    )
                    break

                # 8. Execute Browser Action
                action_result_str = ""
                if action_name == "click":
                    res = session.click(id=int(action_args["id"]))
                    action_result_str = res.message
                elif action_name == "type_text":
                    res = session.type_text(
                        id=int(action_args["id"]),
                        text=str(action_args["text"]),
                        press_enter=bool(action_args.get("press_enter", False)),
                    )
                    action_result_str = res.message
                elif action_name == "select_option":
                    res = session.select_option(
                        id=int(action_args["id"]),
                        value_or_label=str(action_args.get("value_or_label", "")),
                    )
                    action_result_str = res.message
                elif action_name == "goto":
                    res = session.goto(url=str(action_args["url"]))
                    action_result_str = res.message
                elif action_name == "scroll":
                    res = session.scroll(direction=str(action_args.get("direction", "down")))
                    action_result_str = res.message
                elif action_name == "go_back":
                    res = session.go_back()
                    action_result_str = res.message
                elif action_name == "wait":
                    res = session.wait(seconds=float(action_args.get("seconds", 2)))
                    action_result_str = res.message
                else:
                    action_result_str = f"Unknown action: {action_name}"

                history_entries.append(
                    {
                        "action": f"{action_name}({action_args})",
                        "result": action_result_str,
                    }
                )

                step_records.append(
                    StepRecord(
                        step=step_no,
                        url=obs.url,
                        screenshot_path=str(screenshot_file_path),
                        elements_count=len(obs.elements),
                        page_state_hash=page_state_hash,
                        reasoning=action_reasoning,
                        action=action_name,
                        args=action_args,
                        result=action_result_str,
                        input_tokens=decision.input_tokens,
                        output_tokens=decision.output_tokens,
                        latency_ms=llm_latency_ms,
                        upstream_provider=step_upstream,
                    )
                )
                step_limiter.step()

        except Exception as exc:
            logger.exception("Unexpected exception in Agent.run(): %s", exc)
            final_status = "error"
            final_summary = f"Unexpected error during execution: {exc}"

        duration_s = round(time.monotonic() - start_time, 2)
        total_tokens = total_input_tokens + total_output_tokens
        cost_usd = round(sum(p["cost_usd"] for p in provider_usage.values()), 6)

        # Write trace.json
        trace_data = {
            "run_id": run_id,
            "goal": goal,
            "start_url": start_url,
            "provider": current_provider,
            "model": current_model,
            "provider_used": current_provider,
            "model_used": current_model,
            "upstream_provider": latest_upstream_provider,
            "failover_happened": failover_happened,
            "primary_model": primary_model,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "per_provider": provider_usage,
            "status": final_status,
            "failure_category": failure_category,
            "summary": final_summary,
            "question": asked_question,
            "total_steps": len(step_records),
            "duration_s": duration_s,
            "total_tokens": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost_usd": cost_usd,
            "hallucinated_ids_count": hallucinated_ids,
            "llm_calls": total_llm_calls,
            "has_real_cost": has_real_cost,
            "navigation_error": navigation_error,
            "steps": [asdict(record) for record in step_records],
        }

        try:
            with open(trace_json_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to write trace file: %s", exc)

        return RunResult(
            status=final_status,
            steps=len(step_records),
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            duration_s=duration_s,
            final_url=current_url,
            trace_dir=str(run_trace_dir),
            summary=final_summary,
            question=asked_question,
            trace_file=str(trace_json_path),
            primary_model=primary_model,
            model_used=current_model,
            provider_used=current_provider,
            failover_happened=failover_happened,
            failure_category=failure_category,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            per_provider=provider_usage,
            llm_calls=total_llm_calls,
            has_real_cost=has_real_cost,
            upstream_provider=latest_upstream_provider,
            navigation_error=navigation_error,
        )


if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    parser = argparse.ArgumentParser(description="WebPilot Autonomous Browser Agent")
    parser.add_argument("--goal", type=str, required=True, help="Goal for the browser agent")
    parser.add_argument("--url", type=str, required=True, help="Starting URL")
    parser.add_argument("--headed", action="store_true", help="Launch browser in headed mode")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (anthropic, gemini, groq, openrouter)")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--fallback-provider", type=str, default=None, help="Fallback LLM Provider (anthropic, gemini, groq, openrouter)")
    parser.add_argument("--fallback-model", type=str, default=None, help="Fallback model name")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum execution steps")

    args = parser.parse_args()
    console = Console()

    console.print(
        Panel.fit(
            f"[bold cyan]WebPilot Browser Agent[/bold cyan]\n"
            f"[yellow]Goal:[/yellow] {args.goal}\n"
            f"[yellow]URL:[/yellow] {args.url}\n"
            f"[yellow]Headed:[/yellow] {args.headed}",
            border_style="cyan",
        )
    )

    agent = Agent(
        provider=args.provider or os.getenv("PROVIDER"),
        model_name=args.model,
        fallback_provider=args.fallback_provider,
        fallback_model_name=args.fallback_model,
        headless=not args.headed,
        max_steps=args.max_steps,
    )

    try:
        result = agent.run(goal=args.goal, start_url=args.url)

        table = Table(title="Execution Summary", border_style="green")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Status", f"[{'green' if result.status == 'success' else 'red'}]{result.status}[/]")
        table.add_row("Primary Model", str(result.primary_model or "-"))
        table.add_row("Model Used", str(result.model_used or "-"))
        table.add_row("Fallback Used", str(result.fallback_used))
        if result.fallback_reason:
            table.add_row("Fallback Reason", str(result.fallback_reason))
        table.add_row("Steps Executed", str(result.steps))
        table.add_row("Duration", f"{result.duration_s}s")
        table.add_row("Total Tokens", str(result.total_tokens))
        table.add_row("Estimated Cost", f"${result.estimated_cost_usd:.4f}")
        table.add_row("Final URL", result.final_url)
        table.add_row("Trace Directory", result.trace_dir)
        table.add_row("Summary", result.summary)
        if result.question:
            table.add_row("Question for User", result.question)

        console.print(table)
    finally:
        agent.close()
