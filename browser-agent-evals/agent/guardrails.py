"""Guardrails and safety checks for browser agent actions."""

from dataclasses import dataclass
import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Case-insensitive regex patterns for high-risk irreversible actions (English, Hindi, Hinglish)
RISKY_PATTERNS: List[str] = [
    # English keywords
    r"\bpay\b",
    r"\bbuy\s+now\b",
    r"\bplace\s+order\b",
    r"\bfinish\b",
    r"\bconfirm\s+order\b",
    r"\bsend\b",
    r"\bsubmit\b",
    r"\bdelete\b",
    r"\bpurchase\b",
    # Hindi / Hinglish transliterated
    r"\bbhejo\b",
    r"\bbhugtan\b",
    r"\bkharido\b",
    r"\border\s+karo\b",
    # Devanagari script equivalents
    r"भुगतान",
    r"खरीदो",
    r"भेजो",
    r"ऑर्डर\s+करो",
]

COMPILED_RISKY_PATTERNS: List[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE) for pattern in RISKY_PATTERNS
]

GOTO_BLOCKED_PATH_PATTERNS: List[str] = [
    "checkout-complete",
    "order-complete",
    "/pay",
    "/payment",
    "/confirm",
]


def is_goto_url_blocked(target_url: str) -> Optional[str]:
    """Checks whether a goto destination URL path matches any blocked patterns.

    Blocked patterns:
        - checkout-complete (anywhere in path)
        - order-complete (anywhere in path)
        - /pay (as path segment only: /pay, /pay/, /checkout/pay, but NOT /payroll or /payment-methods-help)
        - /payment (blocked explicitly as a segment or html path)
        - /confirm (blocked as segment or html path)

    Allowed examples:
        - /cart.html
        - /inventory.html
        - /payroll
        - /payment-methods-help
    """
    clean = (target_url or "").strip().lower()
    if not clean:
        return None

    parsed = urlparse(clean)
    path = parsed.path
    if not path:
        path = clean

    normalized_path = path if path.startswith("/") else f"/{path}"

    if "checkout-complete" in normalized_path or "checkout-complete" in clean:
        return "checkout-complete"
    if "order-complete" in normalized_path or "order-complete" in clean:
        return "order-complete"

    # /pay must match as a path segment only (e.g. /pay, /pay/, /foo/pay, /foo/pay/)
    # Must NOT match /payroll or /payment-methods-help
    if re.search(r"(?:^|/)pay(?:/|$)", normalized_path):
        return "/pay"

    # /payment must be blocked explicitly
    if re.search(r"(?:^|/)payment(?:/|\.html|$)", normalized_path):
        return "/payment"

    # /confirm must be blocked
    if re.search(r"(?:^|/)confirm(?:/|\.html|$)", normalized_path) or "/confirm" in normalized_path:
        return "/confirm"

    return None


@dataclass
class GuardrailVerdict:
    """Verdict indicating whether an action passes safety guardrails."""

    allowed: bool
    reason: str


@dataclass
class LoopVerdict:
    """Verdict indicating whether a repetitive loop or stagnation is detected."""

    is_loop: bool
    reason: str


@dataclass
class LimitVerdict:
    """Verdict indicating whether step or time limits have been exceeded."""

    exceeded: bool
    reason: str
    steps: int
    elapsed_seconds: float


def element_label(element: Any) -> str:
    """Extracts ONLY the element's visible or accessible label.

    Priority order:
        1. Visible text (text, innerText, textContent)
        2. Accessibility label (aria-label)
        3. Placeholder (placeholder)
        4. Current or initial value (value)

    CRITICAL SAFETY RULE:
        Tag names (e.g. 'button', 'input') and input types (e.g. 'submit', 'button')
        must NEVER be included in the extracted label. This prevents false positives
        such as blocking an innocent login button like <input type="submit" value="Login">
        simply because its HTML type attribute happens to be 'submit'.
    """
    if element is None:
        return ""
    if isinstance(element, str):
        return element.strip()
    if isinstance(element, dict):
        for key in ("text", "aria_label", "aria-label", "placeholder", "value"):
            val = element.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return ""

    # Object or dataclass with attributes (such as InteractiveElement)
    for attr in ("text", "aria_label", "placeholder", "value"):
        val = getattr(element, attr, None)
        if val and str(val).strip():
            return str(val).strip()

    return ""


def compute_page_state_hash(
    url: str,
    elements: Any = None,
    scroll_info: Optional[str] = None,
) -> str:
    """Computes a stable hash representing current page state.

    Includes URL, scroll bucket (if present), element IDs, visible labels,
    truncated current values (up to 40 chars), checked states, and disabled flags.
    """
    clean_url = (url or "").strip()
    parts = [clean_url]
    if scroll_info:
        match = re.search(r"(\d+)%", str(scroll_info))
        if match:
            pct = int(match.group(1))
            bucket = (pct // 10) * 10
            parts.append(f"scroll_bucket:{bucket}%")
        else:
            parts.append(str(scroll_info).strip())
    if elements:
        for el in elements:
            if isinstance(el, dict):
                el_id = str(el.get("id", ""))
                lbl = element_label(el)
                raw_val = el.get("value", "")
                chk = bool(el.get("checked", False))
                dis = bool(el.get("disabled", False))
            else:
                el_id = str(getattr(el, "id", ""))
                lbl = element_label(el)
                raw_val = getattr(el, "value", "")
                chk = bool(getattr(el, "checked", False))
                dis = bool(getattr(el, "disabled", False))

            val_trunc = str(raw_val or "")[:40]
            parts.append(f"{el_id}:{lbl}:{val_trunc}:{chk}:{dis}")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def check_action(
    action: Any,
    element: Any = None,
    url: Optional[str] = "",
    press_enter: Optional[bool] = None,
    element_text: Any = None,
) -> GuardrailVerdict:
    """Evaluates whether a browser action is safe to execute.

    Evaluates ONLY the element's visible label (text/aria-label/placeholder/value).
    Tag/type names (e.g. 'submit') are never evaluated.

    Blocks click and type-with-enter actions where the target element label or URL
    matches any risky pattern.

    Args:
        action: Tool/action name (str), dict, or Decision object.
        element: Element object, dict, or label string.
        url: Current or destination URL.
        press_enter: Optional boolean override for Enter keypress.
        element_text: Backward-compatible alias for `element`.

    Returns:
        GuardrailVerdict with allowed=False if risky, allowed=True otherwise.
    """
    action_name = ""
    is_press_enter = False
    dest_url = (url or "").strip()

    if isinstance(action, dict):
        action_name = str(action.get("tool_name") or action.get("name") or action.get("action") or "").lower()
        args = action.get("tool_args") or action.get("args") or {}
        if press_enter is not None:
            is_press_enter = press_enter
        elif isinstance(args, dict):
            is_press_enter = bool(args.get("press_enter", False))
        if not dest_url and isinstance(args, dict):
            dest_url = str(args.get("url") or "").strip()
    elif hasattr(action, "tool_name"):
        action_name = str(getattr(action, "tool_name", "")).lower()
        args = getattr(action, "tool_args", {})
        if press_enter is not None:
            is_press_enter = press_enter
        elif isinstance(args, dict):
            is_press_enter = bool(args.get("press_enter", False))
        if not dest_url and isinstance(args, dict):
            dest_url = str(args.get("url") or "").strip()
    elif isinstance(action, str):
        action_name = action.strip().lower()
        if action_name in ("type_with_enter", "type_and_enter"):
            is_press_enter = True
            action_name = "type_text"
        else:
            is_press_enter = bool(press_enter)
    else:
        action_name = str(action).lower()
        is_press_enter = bool(press_enter)

    if action_name == "select_option":
        return GuardrailVerdict(allowed=True, reason="Select option action allowed.")

    # Evaluate click, type-with-enter, or goto actions
    is_click = action_name == "click"
    is_type_enter = action_name in ("type_text", "type") and is_press_enter
    is_goto = action_name == "goto"

    if not (is_click or is_type_enter or is_goto):
        return GuardrailVerdict(
            allowed=True,
            reason=f"Action '{action_name}' is not a click or type-with-enter or goto action.",
        )

    if is_goto:
        blocked_pattern = is_goto_url_blocked(dest_url)
        if blocked_pattern:
            logger.warning(
                "Guardrail triggered on goto URL path: pattern '%s' matched '%s'",
                blocked_pattern,
                dest_url,
            )
            return GuardrailVerdict(
                allowed=False,
                reason=f"Blocked risky goto navigation: URL path matches '{blocked_pattern}'.",
            )
        return GuardrailVerdict(allowed=True, reason="Goto action allowed.")

    target_element = element if element is not None else element_text
    clean_label = element_label(target_element)
    clean_url = (url or "").strip()

    # Check element visible label against risky patterns
    for pattern in COMPILED_RISKY_PATTERNS:
        match = pattern.search(clean_label)
        if match:
            matched_str = match.group()
            logger.warning(
                "Guardrail triggered on element label: pattern '%s' matched '%s'",
                pattern.pattern,
                matched_str,
            )
            return GuardrailVerdict(
                allowed=False,
                reason=f"Blocked risky action: matched '{matched_str}' in element label '{clean_label}'.",
            )

    # Check URL against risky patterns
    for pattern in COMPILED_RISKY_PATTERNS:
        match = pattern.search(clean_url)
        if match:
            matched_str = match.group()
            logger.warning(
                "Guardrail triggered on URL: pattern '%s' matched '%s'",
                pattern.pattern,
                matched_str,
            )
            return GuardrailVerdict(
                allowed=False,
                reason=f"Blocked risky action: matched '{matched_str}' in URL '{clean_url}'.",
            )

    return GuardrailVerdict(allowed=True, reason="Action allowed.")


class LoopDetector:
    """Detects repetitive action loops and page state stagnation across navigation steps."""

    def __init__(self, tuple_threshold: int = 3, state_threshold: int = 4) -> None:
        self.tuple_threshold: int = tuple_threshold
        self.state_threshold: int = state_threshold
        self.history: List[Tuple[str, str, str, str]] = []
        self.consecutive_state_count: int = 0
        self.last_state_hash: Optional[str] = None

    def reset(self) -> None:
        """Resets the detector's tracked history."""
        self.history.clear()
        self.consecutive_state_count = 0
        self.last_state_hash = None

    def record_and_check(
        self,
        url: str,
        action: str,
        args: Any = None,
        page_state_hash: Optional[str] = None,
    ) -> LoopVerdict:
        """Records an action step and checks if an execution loop has occurred.

        Flags a loop if:
        1. The exact same (url, action, args, page_state_hash) appears `tuple_threshold` times (default 3).
        2. The exact same page_state_hash is seen `state_threshold` times consecutively (default 4).
        """
        clean_url = (url or "").strip()
        clean_action = (action or "").strip().lower()
        clean_hash = (page_state_hash or "").strip()

        if isinstance(args, dict):
            serialized_args = json.dumps(args, sort_keys=True)
        elif args is None:
            serialized_args = "{}"
        else:
            serialized_args = str(args)

        entry = (clean_url, clean_action, serialized_args, clean_hash)
        self.history.append(entry)

        # 1. Check exact (url, action, args, page_state_hash) tuple repetition
        tuple_count = self.history.count(entry)
        if tuple_count >= self.tuple_threshold:
            return LoopVerdict(
                is_loop=True,
                reason=(
                    f"Repetitive loop detected: action '{clean_action}' with args {serialized_args} "
                    f"at URL '{clean_url}' with page state '{clean_hash}' has occurred {tuple_count} times."
                ),
            )

        # 2. Check consecutive same page_state_hash stagnation (4 times in a row)
        if clean_hash:
            if clean_hash == self.last_state_hash:
                self.consecutive_state_count += 1
            else:
                self.last_state_hash = clean_hash
                self.consecutive_state_count = 1

            if self.consecutive_state_count >= self.state_threshold:
                return LoopVerdict(
                    is_loop=True,
                    reason=(
                        f"Stagnation loop detected: page state '{clean_hash}' remained unchanged "
                        f"for {self.consecutive_state_count} consecutive steps."
                    ),
                )

        return LoopVerdict(is_loop=False, reason="No loop detected.")

    def record(
        self,
        url: str,
        action: str,
        args: Any = None,
        page_state_hash: Optional[str] = None,
    ) -> LoopVerdict:
        """Alias for record_and_check."""
        return self.record_and_check(
            url=url, action=action, args=args, page_state_hash=page_state_hash
        )


class StepLimiter:
    """Enforces execution constraints on maximum step count and wall-clock duration."""

    def __init__(self, max_steps: int = 12, max_wall_time: float = 120.0) -> None:
        self.max_steps: int = max_steps
        self.max_wall_time: float = max_wall_time
        self.steps_taken: int = 0
        self.start_time: float = time.monotonic()

    def reset(self) -> None:
        """Resets the limiter counters and wall-clock timer."""
        self.steps_taken = 0
        self.start_time = time.monotonic()

    def step(self) -> LimitVerdict:
        """Increments the step counter and checks limit thresholds."""
        self.steps_taken += 1
        return self.check()

    def check(self) -> LimitVerdict:
        """Evaluates whether current steps or elapsed time exceed configured limits."""
        elapsed = time.monotonic() - self.start_time

        if self.steps_taken >= self.max_steps:
            return LimitVerdict(
                exceeded=True,
                reason=f"Step limit reached: {self.max_steps}/{self.max_steps} steps executed.",
                steps=self.max_steps,
                elapsed_seconds=round(elapsed, 2),
            )

        if elapsed >= self.max_wall_time:
            return LimitVerdict(
                exceeded=True,
                reason=f"Time limit reached: elapsed {elapsed:.1f}s exceeds maximum {self.max_wall_time:.1f}s.",
                steps=self.steps_taken,
                elapsed_seconds=round(elapsed, 2),
            )

        return LimitVerdict(
            exceeded=False,
            reason="Within execution limits.",
            steps=self.steps_taken,
            elapsed_seconds=round(elapsed, 2),
        )


__all__ = [
    "RISKY_PATTERNS",
    "GuardrailVerdict",
    "LoopVerdict",
    "LimitVerdict",
    "element_label",
    "compute_page_state_hash",
    "check_action",
    "LoopDetector",
    "StepLimiter",
]
