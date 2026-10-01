"""Unit tests for agent guardrails, safety patterns, loop detection, and step limits."""

import time
import pytest

from agent.guardrails import (
    COMPILED_RISKY_PATTERNS,
    RISKY_PATTERNS,
    GuardrailVerdict,
    LimitVerdict,
    LoopDetector,
    LoopVerdict,
    StepLimiter,
    check_action,
    compute_page_state_hash,
    element_label,
)


class TestRiskyPatterns:
    """Tests for regex pattern matching on English and Hindi/Hinglish terms."""

    @pytest.mark.parametrize(
        "text,expected_match",
        [
            ("Pay now", True),
            ("Click here to pay", True),
            ("BUY NOW", True),
            ("Please place order", True),
            ("Finish setup", True),
            ("Confirm order immediately", True),
            ("Send message", True),
            ("Submit registration", True),
            ("Delete item", True),
            ("Purchase license", True),
            # Hindi / Hinglish
            ("Paisa bhejo", True),
            ("Bhugtan karein", True),
            ("Abhi kharido", True),
            ("Abhi order karo", True),
            ("BHUGTAN", True),
            ("KHARIDO", True),
            # Devanagari script
            ("भुगतान करें", True),
            ("अभी खरीदो", True),
            ("संदेश भेजो", True),
            ("ऑर्डर करो", True),
        ],
    )
    def test_risky_patterns_positive(self, text: str, expected_match: bool):
        matched = any(pattern.search(text) for pattern in COMPILED_RISKY_PATTERNS)
        assert matched == expected_match

    @pytest.mark.parametrize(
        "text",
        [
            "Sendai city guide",        # Contains 'Sendai', not 'send'
            "Finnish architecture",     # Contains 'Finnish', not 'finish'
            "Playback controls",        # Contains 'Playback', not 'pay'
            "Payer information",        # Contains 'Payer', not 'pay'
            "Submittal document",       # Contains 'Submittal', not 'submit'
            "Delhi tourism overview",   # Hindi-adjacent safe text
            "Bhejona transport",        # Transliterated Hindi false positive
            "Kharidari guide",          # 'Kharidari' instead of 'kharido'
        ],
    )
    def test_false_positive_cases(self, text: str):
        matched = any(pattern.search(text) for pattern in COMPILED_RISKY_PATTERNS)
        assert not matched, f"Unexpected match on non-risky text: '{text}'"


class TestElementLabelAndCheckAction:
    """Tests for element_label extraction and check_action safety evaluations."""

    def test_input_submit_login_allowed_and_place_order_blocked(self):
        """Input type=submit with value 'Login' must be ALLOWED, button 'Place order' BLOCKED."""
        # 1. Input element with type="submit" and value="Login"
        login_input = {
            "tag": "input",
            "type": "submit",
            "value": "Login",
            "text": "",
        }
        assert element_label(login_input) == "Login"
        verdict_login = check_action(action="click", element=login_input, url="https://example.com/login")
        assert verdict_login.allowed, f"Expected Login to be allowed, got: {verdict_login.reason}"

        # 2. Button with text "Place order"
        order_button = {
            "tag": "button",
            "type": "button",
            "text": "Place order",
        }
        assert element_label(order_button) == "Place order"
        verdict_order = check_action(action="click", element=order_button, url="https://example.com/checkout")
        assert not verdict_order.allowed, "Expected Place order button to be blocked"
        assert "place order" in verdict_order.reason.lower()

    def test_send_feedback_flagged(self):
        verdict = check_action(
            action="click",
            element_text="Send feedback",
            url="https://example.com/support",
        )
        assert not verdict.allowed
        assert "send" in verdict.reason.lower()

    def test_sendai_city_guide_allowed(self):
        verdict = check_action(
            action="click",
            element_text="Sendai city guide",
            url="https://example.com/travel",
        )
        assert verdict.allowed
        assert "allowed" in verdict.reason.lower()

    @pytest.mark.parametrize(
        "element_text,url",
        [
            ("Pay $49.99", "https://checkout.stripe.com"),
            ("Bhugtan karein", "https://paytm.com/checkout"),
            ("Abhi kharido", "https://flipkart.com/item"),
            ("Order karo", "https://zomato.com/cart"),
            ("Delete this address", "https://example.com/profile"),
        ],
    )
    def test_click_risky_elements_blocked(self, element_text: str, url: str):
        verdict = check_action("click", element_text=element_text, url=url)
        assert not verdict.allowed

    def test_risky_url_blocked_even_with_benign_text(self):
        verdict = check_action(
            action="click",
            element_text="Continue",
            url="https://store.com/checkout/pay",
        )
        assert not verdict.allowed
        assert "pay" in verdict.reason.lower()

    def test_type_without_enter_allowed(self):
        verdict = check_action(
            action="type_text",
            element_text="Search pay rates",
            url="https://example.com",
            press_enter=False,
        )
        assert verdict.allowed
        assert "not a click or type-with-enter" in verdict.reason.lower()

    def test_type_with_enter_risky_blocked(self):
        verdict = check_action(
            action="type_text",
            element_text="Submit query",
            url="https://example.com",
            press_enter=True,
        )
        assert not verdict.allowed
        assert "submit" in verdict.reason.lower()

    def test_non_interactive_actions_allowed(self):
        assert check_action("scroll", element_text="pay now").allowed
        assert check_action("wait", element_text="delete all").allowed
        assert check_action("goto", url="https://example.com/pay").allowed

    def test_action_as_dict(self):
        action_dict = {
            "tool_name": "type_text",
            "tool_args": {"id": 1, "text": "foo", "press_enter": True},
        }
        verdict = check_action(action_dict, element_text="Delete user")
        assert not verdict.allowed


class TestLoopDetector:
    """Tests for LoopDetector page_state_hash tracking and stagnation."""

    def test_scroll_down_changing_page_state_is_not_loop(self):
        """Scrolling down 3 times with changing page state is NOT a loop."""
        detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        url = "https://example.com/infinite-scroll"
        action = "scroll"
        args = {"direction": "down"}

        # Simulate 3 scrolls, each resulting in new elements / new page state hash
        v1 = detector.record(url, action, args, page_state_hash="state_page_1")
        assert not v1.is_loop

        v2 = detector.record(url, action, args, page_state_hash="state_page_2")
        assert not v2.is_loop

        v3 = detector.record(url, action, args, page_state_hash="state_page_3")
        assert not v3.is_loop, "Scrolling with changing page state must not be flagged as a loop"

    def test_clicking_same_button_3_times_identical_state_is_loop(self):
        """Clicking the same button 3 times with identical page state IS a loop."""
        detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        url = "https://example.com/broken-page"
        action = "click"
        args = {"id": 7}
        state_hash = "state_same_elements_abc"

        v1 = detector.record(url, action, args, page_state_hash=state_hash)
        assert not v1.is_loop

        v2 = detector.record(url, action, args, page_state_hash=state_hash)
        assert not v2.is_loop

        v3 = detector.record(url, action, args, page_state_hash=state_hash)
        assert v3.is_loop
        assert "occurred 3 times" in v3.reason

    def test_stagnation_detection_same_page_state_4_times(self):
        """Flags a loop if the exact same page_state_hash is seen 4 times in a row."""
        detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        url = "https://example.com/form"
        state_hash = "static_form_state"

        # 3 different actions on the same unchanged page state
        for i in range(1, 4):
            v = detector.record(url, f"action_{i}", {"id": i}, page_state_hash=state_hash)
            assert not v.is_loop, f"Should not flag loop on step {i}"

        # 4th action on unchanged page state triggers stagnation loop
        v4 = detector.record(url, "action_4", {"id": 4}, page_state_hash=state_hash)
        assert v4.is_loop
        assert "4 consecutive steps" in v4.reason

    def test_page_state_change_resets_stagnation(self):
        detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        url = "https://site.com"

        for i in range(1, 4):
            assert not detector.record(url, f"act_{i}", page_state_hash="state_A").is_loop

        # State changes to state_B
        v = detector.record(url, "act_4", page_state_hash="state_B")
        assert not v.is_loop
        assert detector.consecutive_state_count == 1

    def test_compute_page_state_hash(self):
        elements1 = [{"id": 1, "text": "Home"}, {"id": 2, "text": "Contact"}]
        elements2 = [{"id": 1, "text": "Home"}, {"id": 2, "text": "Contact"}]
        elements3 = [{"id": 1, "text": "Home"}, {"id": 2, "text": "About"}]

        h1 = compute_page_state_hash("https://example.com", elements1)
        h2 = compute_page_state_hash("https://example.com", elements2)
        h3 = compute_page_state_hash("https://example.com", elements3)

        assert h1 == h2
        assert h1 != h3

    def test_typing_into_4_fields_values_changing_not_stagnation(self):
        """Typing into 4 different fields one after another (values changing) is NOT flagged as stagnation."""
        detector = LoopDetector(tuple_threshold=3, state_threshold=4)
        url = "https://example.com/checkout"

        # Step 1: type first name
        h1 = compute_page_state_hash(
            url,
            [
                {"id": 1, "text": "First Name", "value": "Jane"},
                {"id": 2, "text": "Last Name", "value": ""},
                {"id": 3, "text": "Email", "value": ""},
                {"id": 4, "text": "Zip Code", "value": ""},
            ],
        )
        v1 = detector.record(url, "type_text", {"id": 1, "text": "Jane"}, page_state_hash=h1)
        assert not v1.is_loop

        # Step 2: type last name
        h2 = compute_page_state_hash(
            url,
            [
                {"id": 1, "text": "First Name", "value": "Jane"},
                {"id": 2, "text": "Last Name", "value": "Doe"},
                {"id": 3, "text": "Email", "value": ""},
                {"id": 4, "text": "Zip Code", "value": ""},
            ],
        )
        v2 = detector.record(url, "type_text", {"id": 2, "text": "Doe"}, page_state_hash=h2)
        assert not v2.is_loop

        # Step 3: type email
        h3 = compute_page_state_hash(
            url,
            [
                {"id": 1, "text": "First Name", "value": "Jane"},
                {"id": 2, "text": "Last Name", "value": "Doe"},
                {"id": 3, "text": "Email", "value": "jane@example.com"},
                {"id": 4, "text": "Zip Code", "value": ""},
            ],
        )
        v3 = detector.record(url, "type_text", {"id": 3, "text": "jane@example.com"}, page_state_hash=h3)
        assert not v3.is_loop

        # Step 4: type zip code
        h4 = compute_page_state_hash(
            url,
            [
                {"id": 1, "text": "First Name", "value": "Jane"},
                {"id": 2, "text": "Last Name", "value": "Doe"},
                {"id": 3, "text": "Email", "value": "jane@example.com"},
                {"id": 4, "text": "Zip Code", "value": "94105"},
            ],
        )
        v4 = detector.record(url, "type_text", {"id": 4, "text": "94105"}, page_state_hash=h4)
        assert not v4.is_loop, "Typing into 4 fields with changing values must not be flagged as stagnation"

    def test_ticking_checkbox_changes_hash(self):
        """Ticking a checkbox modifies the computed page_state_hash."""
        url = "https://example.com/settings"
        unchecked_elements = [
            {"id": 1, "text": "Subscribe to newsletter", "checked": False},
            {"id": 2, "text": "Accept terms", "checked": False},
        ]
        checked_elements = [
            {"id": 1, "text": "Subscribe to newsletter", "checked": True},
            {"id": 2, "text": "Accept terms", "checked": False},
        ]

        h_unchecked = compute_page_state_hash(url, unchecked_elements)
        h_checked = compute_page_state_hash(url, checked_elements)

        assert h_unchecked != h_checked, "Ticking a checkbox must produce a different page state hash"

    def test_value_truncation_to_40_chars(self):
        """Values past 40 characters are truncated when hashing page state."""
        url = "https://example.com"
        el_40 = [{"id": 1, "text": "Input", "value": "A" * 40}]
        el_50 = [{"id": 1, "text": "Input", "value": "A" * 50}]
        el_diff = [{"id": 1, "text": "Input", "value": "B" * 40}]

        h_40 = compute_page_state_hash(url, el_40)
        h_50 = compute_page_state_hash(url, el_50)
        h_diff = compute_page_state_hash(url, el_diff)

        assert h_40 == h_50, "Values past 40 characters must be truncated"
        assert h_40 != h_diff

    def test_detector_reset(self):
        detector = LoopDetector(tuple_threshold=2)
        detector.record("https://site.com", "click", {"id": 1}, page_state_hash="h1")
        detector.reset()

        v = detector.record("https://site.com", "click", {"id": 1}, page_state_hash="h1")
        assert not v.is_loop


class TestStepLimiter:
    """Tests for StepLimiter step and duration caps."""

    def test_step_count_limit(self):
        limiter = StepLimiter(max_steps=3, max_wall_time=100.0)

        assert not limiter.step().exceeded
        assert not limiter.step().exceeded

        v3 = limiter.step()
        assert v3.exceeded
        assert "3/3 steps" in v3.reason
        assert v3.steps == 3

    def test_wall_time_limit(self):
        limiter = StepLimiter(max_steps=100, max_wall_time=0.05)
        time.sleep(0.06)

        verdict = limiter.check()
        assert verdict.exceeded
        assert "Time limit reached" in verdict.reason

    def test_limiter_reset(self):
        limiter = StepLimiter(max_steps=2)
        limiter.step()
        limiter.step()
        assert limiter.check().exceeded

        limiter.reset()
        assert not limiter.check().exceeded
        assert limiter.steps_taken == 0
