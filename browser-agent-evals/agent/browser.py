"""Playwright-based browser session and DOM observation module."""

from dataclasses import dataclass, field
import logging
from typing import Any, List, Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

logger = logging.getLogger(__name__)


@dataclass
class InteractiveElement:
    """Represents an interactive DOM element detected on the page."""

    id: int
    tag: str
    type: str = ""
    text: str = ""
    value: str = ""
    disabled: bool = False
    checked: bool = False
    options: List[str] = field(default_factory=list)


@dataclass
class Observation:
    """Snapshot observation of the current browser page state."""

    url: str
    title: str
    elements: List[InteractiveElement] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    page_text: str = ""
    scroll_info: str = "Scroll: 0% of page"


@dataclass
class ActionResult:
    """Result of an action performed on the browser session."""

    ok: bool
    message: str


JS_EXTRACT_ELEMENTS = """
() => {
    document.querySelectorAll('[data-agent-id]').forEach(el => {
        el.removeAttribute('data-agent-id');
    });

    function isVisible(el) {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
        if (el.closest('[aria-hidden="true"]')) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return false;
        const winWidth = window.innerWidth || document.documentElement.clientWidth;
        if (rect.right <= 0 || rect.left >= winWidth) return false;
        return true;
    }

    function findHeading(el) {
        const container = el.closest('.inventory_item, .cart_item, tr, li, [class*="item"], [class*="card"], [class*="product"], [class*="row"], article, section');
        if (!container) return '';
        const headingEl = container.querySelector('h1, h2, h3, h4, h5, h6, .inventory_item_name, [class*="item_name"], [class*="title"], [class*="heading"], [class*="name"], th, strong');
        if (headingEl && headingEl !== el) {
            return (headingEl.innerText || headingEl.textContent || '').trim().replace(/\\s+/g, ' ');
        }
        return '';
    }

    function extractText(el) {
        let text = '';
        const tag = el.tagName.toLowerCase();
        if (tag === 'select') {
            const selectedOpt = el.options && el.selectedIndex >= 0 ? el.options[el.selectedIndex] : null;
            text = selectedOpt ? (selectedOpt.text || selectedOpt.value || '').trim() : '';
        } else if (tag === 'input' && (el.type === 'button' || el.type === 'submit')) {
            text = el.value || '';
        } else {
            text = el.innerText || el.textContent || '';
        }
        text = text.replace(/[\\r\\n\\t]+/g, ' ').replace(/\\s+/g, ' ').trim();

        // Links whose text is empty or only a number: build label from aria-label/title/href
        if (tag === 'a' || el.getAttribute('role') === 'link' || el.getAttribute('role') === 'button') {
            const isNumericOnly = /^\\d+$/.test(text);
            if (!text || isNumericOnly) {
                const ariaLabel = (el.getAttribute('aria-label') || '').trim();
                const title = (el.getAttribute('title') || '').trim();
                const href = (el.getAttribute('href') || '').trim();
                const badge = el.querySelector('.shopping_cart_badge, [class*="badge"]');
                const badgeCount = badge ? badge.textContent.trim() : (isNumericOnly ? text : '');

                let parts = [];
                if (ariaLabel) {
                    parts.push(ariaLabel);
                } else if (title) {
                    parts.push(title);
                } else if (el.classList.contains('shopping_cart_link') || (href && href.includes('cart'))) {
                    const countStr = badgeCount ? ` (${badgeCount} item${badgeCount === '1' ? '' : 's'})` : '';
                    parts.push(`Cart link${countStr}`);
                } else if (isNumericOnly) {
                    parts.push(`Link (${text})`);
                }

                if (href && href !== '#' && !href.startsWith('javascript:')) {
                    parts.push(href);
                }

                if (parts.length > 0) {
                    text = parts.join(' ').trim();
                }
            }
        }

        if (!text && el.getAttribute('aria-label')) {
            text = el.getAttribute('aria-label').trim();
        }
        if (!text && el.placeholder) {
            text = el.placeholder.trim();
        }
        if (!text && el.name) {
            text = el.name.trim();
        }
        if (!text && el.title) {
            text = el.title.trim();
        }

        // Disambiguate generic labels: "Add to cart", "Remove", "Details", numbers, empty
        const lower = text.toLowerCase();
        const isGeneric = (
            lower === 'add to cart' ||
            lower === 'remove' ||
            lower === 'details' ||
            lower === 'view details' ||
            /^\\d+$/.test(text) ||
            !text
        );
        if (isGeneric) {
            const heading = findHeading(el);
            if (heading) {
                text = text ? `${text} (${heading})` : heading;
            }
        }

        return text.substring(0, 80);
    }

    const selector = 'a, button, input, select, textarea, [role="button"]';
    const elements = Array.from(document.querySelectorAll(selector));
    const items = [];
    let nextId = 1;

    for (const el of elements) {
        if (!isVisible(el)) continue;

        const id = nextId++;
        el.setAttribute('data-agent-id', String(id));

        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || (el.getAttribute('role') === 'button' ? 'button' : '');
        const text = extractText(el);

        let value = '';
        let options = [];
        if (tag === 'select') {
            options = Array.from(el.options || []).map(opt => (opt.text || opt.value || '').trim());
            value = el.value || '';
        } else if (tag === 'input' || tag === 'textarea') {
            if (el.type === 'password') {
                value = el.value ? '••••••' : '';
            } else if (el.value !== undefined && el.value !== null) {
                value = String(el.value).substring(0, 80);
            }
        }

        const disabled = Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true');
        const checked = Boolean(
            el.checked ||
            el.getAttribute('aria-checked') === 'true' ||
            el.getAttribute('aria-selected') === 'true'
        );

        items.push({
            id: id,
            tag: tag,
            type: type,
            text: text,
            value: value,
            options: options,
            disabled: disabled,
            checked: checked
        });

        if (items.length >= 80) {
            break;
        }
    }

    // Page text extraction (up to 600 chars of visible headings, alerts, errors, messages, badges)
    const pageTextSelector = 'h1, h2, h3, h4, h5, h6, [role="alert"], [class*="error"], [data-test*="error"], [class*="message"], [data-test*="message"], [class*="flash"], [data-test*="flash"], [class*="title"], [data-test*="title"], .shopping_cart_badge, [class*="cart_badge"], [data-test*="badge"]';
    const ptElements = Array.from(document.querySelectorAll(pageTextSelector));
    const ptTexts = [];
    const seenTexts = new Set();
    for (const el of ptElements) {
        if (!isVisible(el)) continue;
        let txt = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
        if (txt && !seenTexts.has(txt)) {
            seenTexts.add(txt);
            ptTexts.push(txt);
        }
    }
    const pageText = ptTexts.join(' | ').substring(0, 600);

    // Scroll position calculation
    const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const scrollPercent = Math.round((window.scrollY / maxScroll) * 100);
    const scrollInfo = `Scroll: ${Math.min(100, Math.max(0, scrollPercent))}% of page`;

    return {
        items: items,
        pageText: pageText,
        scrollInfo: scrollInfo
    };
}
"""


def to_prompt_text(observation: Observation) -> str:
    """Renders the page observation as compact, human-readable prompt text."""
    lines = [
        f"URL: {observation.url}",
        f"Title: {observation.title}",
    ]
    if observation.scroll_info:
        lines.append(observation.scroll_info)

    if observation.page_text:
        lines.append(f"Page text: {observation.page_text}")

    lines.append("Interactive Elements:")
    if not observation.elements:
        lines.append("  (none)")
        return "\n".join(lines)

    for el in observation.elements:
        parts = [f"[{el.id}]", el.tag]
        if el.type and el.type != el.tag:
            parts.append(f"(type={el.type})")
        if el.text:
            parts.append(f"'{el.text}'")
        if el.value:
            parts.append(f"value='{el.value}'")
        if el.options:
            parts.append(f"options={el.options}")
        if el.checked:
            parts.append("[checked]")
        if el.disabled:
            parts.append("[disabled]")
        lines.append(" ".join(parts))

    return "\n".join(lines)


class BrowserSession:
    """Manages a Playwright browser session and synchronous interactions."""

    def __init__(self, headless: bool = True) -> None:
        self.headless: bool = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def start(self) -> "BrowserSession":
        """Starts the browser session if not already running."""
        if self._page is not None:
            return self

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})
            self._page = self._context.new_page()
            logger.info("Browser session started (headless=%s).", self.headless)
        except Exception as exc:
            logger.error("Failed to start browser session: %s", exc)
            self.close()
            raise
        return self

    def close(self) -> None:
        """Closes browser page, context, browser, and playwright instance."""
        if self._context:
            try:
                self._context.close()
            except Exception as exc:
                logger.debug("Error closing context: %s", exc)
            self._context = None

        if self._browser:
            try:
                self._browser.close()
            except Exception as exc:
                logger.debug("Error closing browser: %s", exc)
            self._browser = None

        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as exc:
                logger.debug("Error stopping playwright: %s", exc)
            self._playwright = None

        self._page = None
        logger.info("Browser session closed.")

    def __enter__(self) -> "BrowserSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _wait_network_idle(self, timeout_ms: int = 5000) -> None:
        """Waits for network idle state without raising an exception on timeout."""
        if not self._page:
            return
        try:
            self._page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as exc:
            logger.debug("Network idle wait timeout or notice: %s", exc)
        try:
            self._page.wait_for_timeout(300)
        except Exception:
            pass

    def observe(self, screenshot_path: Optional[str] = None) -> Observation:
        """Observes the current page state, extracting visible interactive elements."""
        if not self._page:
            logger.warning("observe() called with no active page.")
            return Observation(url="", title="", elements=[])

        try:
            url = self._page.url or ""
            title = self._page.title() or ""

            eval_result = self._page.evaluate(JS_EXTRACT_ELEMENTS)
            if isinstance(eval_result, dict):
                raw_elements = eval_result.get("items", [])
                page_text = eval_result.get("pageText", "")
                scroll_info = eval_result.get("scrollInfo", "")
            else:
                raw_elements = eval_result or []
                page_text = ""
                scroll_info = ""

            elements: List[InteractiveElement] = []
            for item in raw_elements:
                elements.append(
                    InteractiveElement(
                        id=int(item["id"]),
                        tag=str(item.get("tag", "")),
                        type=str(item.get("type", "")),
                        text=str(item.get("text", "")),
                        value=str(item.get("value", "")),
                        options=list(item.get("options", [])),
                        disabled=bool(item.get("disabled", False)),
                        checked=bool(item.get("checked", False)),
                    )
                )

            saved_screenshot: Optional[str] = None
            if screenshot_path:
                try:
                    self._page.screenshot(path=screenshot_path)
                    saved_screenshot = screenshot_path
                except Exception as exc:
                    logger.warning("Failed to capture screenshot to %s: %s", screenshot_path, exc)

            return Observation(
                url=url,
                title=title,
                elements=elements,
                page_text=page_text,
                scroll_info=scroll_info,
                screenshot_path=saved_screenshot,
            )
        except Exception as exc:
            logger.error("Failed to observe page: %s", exc)
            return Observation(url=self._page.url or "", title="", elements=[])

    def click(self, id: int) -> ActionResult:
        """Clicks an element by its assigned sequential agent id."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        try:
            locator = self._page.locator(f'[data-agent-id="{id}"]')
            if locator.count() == 0:
                return ActionResult(ok=False, message=f"Element with id [{id}] not found on page.")
            locator.first.click(timeout=5000)
            self._wait_network_idle()
            return ActionResult(ok=True, message=f"Clicked element [{id}].")
        except Exception as exc:
            logger.error("Error clicking element [%s]: %s", id, exc)
            return ActionResult(ok=False, message=f"Failed to click element [{id}]: {exc}")

    def type_text(self, id: int, text: str, press_enter: bool = False) -> ActionResult:
        """Fills text into an element by its assigned agent id."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        try:
            locator = self._page.locator(f'[data-agent-id="{id}"]')
            if locator.count() == 0:
                return ActionResult(ok=False, message=f"Element with id [{id}] not found on page.")
            locator.first.fill(text, timeout=5000)
            if press_enter:
                locator.first.press("Enter", timeout=5000)
            self._wait_network_idle()
            suffix = " and pressed Enter" if press_enter else ""
            return ActionResult(ok=True, message=f"Typed text into element [{id}]{suffix}.")
        except Exception as exc:
            logger.error("Error typing into element [%s]: %s", id, exc)
            return ActionResult(ok=False, message=f"Failed to type into element [{id}]: {exc}")

    def select_option(self, id: int, value_or_label: str) -> ActionResult:
        """Selects an option from a dropdown element by value, then label."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        try:
            locator = self._page.locator(f'[data-agent-id="{id}"]')
            if locator.count() == 0:
                return ActionResult(ok=False, message=f"Element with id [{id}] not found on page.")
            try:
                locator.first.select_option(value=value_or_label, timeout=3000)
            except Exception:
                locator.first.select_option(label=value_or_label, timeout=3000)
            self._wait_network_idle()
            return ActionResult(ok=True, message=f"Selected option '{value_or_label}' on element [{id}].")
        except Exception as exc:
            logger.error("Error selecting option on element [%s]: %s", id, exc)
            return ActionResult(ok=False, message=f"Failed to select option '{value_or_label}' on element [{id}]: {exc}")

    def goto(self, url: str) -> ActionResult:
        """Navigates to the specified URL."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        target_url = url
        if not target_url.startswith(("http://", "https://", "about:", "file://")):
            target_url = "https://" + target_url

        last_error = None
        for attempt in range(1, 3):
            try:
                self._page.goto(target_url, timeout=45000, wait_until="domcontentloaded")
                self._wait_network_idle()
                return ActionResult(ok=True, message=f"Navigated to {target_url}.")
            except Exception as exc:
                last_error = exc
                is_timeout = "timeout" in str(exc).lower() or isinstance(exc, TimeoutError)
                if is_timeout and attempt == 1:
                    logger.warning(
                        "Navigation to %s timed out on attempt 1. Retrying navigation once...",
                        target_url,
                    )
                    continue
                logger.error("Error navigating to %s (attempt %d/2): %s", target_url, attempt, exc)
                break

        msg = (
            f"Failed to navigate to {target_url} after retry: {last_error}"
            if is_timeout
            else f"Failed to navigate to {target_url}: {last_error}"
        )
        return ActionResult(ok=False, message=msg)

    def scroll(self, direction: str) -> ActionResult:
        """Scrolls the page in the specified direction ('up', 'down', 'top', 'bottom')."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        normalized = direction.strip().lower()
        try:
            if normalized == "down":
                self._page.evaluate("window.scrollBy(0, window.innerHeight * 0.8);")
            elif normalized == "up":
                self._page.evaluate("window.scrollBy(0, -window.innerHeight * 0.8);")
            elif normalized == "top":
                self._page.evaluate("window.scrollTo(0, 0);")
            elif normalized == "bottom":
                self._page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            else:
                return ActionResult(
                    ok=False,
                    message=f"Unsupported scroll direction '{direction}'. Use 'up', 'down', 'top', or 'bottom'.",
                )

            self._wait_network_idle()
            return ActionResult(ok=True, message=f"Scrolled {normalized}.")
        except Exception as exc:
            logger.error("Error scrolling %s: %s", direction, exc)
            return ActionResult(ok=False, message=f"Failed to scroll {direction}: {exc}")

    def go_back(self) -> ActionResult:
        """Navigates back to the previous page in history."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        try:
            self._page.go_back(timeout=5000)
            self._wait_network_idle()
            return ActionResult(ok=True, message="Navigated back.")
        except Exception as exc:
            logger.error("Error navigating back: %s", exc)
            return ActionResult(ok=False, message=f"Failed to go back: {exc}")

    def wait(self, seconds: float) -> ActionResult:
        """Waits for the specified duration (up to 5 seconds)."""
        if not self._page:
            return ActionResult(ok=False, message="Browser session is not running.")

        if seconds < 0 or seconds > 5:
            return ActionResult(
                ok=False,
                message=f"Wait duration must be between 0 and 5 seconds, got {seconds}.",
            )

        try:
            self._page.wait_for_timeout(seconds * 1000)
            self._wait_network_idle()
            return ActionResult(ok=True, message=f"Waited for {seconds:.2f}s.")
        except Exception as exc:
            logger.error("Error during wait: %s", exc)
            return ActionResult(ok=False, message=f"Failed to wait: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Launching BrowserSession demo on https://www.saucedemo.com ...")

    with BrowserSession(headless=True) as session:
        nav_result = session.goto("https://www.saucedemo.com")
        print(f"Action: {nav_result.message} (ok={nav_result.ok})")

        obs = session.observe()
        print("\nObservation compact representation:")
        print(to_prompt_text(obs))
