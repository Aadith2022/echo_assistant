"""Patchright browser manager, pinned to a single dedicated thread.

Playwright's synchronous API is thread-affine: objects created on one thread
raise if touched from another, and this codebase is multi-threaded by design.
So the browser lives on one thread it owns for its whole life, and every
operation is submitted as a callable and awaited on a Future - callers never
touch a Playwright object directly.

That constraint is about threads, not concurrency. Fan-out runs one worker
thread per branch, each blocked on its own model calls (~82% of a step) and
each addressing its own tab, while the browser thread keeps serialising the
page operations. `_serve` owns the tab registry; a caller selects a tab with
`use_page`, which sets a thread-local key that `submit` reads.

Patchright's documented stealth configuration is
`launch_persistent_context(user_data_dir=..., channel="chrome",
headless=False, no_viewport=True)` with NO custom user agent or headers. Every
instinct says to run headless with a spoofed UA; both are exactly what modern
fingerprinting detects. Patchright patches CDP leaks and runtime signatures
rather than lying in JavaScript, and adding injections re-introduces the tells
it removed.

Two consequences: there is no `Browser` object and so no `new_context()`, so
one cookie jar is shared by every task and isolation is the Origin Set's job;
and the window is visible, which steals foreground focus during a task.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
_CLOSE_PAGE = object()

# The tab everything uses unless it says otherwise. One page is still the
# normal case; the registry exists for fan-out, not as a new default.
MAIN_PAGE = "main"

# Which tab this thread's page operations address. Thread-local rather than a
# parameter on every call: a branch declares its tab once with `use_page` and
# every existing call underneath resolves to it, leaving `page_actions`
# untouched. Thread-local works because a branch is a thread here.
_local = threading.local()


def current_page_key() -> str:
    return getattr(_local, "page_key", MAIN_PAGE)


@contextmanager
def use_page(key: str):
    """Point this thread's page operations at `key` for the duration.

    Restores the previous key rather than resetting to MAIN_PAGE, so nesting
    behaves and a branch cannot strand the thread that spawned it.
    """
    previous = current_page_key()
    _local.page_key = key
    try:
        yield key
    finally:
        _local.page_key = previous


class BrowserUnavailable(RuntimeError):
    """Raised when the browser cannot be started at all."""


class BrowserEngine:
    """Owns one Patchright persistent context on one dedicated thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._lock = threading.Lock()
        self._running = False
        # Published so `submit` can widen its timeout. Owned by the browser
        # thread and only read elsewhere, so a plain int is enough.
        self._open_pages = 1

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Launch the browser if it isn't already up. Idempotent and lazy.

        The lock is held across the readiness wait, deliberately. `_running` is
        set deep inside `_worker` only after Chrome has launched, so releasing
        early leaves seconds in which the engine is starting but does not look
        started - and a second caller arriving there launches a second Chrome on
        the same user-data-dir, which Chrome refuses, after replacing
        `self._queue` and orphaning the first caller's work.

        Serialising startup costs nothing: it happens once and every caller
        wants the same browser. `_worker` never takes this lock.
        """
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return

            # A second caller arriving mid-startup falls through to the wait
            # below and rides the launch already in progress.
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._start_error = None
                self._queue = queue.Queue()
                self._thread = threading.Thread(
                    target=self._worker, name="browser-engine", daemon=True
                )
                self._thread.start()

            # A cold Chrome profile is slow to launch.
            if not self._ready.wait(timeout=config.BROWSER_TIMEOUT_SECONDS * 2):
                raise BrowserUnavailable("Browser did not start within the timeout.")
            if self._start_error is not None:
                raise BrowserUnavailable(f"Browser failed to start: {self._start_error}")

    def shutdown(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._queue.put((_SHUTDOWN, None, MAIN_PAGE))
        thread = self._thread
        if thread:
            thread.join(timeout=15)

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._thread and self._thread.is_alive())

    @property
    def open_pages(self) -> int:
        return self._open_pages

    # --- submission ----------------------------------------------------------

    def submit(
        self,
        fn: Callable[[Any], Any],
        timeout: float | None = None,
        page_key: str | None = None,
    ) -> Any:
        """Run `fn(page)` on the browser thread and return its result.

        Which page depends on the calling thread - see `use_page`; callers that
        do not care get MAIN_PAGE. Exceptions raised inside `fn` propagate
        unchanged.
        """
        self.start()
        future: Future = Future()
        self._queue.put((fn, future, page_key or current_page_key()))

        # One operation at a time, so with several tabs live a caller waits for
        # its own operation plus whatever is queued ahead of it. Unscaled, this
        # would manufacture timeouts in the slowest branch.
        budget = timeout or config.BROWSER_TIMEOUT_SECONDS * 2 * max(1, self._open_pages)
        return future.result(timeout=budget)

    def close_page(self, key: str) -> None:
        """Close a fan-out tab. Never closes MAIN_PAGE."""
        if key == MAIN_PAGE or not self.is_running:
            return
        future: Future = Future()
        self._queue.put((_CLOSE_PAGE, future, key))
        try:
            future.result(timeout=config.BROWSER_TIMEOUT_SECONDS)
        except Exception:
            logger.debug("Closing page %s failed; continuing", key, exc_info=True)

    # --- worker --------------------------------------------------------------

    def _worker(self) -> None:
        try:
            from patchright.sync_api import sync_playwright
        except ImportError as e:
            self._start_error = RuntimeError(
                "patchright is not installed. Run: pip install patchright "
                "&& python -m patchright install chrome"
            )
            logger.error("Patchright import failed: %s", e)
            self._ready.set()
            return

        playwright = None
        context = None
        try:
            playwright = sync_playwright().start()
            context = self._launch(playwright)
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(config.BROWSER_TIMEOUT_SECONDS * 1000)

            self._running = True
            self._ready.set()
            logger.info("Browser engine ready")

            self._serve(context, page)

        except BaseException as e:  # noqa: BLE001 - reported to the caller
            self._start_error = e
            logger.exception("Browser engine failed")
            self._ready.set()
        finally:
            self._running = False
            for closer in (
                lambda: context.close() if context else None,
                lambda: playwright.stop() if playwright else None,
            ):
                try:
                    closer()
                except Exception:
                    logger.debug("Browser teardown step failed", exc_info=True)
            logger.info("Browser engine stopped")

    def _launch(self, playwright):
        launch_kwargs: dict[str, Any] = {}
        if config.RESIDENTIAL_PROXY_URL:
            launch_kwargs["proxy"] = {"server": config.RESIDENTIAL_PROXY_URL}

        # A self-hosted stealth browser driven over CDP, for sites whose bot
        # management defeats local Patchright.
        if config.STEEL_BROWSER_ENDPOINT:
            logger.info("Connecting to Steel Browser at %s", config.STEEL_BROWSER_ENDPOINT)
            browser = playwright.chromium.connect_over_cdp(config.STEEL_BROWSER_ENDPOINT)
            return browser.contexts[0] if browser.contexts else browser.new_context()

        import os

        os.makedirs(config.BROWSER_USER_DATA_DIR, exist_ok=True)
        logger.info(
            "Launching %s (profile=%s, headless=%s)",
            config.BROWSER_CHANNEL,
            config.BROWSER_USER_DATA_DIR,
            config.BROWSER_HEADLESS,
        )
        return playwright.chromium.launch_persistent_context(
            user_data_dir=config.BROWSER_USER_DATA_DIR,
            channel=config.BROWSER_CHANNEL,
            headless=config.BROWSER_HEADLESS,
            # A viewport that doesn't match the window is itself a fingerprint.
            no_viewport=True,
            # ...so the window must be a normal desktop size. Chrome defaults
            # to ~800x600, at which responsive sites serve a compact layout and
            # hide controls the agent needs.
            args=[f"--window-size={config.BROWSER_WINDOW_SIZE}"],
            # Deliberately NOT set: user_agent, extra_http_headers, init
            # scripts. See the module docstring.
            **launch_kwargs,
        )

    def _serve(self, context, primary) -> None:
        """Pump the command queue until shutdown or idle timeout.

        Owns the page registry. Tabs are created lazily and only ever touched
        from this thread, which is what satisfies Playwright's thread-affinity
        rule while several worker threads submit work concurrently.
        """
        pages = {MAIN_PAGE: primary}

        while True:
            try:
                item = self._queue.get(timeout=config.BROWSER_IDLE_SHUTDOWN_SECONDS)
            except queue.Empty:
                logger.info(
                    "Browser idle for %ss; shutting down to release memory and the "
                    "Chrome profile lock",
                    config.BROWSER_IDLE_SHUTDOWN_SECONDS,
                )
                return

            fn, future, key = item
            if fn is _SHUTDOWN:
                return

            if fn is _CLOSE_PAGE:
                page = pages.pop(key, None)
                self._open_pages = len(pages)
                try:
                    if page is not None and not page.is_closed():
                        page.close()
                except Exception:
                    logger.debug("Could not close page %s", key, exc_info=True)
                if future.set_running_or_notify_cancel():
                    future.set_result(True)
                continue

            if future.set_running_or_notify_cancel():
                try:
                    page = pages.get(key)
                    # Recreate a closed tab as well as a missing one - the
                    # window is visible, so the user can close one by hand.
                    if page is None or page.is_closed():
                        page = context.new_page()
                        page.set_default_timeout(config.BROWSER_TIMEOUT_SECONDS * 1000)
                        pages[key] = page
                        self._open_pages = len(pages)
                        logger.info("Opened browser tab %r (%d open)", key, len(pages))
                    future.set_result(fn(page))
                except BaseException as e:  # noqa: BLE001 - handed to the caller
                    future.set_exception(e)


# --- behavioural mimicry -----------------------------------------------------
# Anti-bot systems profile timing and pointer movement, not just fingerprints.


def clear_site_data() -> None:
    """Drop cookies and site storage, so a task starts without inherited state.

    Only called when BROWSER_CLEAR_SITE_DATA is on.
    """
    def _clear(page):
        page.context.clear_cookies()
        try:
            page.evaluate(
                "() => { try { localStorage.clear(); sessionStorage.clear(); } catch (e) {} }"
            )
        except Exception:
            # about:blank and opaque origins have no storage to clear.
            pass
        return True

    try:
        engine.submit(_clear)
        logger.info("Cleared cookies and site storage for this task")
    except Exception:
        logger.warning("Could not clear site data; continuing", exc_info=True)


def random_delay(minimum: float | None = None, maximum: float | None = None) -> None:
    low = config.BROWSER_MIN_ACTION_DELAY if minimum is None else minimum
    high = config.BROWSER_MAX_ACTION_DELAY if maximum is None else maximum
    if high > 0:
        time.sleep(random.uniform(low, max(low, high)))


def human_mouse_path(
    page, x: float, y: float, steps: int = 12, jitter: float = 12.0
) -> None:
    """Move the pointer to (x, y) along a curved, uneven path.

    A straight-line jump in one event is a strong bot signal. This traces a
    quadratic Bezier with per-step jitter and an ease-out.
    """
    try:
        start = getattr(page, "_echo_mouse", (0.0, 0.0))
        sx, sy = start
        cx = (sx + x) / 2 + random.uniform(-jitter * 3, jitter * 3)
        cy = (sy + y) / 2 + random.uniform(-jitter * 3, jitter * 3)

        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)  # smoothstep: slow start, slow finish
            px = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t**2 * x
            py = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t**2 * y
            if i < steps:
                px += random.uniform(-jitter / 4, jitter / 4)
                py += random.uniform(-jitter / 4, jitter / 4)
            page.mouse.move(px, py)
            time.sleep(random.uniform(0.008, 0.025))

        page._echo_mouse = (x, y)
    except Exception:
        # A nicety; never fail an action over it.
        logger.debug("human_mouse_path failed; continuing", exc_info=True)


engine = BrowserEngine()
