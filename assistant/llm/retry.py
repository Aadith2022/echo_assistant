"""Retrying model calls that the service failed to answer.

A browser task is hundreds of model calls, so an unretried transient is the
single most likely way a long task dies.

Deliberately narrow: only failures where the service did not answer. A
schema-validation failure is NOT transient - that is the Dual-LLM hard gate
rejecting output, and quietly having another go is the "route around the guard"
move the pattern exists to prevent. Nor is a content filter, an auth error, or
a bad request.

Classified by the SDK's exception TYPE rather than by wording: a substring list
matched against exception text goes stale, and each time it does a dropped
connection scores as the agent confidently lying. Substring matching survives
only as a fallback for exceptions raised outside those classes.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Under the SDK's private `_gaos` package, so it may move. If it does, the
# substring fallback still works and the classification degrades rather than
# breaking.
try:  # pragma: no cover - import shape depends on the installed SDK
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIStatusError

    _CONNECTION_ERRORS: tuple[type[BaseException], ...] = (APIConnectionError,)
    _STATUS_ERRORS: tuple[type[BaseException], ...] = (APIStatusError,)
except Exception:  # noqa: BLE001
    _CONNECTION_ERRORS = ()
    _STATUS_ERRORS = ()

try:  # pragma: no cover
    import httpx

    _CONNECTION_ERRORS = _CONNECTION_ERRORS + (
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    )
except Exception:  # noqa: BLE001
    pass

# These plus any 5xx may succeed on a second attempt. Other 4xx means we asked
# for something the service will refuse just as firmly next time.
_RETRYABLE_STATUS = {408, 429}

# Last resort. Kept short: the type checks above are the real rule, and a long
# list here is the failure mode this module moved away from.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "deadline",
    "unavailable",
    "connection",
    "resource_exhausted",
)


def is_transient(exc: BaseException) -> bool:
    """Is this the service failing to answer, rather than a real refusal?"""
    if _CONNECTION_ERRORS and isinstance(exc, _CONNECTION_ERRORS):
        return True

    if _STATUS_ERRORS and isinstance(exc, _STATUS_ERRORS):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status in _RETRYABLE_STATUS or status >= 500
        return False

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500

    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def call_with_retry(fn: Callable[[], T], *, label: str, attempts: int = 2) -> T:
    """Run `fn`, retrying only transient failures.

    No backoff: these are minute-long client timeouts rather than rate-limit
    bursts, so the delay has already happened.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not is_transient(e):
                raise
            logger.warning(
                "%s call failed transiently (%s); retrying (%d/%d)",
                label,
                type(e).__name__,
                attempt + 2,
                attempts,
            )
    raise AssertionError("unreachable")  # pragma: no cover
