"""Human confirmation for actions that cannot be undone.

Confirmation is a registered callback, not a `print()` and an `input()`. A
browser task can run on the engine's worker thread or a voice listener thread
while the REPL is already blocked in `input()` on the main thread, and a second
`input()` does not queue - the two race for stdin. Whatever owns the user
interface installs the handler: `main.py` installs a terminal one, a desktop UI
would install a dialog, and a headless run installs one that always denies.

HIGH is a yes/no. CRITICAL requires typing a specific word, because a reflexive
"y" should not be able to authorise a payment.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import config
from guardrails.audit_log import log_event
from guardrails.risk_classifier import CRITICAL, HIGH

logger = logging.getLogger(__name__)


@dataclass
class ConfirmationRequest:
    action: str
    description: str
    tier: str
    detail: str = ""

    @property
    def requires_typed_response(self) -> bool:
        return self.tier == CRITICAL

    @property
    def keyword(self) -> str:
        """The word the user must type to authorise a CRITICAL action."""
        return "CONFIRM"


# Signature: (request) -> bool. Returning False denies.
ConfirmationHandler = Callable[[ConfirmationRequest], bool]

_handler: ConfirmationHandler | None = None
_handler_lock = threading.Lock()


def set_handler(handler: ConfirmationHandler | None) -> None:
    """Register the UI that asks the user. Called once, at startup."""
    global _handler
    with _handler_lock:
        _handler = handler


def terminal_handler(request: ConfirmationRequest) -> bool:
    """Blocking terminal prompt. Only safe on the thread that owns stdin."""
    print(f"\n[confirm] {request.description}")
    if request.detail:
        print(f"          {request.detail}")

    if request.requires_typed_response:
        print(f"          This cannot be undone. Type {request.keyword} to proceed.")
        answer = input("          > ").strip()
        return answer == request.keyword

    answer = input("          Proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def deny_handler(request: ConfirmationRequest) -> bool:
    """Refuse everything: an agent that cannot ask must not assume permission."""
    logger.warning("No confirmation UI available; denying %s", request.action)
    return False


def confirm(action: str, description: str, tier: str, detail: str = "") -> bool:
    """Ask the user to approve an action. Returns True only on explicit yes."""
    if tier not in (HIGH, CRITICAL):
        return True
    if not config.CONFIRM_HIGH_RISK:
        # Logged: "the user turned the safety off" is what an audit trail is for.
        log_event("confirmation_skipped", action=action, tier=tier, reason="CONFIRM_HIGH_RISK=false")
        return True

    request = ConfirmationRequest(action=action, description=description, tier=tier, detail=detail)

    with _handler_lock:
        handler = _handler or deny_handler

    _notify(f"Confirmation needed: {description[:70]}")

    try:
        approved = bool(handler(request))
    except Exception:
        # A UI that throws is not consent.
        logger.exception("Confirmation handler failed; treating as denied")
        approved = False

    log_event(
        "confirmation",
        action=action,
        tier=tier,
        description=description,
        approved=approved,
    )
    return approved


def _notify(message: str) -> None:
    try:
        from vision.overlay import overlay

        overlay.show_toast(message, level="warning")
    except Exception:
        logger.debug("Confirmation toast failed; continuing", exc_info=True)
