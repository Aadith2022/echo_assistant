"""Agent Origin Sets - task-scoped data access, enforced deterministically.

A task declares the domains it needs before it starts, and anything outside
that set is refused at the guardrail layer. Deterministic is the important
word: a set membership test in Python, decided before any model sees the
destination, rather than a rule the Actor is asked to respect - a model that
has just read an attacker's page is the last thing that should adjudicate where
it may go next.

Scoping rules:
* A declared domain covers its subdomains but never a suffix match:
  `notexample.com` is a different origin.
* Redirects are checked at their destination, or an open redirect on an allowed
  host is a free pass.
* `about:` and `data:` are inert and allowed; every other scheme is refused,
  because `file://` is how a browser task reaches the disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

from guardrails.audit_log import log_event

logger = logging.getLogger(__name__)

_INERT_SCHEMES = {"about", "data"}
_ALLOWED_SCHEMES = {"http", "https"}


class OriginSetViolation(Exception):
    """Raised when an action targets something outside the task's scope."""


@dataclass
class OriginSet:
    """The data scope a single task is allowed to touch."""

    task: str
    domains: set[str] = field(default_factory=set)
    # Filesystem paths a task may read or write. Empty by default: browser
    # tasks have no business touching the disk.
    paths: set[str] = field(default_factory=set)

    @classmethod
    def from_domains(cls, task: str, domains: list[str] | None) -> "OriginSet":
        return cls(task=task, domains={_normalize(d) for d in (domains or []) if d.strip()})

    def allows_url(self, url: str) -> bool:
        if not url:
            return False

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()

        if scheme in _INERT_SCHEMES:
            return True
        if scheme not in _ALLOWED_SCHEMES:
            return False

        host = _normalize(parsed.hostname or "")
        if not host:
            return False

        return any(host == d or host.endswith("." + d) for d in self.domains)

    def check_url(self, url: str, action: str = "navigate") -> None:
        """Raise OriginSetViolation if `url` is out of scope."""
        if self.allows_url(url):
            return

        host = urlparse(url).hostname or url
        log_event(
            "origin_set_blocked",
            task=self.task,
            action=action,
            url=url,
            host=host,
            allowed=sorted(self.domains),
        )
        logger.warning("Origin Set blocked %s to %s (allowed: %s)", action, host, sorted(self.domains))
        _notify(f"Blocked: {host} is outside this task's scope")
        raise OriginSetViolation(
            f"{host!r} is outside this task's allowed scope "
            f"({', '.join(sorted(self.domains)) or 'nothing declared'}). "
            "The action was not performed."
        )

    def widen(self, domain: str) -> None:
        """Add a domain mid-task.

        Only ever called from code acting on the user's instruction, never in
        response to page content - a page that could widen its own task's scope
        would make the whole mechanism decorative.
        """
        normalized = _normalize(domain)
        if normalized and normalized not in self.domains:
            self.domains.add(normalized)
            log_event("origin_set_widened", task=self.task, domain=normalized)

    def describe(self) -> str:
        return ", ".join(sorted(self.domains)) or "(empty)"


def _normalize(domain: str) -> str:
    """Reduce a user-supplied domain or URL to a bare lowercase hostname."""
    domain = domain.strip().lower()
    if "://" in domain:
        domain = urlparse(domain).hostname or ""
    # Tolerate a pasted "example.com/path" or a trailing dot.
    domain = domain.split("/")[0].strip(".")
    if domain.startswith("*."):
        domain = domain[2:]
    return domain


def domain_of(url: str) -> str:
    return _normalize(urlparse(url).hostname or "")


def _notify(message: str) -> None:
    try:
        from vision.overlay import overlay

        overlay.show_toast(message, level="veto")
    except Exception:
        logger.exception("Origin Set notification toast failed; continuing")
