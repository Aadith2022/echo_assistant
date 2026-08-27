"""Deterministic risk-tier classification for proposed actions.

This is a Layer-1 (deterministic policy) control, run *outside* the LLM's
reasoning: no model call, instant. It decides how much scrutiny an action needs
so we don't pay for a Critic review on read-only, low-stakes tools.
"""

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"

# Per-tool risk map. Anything not listed defaults to HIGH so new/unrecognized
# tools are reviewed by default rather than silently auto-executed.
_TOOL_RISK = {
    "web_search": LOW,
    "get_weather": LOW,
    "get_upcoming_events": LOW,
    "recall_memories": LOW,
    "remember_fact": LOW,  # local, reversible, benign personal facts
    "get_screen_context": LOW,  # read-only, local, no network
    "see_screen": MEDIUM,  # screen contents leave the device to a cloud model
    "read_web_page": LOW,  # a fetch: no browser, no state change
    # MEDIUM by design - automation is the point of the product, so it is
    # logged and auto-executed rather than prompting. Individual actions inside
    # the task go through classify_browser_action().
    "browse_task": MEDIUM,
}

DEFAULT_RISK = HIGH

# --- argument-based rules ----------------------------------------------------
#
# This list is deliberately short: high precision, low recall. A broad keyword
# list (submit, send, post, apply, share) is a security failure, not merely an
# annoyance - a user prompted about "Submit feedback" ten times learns to hit
# "y" without reading, and the eleventh prompt gets the same reflex. An
# over-triggering classifier trains the human controller to stop being one.
#
# So this catches only what is unambiguously irreversible. Everything else
# lands on MEDIUM and goes to the Critic, which has the user's actual request
# in front of it: "Submit" is fine when the user asked to submit and escalates
# when they asked to check a delivery date. Nuanced judgement belongs in the
# model that knows the intent; this layer exists to be the thing an injected or
# miscalibrated model cannot talk its way past.

_CRITICAL_LABELS = (
    "place order", "place your order", "buy now", "confirm purchase",
    "complete purchase", "complete order", "pay now", "make payment",
    "confirm payment", "confirm and pay", "authorise payment", "authorize payment",
    "transfer funds", "send money", "wire transfer",
    "delete account", "close account", "delete permanently", "permanently delete",
    "delete forever",
)

# Fields that must never be filled from anything except an explicit user
# instruction - not from page content, and not from the model's inference.
_SENSITIVE_FIELD_HINTS = (
    "password", "card number", "cvv", "cvc", "security code", "ssn",
    "social security", "passport number", "account number", "sort code",
    "iban", "routing number",
)


def is_credential_field(label: str) -> bool:
    """True if this field holds a secret the agent has no business supplying.

    Anything it types into such a field is invented, and repeated wrong
    passwords lock accounts. Confirmation is not the answer either: approving
    "fill Password with 'password123'" only launders a bad action.
    """
    return any(hint in (label or "").lower() for hint in _SENSITIVE_FIELD_HINTS)


def classify(tool_name: str, args: dict | None = None) -> str:
    """Return the risk tier for a proposed tool call."""
    return _TOOL_RISK.get(tool_name, DEFAULT_RISK)


# Choosing among options a widget is already presenting, rather than committing
# anything - the commitment happens later, at a button or a link, and that
# still goes to the Critic.
#
# LOW because the Critic is not merely slow on these, it is wrong on them, in
# the direction that stops work. It sees an action name and a label with no view
# of the page, and a widget's label is often its current VALUE rather than its
# effect - the control you click to leave "Round trip" is itself labelled
# "Round trip", which reads as acting against the user's request.
#
# The narrow CRITICAL rules still run first: a menu entry reading "Delete
# account" is CRITICAL whatever role it carries.
_WIDGET_CONTROLS = frozenset(
    {
        "option",
        "gridcell",
        "menuitem",
        "menuitemradio",
        "menuitemcheckbox",
        "tab",
        "treeitem",
        "combobox",
        "listbox",
        "select",
        "spinbutton",
    }
)


def classify_browser_action(
    action: str, element_label: str = "", element_control: str = ""
) -> str:
    """Risk-tier a single action inside a browser task.

    Deterministic on purpose: this runs before the Critic and decides whether
    the Critic runs at all, so it must not be something an injected page could
    influence. CRITICAL only for the narrow irreversible set above; ordinary
    state-changing actions are MEDIUM and judged in context.

    `element_control` is the widget role from our own enumeration, never
    anything a page told us in prose.
    """
    label = (element_label or "").strip().lower()
    control = (element_control or "").strip().lower()

    if action in ("navigate", "scroll", "press", "extract", "observe"):
        return LOW

    if action == "fill":
        if any(hint in label for hint in _SENSITIVE_FIELD_HINTS):
            return CRITICAL
        # Typing into a combobox is typing, not choosing - it stays MEDIUM.
        return MEDIUM

    if action == "click":
        # Irreversible always wins, whatever kind of control carries the label.
        if any(phrase in label for phrase in _CRITICAL_LABELS):
            return CRITICAL
        if control in _WIDGET_CONTROLS:
            return LOW
        return MEDIUM

    return DEFAULT_RISK


# Things an agent must never ask the user for mid-task. A clarification
# question is written by a model that has just read a web page, and the user
# sees it as coming from their own assistant - exactly the trust a social
# engineering attack needs. Deterministic, because the judgement must not be
# made by the model that read the page.
_SECRET_REQUEST_PATTERNS = (
    "password", "passcode", "pin number", "your pin", "cvv", "cvc",
    "security code", "card number", "credit card", "debit card", "account number",
    "sort code", "iban", "routing number", "social security", "ssn",
    "passport number", "one-time code", "otp", "verification code", "2fa",
    "two-factor", "authentication code", "seed phrase", "private key",
    "api key", "secret key", "security question",
)


def question_requests_secret(question: str) -> bool:
    """True if a clarification question is fishing for credentials or secrets."""
    lowered = (question or "").lower()
    return any(p in lowered for p in _SECRET_REQUEST_PATTERNS)


_ORDER = (LOW, MEDIUM, HIGH, CRITICAL)


def escalate(tier: str, floor: str) -> str:
    """Raise `tier` to at least `floor`, never lowering it.

    Used for Critic ESCALATE: nothing can reduce a tier set by the categorical
    rules.
    """
    return max(tier, floor, key=lambda t: _ORDER.index(t) if t in _ORDER else len(_ORDER))


def needs_critic(tier: str) -> bool:
    """LOW actions auto-execute; everything else goes through the Critic."""
    return tier != LOW


def needs_confirmation(tier: str) -> bool:
    """Only genuinely irreversible actions interrupt the user.

    HIGH is reached mostly via Critic escalation rather than the keyword list.
    """
    return tier in (HIGH, CRITICAL)
