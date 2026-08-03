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
    "see_screen": MEDIUM,  # screen contents leave the device to a cloud vision call
}

DEFAULT_RISK = HIGH


def classify(tool_name: str, args: dict | None = None) -> str:
    """Return the risk tier for a proposed tool call. `args` is accepted for
    future arg-based rules (e.g. a file_ops delete outside the user's home)."""
    return _TOOL_RISK.get(tool_name, DEFAULT_RISK)


def needs_critic(tier: str) -> bool:
    """LOW actions auto-execute; everything else goes through the Critic."""
    return tier != LOW
