"""Tier 1 screen context: the Windows Accessibility API via `uiautomation`.

Returns a condensed text description of the foreground window's UI element
tree - no pixels, no vision model, <50ms in practice. This is the cheap,
privacy-safe default the model should reach for before ever taking a
screenshot (Tier 2).
"""

import logging

import config

logger = logging.getLogger(__name__)

# Structural containers are only worth descending into, not reporting - they
# clutter the tree without telling the model anything actionable.
_SKIP_IF_UNNAMED = {"PaneControl", "GroupControl", "CustomControl", "WindowControl"}

_MAX_OUTPUT_CHARS = 4000
_MAX_VALUE_CHARS = 200


def _describe(control) -> str | None:
    control_type = getattr(control, "ControlTypeName", "") or "Control"
    name = (getattr(control, "Name", "") or "").strip()

    if not name and control_type in _SKIP_IF_UNNAMED:
        return None

    value = ""
    try:
        value_pattern = control.GetValuePattern()
        if value_pattern:
            raw_value = (value_pattern.Value or "").strip()
            if raw_value:
                value = raw_value[:_MAX_VALUE_CHARS]
    except Exception:
        pass

    label = f"{control_type}: '{name}'" if name else control_type
    if value and value != name:
        label += f" = '{value}'"
    return label


def _walk(control, depth: int, budget: list[int], lines: list[str]) -> None:
    if depth > config.ACCESSIBILITY_MAX_DEPTH or budget[0] <= 0:
        return

    for child in control.GetChildren():
        if budget[0] <= 0:
            return

        description = _describe(child)
        if description is not None:
            lines.append("  " * depth + description)
            budget[0] -= 1

        _walk(child, depth + 1, budget, lines)


def get_screen_context() -> str:
    """Describe the foreground window's UI elements as a text tree."""
    try:
        import uiautomation as auto
    except Exception as e:
        return f"Screen context unavailable: uiautomation is not installed ({e})."

    try:
        root = auto.GetForegroundControl()
        if root is None:
            return "Screen context unavailable: no foreground window detected."

        window_name = (getattr(root, "Name", "") or "").strip() or "(untitled)"
        class_name = getattr(root, "ClassName", "") or "unknown"

        lines = [f"Focused window: '{window_name}' ({class_name})"]
        budget = [config.ACCESSIBILITY_MAX_ELEMENTS]
        _walk(root, depth=1, budget=budget, lines=lines)

        text = "\n".join(lines)
        if len(text) > _MAX_OUTPUT_CHARS:
            text = text[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return text
    except Exception as e:
        logger.exception("Accessibility read failed")
        return f"Screen context unavailable: {e}"
