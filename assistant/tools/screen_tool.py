from vision.accessibility_reader import get_screen_context as _get_screen_context
from vision.context_analyzer import analyze_screen


def get_screen_context() -> str:
    """Tier 1: describe the focused window and its UI elements via the
    Windows Accessibility API. Fast, text-only, no screenshot taken."""
    return _get_screen_context()


def see_screen(question: str = "") -> str:
    """Tier 2: take a screenshot of the focused window and visually analyze
    it. Use only when a text description of UI elements isn't enough - e.g.
    the user is asking about an image, video, or visual layout."""
    return analyze_screen(question)
