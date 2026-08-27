"""Screenshot capture primitive, shared by the Tier 2 and Tier 3 screen paths.

Returns raw PNG bytes and never writes to disk - what happens to them is the
caller's decision.
"""

import io
import logging

import mss
from PIL import Image

logger = logging.getLogger(__name__)


def _foreground_bounds() -> tuple[int, int, int, int] | None:
    """Return (left, top, width, height) of the foreground window, or None."""
    try:
        import uiautomation as auto

        root = auto.GetForegroundControl()
        if root is None:
            return None
        rect = root.BoundingRectangle
        left, top = rect.left, rect.top
        width, height = rect.width(), rect.height()
        if width <= 0 or height <= 0:
            return None
        return left, top, width, height
    except Exception:
        return None


def capture_foreground_window() -> bytes:
    """Screenshot the foreground window (falls back to the primary monitor
    if its bounds can't be read) and return PNG bytes."""
    bounds = _foreground_bounds()

    with mss.mss() as sct:
        if bounds is not None:
            left, top, width, height = bounds
            region = {"left": left, "top": top, "width": width, "height": height}
        else:
            region = sct.monitors[1]  # index 0 is the union of all monitors

        shot = sct.grab(region)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
