"""Always-on-top overlay: security toasts and element highlighting.

Wires up the "Security notifications" row of CLAUDE.md's five-layer-defense
table - a toast fires when the User Alignment Critic VETOes an action.
`highlight_element` is a working primitive for the same purpose Phase 6/7
will need (drawing attention to a specific screen region during a browser/OS
task) - built now because it's cheap, but nothing calls it yet.

Each toast/highlight runs as its own short-lived subprocess
(`vision/_overlay_process.py`) rather than on a background thread of the main
process. PyQt6's QApplication must own whichever thread it runs on for its
whole lifetime; driving one from a worker thread segfaulted in testing here.
A dedicated process's main thread sidesteps that, and multiple toasts can
overlap without any shared Qt state to corrupt.

If PyQt6 isn't importable at all (headless box, missing DLL), every method
silently becomes a no-op - the overlay is a UX nicety and must never be able
to break the tool loop or the Critic path.
"""

import logging
import subprocess
import sys

import config

logger = logging.getLogger(__name__)


class OverlayManager:
    def __init__(self):
        self._available = False

    def start(self) -> None:
        if not config.OVERLAY_ENABLED:
            logger.info("Overlay disabled via OVERLAY_ENABLED=false.")
            return
        try:
            import PyQt6  # noqa: F401
        except Exception as e:
            logger.warning("Overlay unavailable (%s); toasts/highlights will be no-ops.", e)
            return
        self._available = True

    def stop(self) -> None:
        pass  # Nothing persistent to tear down - each toast is its own process.

    def show_toast(self, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        if not self._available:
            return
        self._spawn(
            ["toast", "--message", message, "--level", level, "--duration-ms", str(duration_ms)]
        )

    def highlight_element(self, x: int, y: int, w: int, h: int, duration: float = 2.0) -> None:
        if not self._available:
            return
        self._spawn(
            [
                "highlight",
                "--x", str(x), "--y", str(y), "--w", str(w), "--h", str(h),
                "--duration-ms", str(int(duration * 1000)),
            ]
        )

    def _spawn(self, args: list[str]) -> None:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "vision._overlay_process", *args],
                cwd=config.BASE_DIR,
            )
        except Exception:
            logger.exception("Failed to launch overlay process; continuing without it")


overlay = OverlayManager()
