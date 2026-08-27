"""Always-on-top overlay: security toasts and element highlighting.

A toast fires when the User Alignment Critic vetoes an action.

Each widget runs as its own short-lived subprocess rather than on a background
thread: PyQt6's QApplication must own whichever thread it runs on for that
thread's whole life, and driving one from a worker thread segfaults. A
dedicated process's main thread sidesteps that, and overlapping toasts share
no Qt state.

If PyQt6 is not importable, every method becomes a silent no-op - the overlay
must never be able to break the tool loop or the Critic path.
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
