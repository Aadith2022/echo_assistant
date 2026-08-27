"""Standalone entry point for a single overlay widget (toast or highlight box).

Run as its own process rather than on a background thread: PyQt6's
QApplication must own whichever thread it runs on for that thread's whole
life, and driving one from a worker thread segfaults. Each invocation shows
one widget, waits out its own timer, and exits.
"""

import argparse
import sys

_LEVEL_COLORS = {
    "info": "#2b6cb0",
    "warning": "#c05621",
    "veto": "#c53030",
}


def _run_toast(message: str, level: str, duration_ms: int) -> None:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import QApplication, QLabel

    app = QApplication([])
    color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])

    # No WA_TranslucentBackground: it silently breaks stylesheet rendering,
    # falling back to a tiny unstyled native tooltip. Every programmatic check
    # still passes, so this is only visible in a screenshot.
    label = QLabel(message)
    label.setWindowFlags(
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
    )
    label.setStyleSheet(
        f"background-color: {color}; color: white; padding: 12px 16px; "
        "border-radius: 8px; font-size: 13px;"
    )
    label.setWordWrap(True)
    label.setFixedWidth(340)
    label.adjustSize()

    screen = QApplication.primaryScreen()
    geo = screen.availableGeometry()
    label.move(geo.right() - label.width() - 24, geo.top() + 24)
    label.show()

    QTimer.singleShot(duration_ms, app.quit)
    app.exec()


def _run_highlight(x: int, y: int, w: int, h: int, duration_ms: int) -> None:
    """Draw a border using four thin opaque strips rather than one widget with
    a painted border on a translucent background, which the rendering failure
    noted in _run_toast would make invisible. Four bars need no transparency
    at all and leave the highlighted content visible."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication([])
    color = _LEVEL_COLORS["veto"]
    thickness = 4
    style = f"background-color: {color};"
    flags = (
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
    )

    bars = []
    for geometry in (
        (x, y, w, thickness),  # top
        (x, y + h - thickness, w, thickness),  # bottom
        (x, y, thickness, h),  # left
        (x + w - thickness, y, thickness, h),  # right
    ):
        bar = QWidget()
        bar.setWindowFlags(flags)
        bar.setStyleSheet(style)
        bar.setGeometry(*geometry)
        bar.show()
        bars.append(bar)

    QTimer.singleShot(duration_ms, app.quit)
    app.exec()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    toast_parser = sub.add_parser("toast")
    toast_parser.add_argument("--message", required=True)
    toast_parser.add_argument("--level", default="info")
    toast_parser.add_argument("--duration-ms", type=int, default=4000)

    highlight_parser = sub.add_parser("highlight")
    highlight_parser.add_argument("--x", type=int, required=True)
    highlight_parser.add_argument("--y", type=int, required=True)
    highlight_parser.add_argument("--w", type=int, required=True)
    highlight_parser.add_argument("--h", type=int, required=True)
    highlight_parser.add_argument("--duration-ms", type=int, default=2000)

    args = parser.parse_args()

    if args.kind == "toast":
        _run_toast(args.message, args.level, args.duration_ms)
    else:
        _run_highlight(args.x, args.y, args.w, args.h, args.duration_ms)


if __name__ == "__main__":
    sys.exit(main())
