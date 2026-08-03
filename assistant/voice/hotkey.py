"""Global push-to-talk hotkey.

`pynput` is used rather than the `keyboard` package because it does not require
running as administrator on Windows.

The hotkey is a toggle rather than a literal hold-to-talk: the VAD already
decides when the utterance ends, so requiring the user to keep a chord held
down would add nothing but strain.
"""

import logging

import config

logger = logging.getLogger(__name__)


class HotkeyListener:
    """Calls `on_press()` when the configured chord is pressed."""

    def __init__(self, on_press, combination: str = None):
        self.on_press = on_press
        self.combination = combination or config.VOICE_HOTKEY
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        def handler():
            try:
                self.on_press()
            except Exception:
                logger.exception("hotkey handler raised")

        try:
            self._listener = keyboard.GlobalHotKeys({self.combination: handler})
            self._listener.start()
            logger.info("push-to-talk hotkey: %s", self.combination)
        except Exception:
            # A bad chord string shouldn't take down voice mode - wake word or
            # typing still work.
            logger.exception("could not register hotkey %r", self.combination)
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
