"""Suppress the assistant's own speech from the microphone path.

To allow interrupting by voice, the mic stays armed while the assistant talks.
But the VAD detects *speech-shaped audio* without regard to whose voice it is,
so on speakers it reliably hears the assistant's own TTS and either fires a
false capture or corrupts the endpointing of a real one. On this machine the
default mic and speakers are in the same laptop chassis, so coupling is high.

We don't need to infer whose voice it is. `Speaker` owns playback, so we know
the exact samples just sent to the sound card. Each mic frame is matched
against that reference signal by normalized cross-correlation over a range of
delays; a strong peak means the frame is a delayed, attenuated copy of our own
output, and it is dropped before reaching the VAD or wake-word model.

This is the detection half of acoustic echo cancellation without the adaptive
filter. It is a *gate*, not a *separator*: when the user talks over the
assistant the mic captures a genuine mixture, correlation correctly falls below
threshold, and barge-in fires - but the overlapping moment may transcribe
poorly. That is the accepted trade for having voice barge-in at all.
"""

import logging
import threading
import time

import numpy as np
from scipy.signal import resample_poly

import config

logger = logging.getLogger(__name__)

# One second of reference history. Must comfortably exceed the worst-case
# round trip of (output buffer latency + acoustic flight + input buffering),
# since the echo of a sample arrives well after we hand it to the sound card.
_REF_SECONDS = 1.0

# If nothing has been played for this long, playback is over and the guard
# stands down - otherwise stale reference audio could suppress a user who
# happens to say something similar to what was just spoken.
_STALE_AFTER_SEC = 0.35

# Widest plausible round trip: output buffering + acoustic flight + input
# buffering. Searching beyond this buys nothing and costs accuracy (see below).
_MAX_DELAY_SEC = 0.4

# Taking argmax over thousands of lags inflates correlation by chance alone -
# enough for unrelated speech to score ~0.75 and be wrongly suppressed, which
# would make the assistant deaf to the user. So the delay is *locked*: the
# first match must clear a higher bar over the full search, and subsequent
# frames only re-check a narrow window around that delay, where the number of
# candidate lags (and therefore the chance peak) is small.
_ACQUIRE_THRESHOLD = 0.80
_LOCK_TOLERANCE_SEC = 0.02
_LOCK_HOLD_SEC = 0.5

_EPS = 1e-9


class EchoGuard:
    """Matches mic frames against recently played audio to reject self-echo."""

    def __init__(self, sample_rate: int = None, threshold: float = None):
        self.sample_rate = sample_rate or config.VOICE_SAMPLE_RATE
        self.threshold = threshold if threshold is not None else config.VOICE_ECHO_THRESHOLD

        self._ref = np.zeros(int(self.sample_rate * _REF_SECONDS), dtype=np.float32)
        self._lock = threading.Lock()
        self._last_played_at = 0.0

        self._locked_delay: int | None = None
        self._lock_updated_at = 0.0

        # Diagnostics, read by the verification scripts.
        self.last_peak = 0.0
        self.last_delay_ms = 0.0

    def note_played(self, samples: np.ndarray, sample_rate: int) -> None:
        """Record audio being sent to the speakers. Called by `Speaker`."""
        if samples.size == 0:
            return

        mono = samples if samples.ndim == 1 else samples.mean(axis=1)
        if sample_rate != self.sample_rate:
            # e.g. Kokoro's native 24kHz -> 16kHz mic rate (down by 2/3).
            gcd = np.gcd(int(sample_rate), int(self.sample_rate))
            mono = resample_poly(mono, self.sample_rate // gcd, sample_rate // gcd)

        mono = np.asarray(mono, dtype=np.float32)
        with self._lock:
            if mono.size >= self._ref.size:
                self._ref = mono[-self._ref.size :].copy()
            else:
                self._ref = np.concatenate((self._ref[mono.size :], mono))
            self._last_played_at = time.monotonic()

    def reset(self) -> None:
        """Forget reference history - called when playback is cancelled."""
        with self._lock:
            self._ref[:] = 0.0
            self._last_played_at = 0.0
            self._locked_delay = None

    @property
    def active(self) -> bool:
        return (time.monotonic() - self._last_played_at) < _STALE_AFTER_SEC

    def is_echo(self, frame: np.ndarray) -> bool:
        """True if `frame` is a delayed copy of audio we recently played."""
        if not config.VOICE_MIC_DURING_PLAYBACK:
            # Mic is hard-gated elsewhere; nothing to match against.
            return False
        if not self.active:
            return False

        with self._lock:
            ref = self._ref.copy()

        n = frame.size
        if ref.size < n:
            return False

        frame_norm = float(np.sqrt(np.dot(frame, frame)))
        if frame_norm < _EPS:
            return False  # silence: nothing to suppress

        # Only the most recent stretch can plausibly be the echo of this frame.
        span = min(ref.size, int(self.sample_rate * _MAX_DELAY_SEC) + n)
        window = ref[-span:]

        locked = (
            self._locked_delay is not None
            and (time.monotonic() - self._lock_updated_at) < _LOCK_HOLD_SEC
        )

        if locked:
            tolerance = int(self.sample_rate * _LOCK_TOLERANCE_SEC)
            centre = span - n - self._locked_delay
            low = max(0, centre - tolerance)
            high = min(span - n, centre + tolerance)
            if low > high:
                locked = False

        if not locked:
            low, high = 0, span - n

        ncc = self._correlate(window, frame, frame_norm, low, high)
        if ncc.size == 0:
            return False

        offset = int(np.argmax(ncc))
        peak = float(ncc[offset])
        delay = span - n - (low + offset)

        self.last_peak = peak
        self.last_delay_ms = delay / self.sample_rate * 1000.0

        threshold = self.threshold if locked else _ACQUIRE_THRESHOLD
        if peak >= threshold:
            self._locked_delay = delay
            self._lock_updated_at = time.monotonic()
            return True

        if locked:
            # Lost the lock - the frame is no longer purely our own output,
            # which is exactly what happens when the user starts talking over
            # the assistant. Drop the lock so the next acquisition is honest.
            self._locked_delay = None
        return False

    @staticmethod
    def _correlate(window, frame, frame_norm, low, high):
        """Normalized cross-correlation of `frame` against window[low:high+n]."""
        n = frame.size
        segment = window[low : high + n]
        if segment.size < n:
            return np.zeros(0)

        # np.correlate(..., "valid") gives dot(segment[i:i+n], frame) for every
        # lag i at once - the numerator of the NCC.
        numerator = np.correlate(segment, frame, mode="valid")

        # Sliding window norms for the denominator, via a cumulative sum of
        # squares so each window's energy is O(1).
        squared = np.concatenate(([0.0], np.cumsum(segment.astype(np.float64) ** 2)))
        energy = squared[n:] - squared[:-n]
        norms = np.sqrt(np.maximum(energy, 0.0))

        return numerator / (norms * frame_norm + _EPS)
