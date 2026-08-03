"""Turn a stream of text deltas into complete sentences.

The Gemini client emits text token-by-token via `on_delta`. Speaking each
fragment as it arrives would be gibberish, and waiting for the whole reply
would throw away the streaming latency win. This buffers deltas and emits
whole sentences, so the assistant starts talking after the first sentence is
complete rather than after the last.

**Idle flush.** Gemini holds the stream open for ~2.4s after the final text
token arrives (measured: last token at 1.65s, `interaction.completed` at
4.10s). A reply whose final fragment has no terminal punctuation would
otherwise sit finished-but-unspoken for that whole tail, because `flush()`
only runs once the stream loop ends. So a watchdog emits the tail once no
delta has arrived for `idle_flush_seconds`.

The threshold is set from measurement, not guesswork: across sampled
generations the largest observed gap between consecutive text deltas was
108ms (p99 107ms), so the 500ms default has ~4.6x headroom and will not split
a sentence the model is still writing.
"""

import logging
import re
import threading

logger = logging.getLogger(__name__)

# Terminal punctuation followed by whitespace. Requiring the whitespace is what
# keeps decimals ("15.5 degrees") and domains ("example.com") intact, since
# their dot is followed by a character rather than a space.
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)(\s+)")

# A line break is also a spoken boundary. Without this a bulleted list - whose
# items often carry no terminal punctuation - accumulates into one breathless
# run that only flushes at the soft limit.
_LINE_BOUNDARY = re.compile(r"([^\n]*[^\s\n])(\s*\n+\s*)")

# A trailing dot here is an abbreviation, not a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt",
    "inc", "ltd", "co", "corp", "dept", "est", "fig", "vol",
    "e.g", "i.e", "etc", "vs", "approx", "avg", "min", "max",
    "a.m", "p.m", "u.s", "u.k", "no",
}

# Don't emit a "sentence" shorter than this; it is almost always a stray
# fragment, and very short utterances synthesize with odd prosody.
_MIN_CHARS = 2

# If the model produces a long run with no terminal punctuation (lists, code,
# rambling), flush at a comma or space past this length so speech doesn't stall.
_SOFT_LIMIT = 220


class SentenceBuffer:
    """Accumulates deltas, invoking `on_sentence` for each complete sentence."""

    def __init__(self, on_sentence, idle_flush_seconds: float | None = None):
        self.on_sentence = on_sentence
        self.idle_flush_seconds = idle_flush_seconds
        self._pending = ""
        # feed() runs on the stream thread while the idle watchdog fires on a
        # timer thread, so _pending needs guarding.
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def feed(self, delta: str) -> None:
        if not delta:
            return

        ready = []
        with self._lock:
            self._pending += delta
            while True:
                sentence, remainder = self._split(self._pending)
                if sentence is None:
                    break
                self._pending = remainder
                ready.append(sentence)
            still_pending = bool(self._pending.strip())

        # Emit outside the lock: on_sentence hands off to the TTS queue and
        # should never be able to block a subsequent feed().
        for sentence in ready:
            self._emit(sentence)

        self._rearm(still_pending)

    def flush(self) -> None:
        """Emit whatever is left. Call at the end of a turn. Idempotent."""
        self._cancel_timer()
        with self._lock:
            remaining = self._pending.strip()
            self._pending = ""
        if remaining:
            self._emit(remaining)

    def reset(self) -> None:
        self._cancel_timer()
        with self._lock:
            self._pending = ""

    # --- idle watchdog ---------------------------------------------------

    def _rearm(self, has_pending: bool) -> None:
        self._cancel_timer()
        if not has_pending or not self.idle_flush_seconds:
            return
        timer = threading.Timer(self.idle_flush_seconds, self._on_idle)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _on_idle(self) -> None:
        """No delta for a while: the reply is finished, speak the tail."""
        with self._lock:
            remaining = self._pending.strip()
            self._pending = ""
        if remaining:
            logger.debug("idle flush of trailing fragment: %r", remaining[:60])
            self._emit(remaining)

    def _emit(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        try:
            self.on_sentence(sentence)
        except Exception:
            logger.exception("sentence sink raised for %r", sentence[:60])

    def _split(self, text: str):
        """Return (sentence, remainder), or (None, text) if none is complete."""
        for match in _BOUNDARY.finditer(text):
            end = match.end(1)
            candidate = text[:end]

            if len(candidate.strip()) < _MIN_CHARS:
                continue
            if self._ends_with_abbreviation(candidate):
                continue

            return candidate, text[match.end() :]

        # No punctuated boundary; fall back to a completed line.
        line = _LINE_BOUNDARY.match(text)
        if line and len(line.group(1).strip()) >= _MIN_CHARS:
            return line.group(1), text[line.end() :]

        # No boundary found. Emit early if the run has grown too long to hold.
        if len(text) >= _SOFT_LIMIT:
            cut = max(text.rfind(", ", 0, _SOFT_LIMIT), text.rfind(" ", 0, _SOFT_LIMIT))
            if cut > _MIN_CHARS:
                return text[: cut + 1], text[cut + 1 :]

        return None, text

    @staticmethod
    def _ends_with_abbreviation(candidate: str) -> bool:
        if not candidate.endswith("."):
            return False
        # Take the last whitespace-delimited token, minus the trailing dot.
        token = candidate[:-1].split()[-1] if candidate[:-1].split() else ""
        token = token.strip("(\"'[").lower()
        if token in _ABBREVIATIONS:
            return True
        # A single letter before the dot is an initial ("J. Smith"), not an end.
        return len(token) == 1 and token.isalpha()
