"""Prepare model output for a text-to-speech voice.

The model writes for a screen: bold, bullets, links, headings. Fed straight to
Kokoro those become spoken noise - `"**97% Tomatometer score**"` is read with
the asterisks, and a single marked-up sentence ballooned to 14.5s of audio in
testing.

The real fix is upstream: spoken turns use a system prompt that asks for plain
conversational prose (see VOICE_SYSTEM_PROMPT in llm/gemini_client.py). This
module is the safety net for whatever still slips through, and it runs only on
the speech path - what gets *printed* keeps its formatting.
"""

import re

# Fenced code blocks: unspeakable, so drop them entirely.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")

# ![alt](url) before [text](url), so image alt text isn't left with a stray "!".
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+|www\.\S+")

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•·]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_HRULE = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$", re.MULTILINE)

# Emphasis. Longest markers first so ** is consumed before *.
_EMPHASIS = [
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL), r"\1"),
    (re.compile(r"___(.+?)___", re.DOTALL), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"__(.+?)__", re.DOTALL), r"\1"),
    (re.compile(r"~~(.+?)~~", re.DOTALL), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9])\*(.+?)\*(?![A-Za-z0-9])", re.DOTALL), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9])_(.+?)_(?![A-Za-z0-9])", re.DOTALL), r"\1"),
]

# Any emphasis markers left over after the paired forms above (e.g. an
# unterminated "**" from a truncated stream).
_STRAY_MARKERS = re.compile(r"[*`~]+")

# Stray underscores, but only at word boundaries - a word-internal underscore
# is part of an identifier like remember_fact, not leftover markup.
_STRAY_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])")

# Symbols a TTS voice either mispronounces or skips silently.
_SYMBOLS = [
    (re.compile(r"°\s*C\b"), " degrees Celsius"),
    (re.compile(r"°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"°"), " degrees "),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*\|\s*"), ", "),      # table cells
    (re.compile(r"(\w)/(\w)"), r"\1 or \2"),
]

# Emoji and pictographs.
_EMOJI = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF" "\U0000FE00-\U0000FE0F" "]+"
)

_WHITESPACE = re.compile(r"[ \t]+")
_REPEATED_PUNCT = re.compile(r"([.,;:!?])\1+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_ORPHAN_PUNCT = re.compile(r"^\s*[.,;:]+\s*")


def for_speech(text: str) -> str:
    """Strip formatting so a TTS voice reads the words, not the markup.

    Returns an empty string when nothing speakable is left (e.g. the input was
    only a horizontal rule or a bare URL), which callers should treat as
    "say nothing".
    """
    if not text:
        return ""

    out = _CODE_FENCE.sub(" ", text)
    out = _INLINE_CODE.sub(r"\1", out)

    out = _IMAGE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _BARE_URL.sub(" ", out)

    out = _HRULE.sub(" ", out)
    out = _HEADING.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _BULLET.sub("", out)
    out = _NUMBERED.sub("", out)

    for pattern, replacement in _EMPHASIS:
        out = pattern.sub(replacement, out)
    out = _STRAY_MARKERS.sub("", out)
    out = _STRAY_UNDERSCORE.sub("", out)

    for pattern, replacement in _SYMBOLS:
        out = pattern.sub(replacement, out)

    out = _EMOJI.sub(" ", out)

    # A line break is a natural spoken pause. Give bullet lists sentence
    # separation so they don't run together into one breathless clause.
    out = re.sub(r"\s*\n\s*", lambda m: ". " if _needs_stop(out, m.start()) else " ", out)

    out = _WHITESPACE.sub(" ", out)
    out = _REPEATED_PUNCT.sub(r"\1", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _ORPHAN_PUNCT.sub("", out)
    out = out.strip()

    # Nothing but punctuation left means there was nothing to say.
    if not any(ch.isalnum() for ch in out):
        return ""
    return out


def _needs_stop(text: str, index: int) -> bool:
    """True if the text before a line break doesn't already end a sentence."""
    before = text[:index].rstrip()
    return bool(before) and before[-1] not in ".!?:,;"
