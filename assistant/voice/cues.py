"""Short audio cues that tell the user what the assistant is doing.

Without these, saying the wake word produces silence and you have no idea
whether you were heard until the reply arrives.

Generated rather than shipped as audio files - they are a few sine cycles, so
a WAV asset would be pure overhead. Default to Kokoro's 24kHz so `Speaker`
never reopens the output stream at a different rate just to play one.

Deliberately quiet and short: a cue that startles you, or that you wait through
before speaking, is worse than no cue.
"""

import numpy as np

DEFAULT_SAMPLE_RATE = 24000

# Raised-cosine fade at each end. Without it the abrupt start and stop of a
# sine produces an audible click.
_FADE_MS = 8.0


def _tone(freq: float, ms: float, amplitude: float, sample_rate: int) -> np.ndarray:
    samples = int(sample_rate * ms / 1000.0)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)

    fade = max(1, int(sample_rate * _FADE_MS / 1000.0))
    if fade * 2 < samples:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]

    return wave * amplitude


def _silence(ms: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * ms / 1000.0), dtype=np.float32)


def listening(sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Rising two-note chirp: 'I'm listening, go ahead.'"""
    audio = np.concatenate([
        _tone(784.0, 70, 0.20, sample_rate),    # G5
        _silence(18, sample_rate),
        _tone(1046.5, 90, 0.20, sample_rate),   # C6
    ])
    return audio, sample_rate


def captured(sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Single soft note: 'got it, working on it.' Quieter than the open cue."""
    audio = _tone(880.0, 80, 0.11, sample_rate)  # A5
    return audio, sample_rate


def cancelled(sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Falling two-note: 'I heard nothing, standing down.'"""
    audio = np.concatenate([
        _tone(659.3, 70, 0.13, sample_rate),    # E5
        _silence(14, sample_rate),
        _tone(493.9, 100, 0.13, sample_rate),   # B4
    ])
    return audio, sample_rate
