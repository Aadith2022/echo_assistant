"""Wake-word detection via openWakeWord.

The project is named Echo, but the spoken activation phrase is still "Hey
Jarvis" - openWakeWord only ships six pretrained models (alexa, hey_jarvis,
hey_mycroft, hey_rhasspy, timer, weather) and none of them is "Echo". A custom
"Hey Echo" model is a separate project (synthetic TTS training data, negative
audio, training a classifier), not a quick swap, so this is a known,
deliberate gap rather than an oversight. Uses the pretrained model as-is, so
nothing needs training today.

Frames arrive from `AudioInput` as 512-sample float32; openWakeWord expects
int16 and performs best on 1280-sample (80ms) chunks, so they are re-blocked
here.

Pinned to openwakeword==0.6.0 in requirements.txt: if resolution falls back to
0.4.0 the API is completely different and fails in confusing, unrelated ways.
"""

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

# openWakeWord's expected chunk (80ms at 16kHz).
_CHUNK = 1280

# After firing, ignore detections for this long so one utterance of the wake
# word doesn't trigger repeatedly as it slides through the model's window.
_REFRACTORY_CHUNKS = 12  # ~1s


class WakeWordDetector:
    """Calls `on_detect()` when the wake word is heard."""

    def __init__(self, on_detect, name: str = None, threshold: float = None):
        self.on_detect = on_detect
        self.name = name or config.VOICE_WAKE_WORD
        self.threshold = threshold if threshold is not None else config.VOICE_WAKE_THRESHOLD

        from openwakeword.model import Model

        try:
            self._model = Model(wakeword_models=[self.name], inference_framework="onnx")
        except Exception:
            # Models ship separately from the package; fetch on first use.
            import openwakeword.utils

            logger.info("downloading openWakeWord models...")
            openwakeword.utils.download_models()
            self._model = Model(wakeword_models=[self.name], inference_framework="onnx")

        self._pending = np.zeros(0, dtype=np.float32)
        self._cooldown = 0
        self.enabled = True
        self.last_score = 0.0
        logger.info("wake word ready: %r (threshold %.2f)", self.name, self.threshold)

    def feed(self, frame: np.ndarray) -> None:
        """Consume a 512-sample float32 frame from AudioInput."""
        if not self.enabled:
            return

        self._pending = np.concatenate((self._pending, frame))

        while self._pending.size >= _CHUNK:
            chunk = self._pending[:_CHUNK]
            self._pending = self._pending[_CHUNK:]

            pcm = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
            scores = self._model.predict(pcm)
            score = max(scores.values()) if scores else 0.0
            self.last_score = score

            if self._cooldown > 0:
                self._cooldown -= 1
                continue

            if score >= self.threshold:
                self._cooldown = _REFRACTORY_CHUNKS
                logger.info("wake word detected (%.2f)", score)
                try:
                    self.on_detect()
                except Exception:
                    logger.exception("wake word handler raised")

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._cooldown = _REFRACTORY_CHUNKS
