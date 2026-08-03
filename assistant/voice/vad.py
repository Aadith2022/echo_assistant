"""Voice activity detection and end-of-utterance detection (Silero v5).

Runs the Silero ONNX graph directly on onnxruntime rather than through the
`silero-vad` pip package. That package imports `torchaudio` at module load,
which pulled in a torchaudio built against a different torch ABI than the one
sentence-transformers already relies on (WinError 127 on import). Going
straight to onnxruntime - already present as a faster-whisper dependency -
keeps torch out of the realtime audio path entirely and removes the coupling.

Two things about this model fail *silently* if you get them wrong, so both are
load-bearing:

1. It is stateful - a [2, 1, 128] LSTM state is threaded between calls and must
   be reset between utterances.
2. Since v5 it expects 64 samples of preceding context prepended to each
   512-sample frame (576 total). Its declared input shape is [None, None], so
   passing a bare 512 samples is accepted without error and simply returns
   near-zero probabilities forever. Measured on clean speech: max probability
   0.010 without the context window versus 1.000 with it.
"""

import logging

import numpy as np
import onnxruntime as ort

import config

logger = logging.getLogger(__name__)

# Speech probability above which a frame counts as speech.
_SPEECH_THRESHOLD = 0.5
# Hysteresis: dropping out of speech uses a lower bar than entering it, so
# ordinary pauses between words don't flap the state machine.
_SILENCE_THRESHOLD = 0.35

# Audio kept from before speech was confirmed, so the first word isn't clipped.
_PREROLL_MS = 320

# Samples of preceding audio the model expects prepended to each frame.
_CONTEXT_SAMPLES = 64


class EndpointDetector:
    """Detects when the user starts speaking and when they've finished.

    Feed it 512-sample float32 frames. It buffers the utterance internally
    (including pre-roll) and returns the complete audio when the user stops.
    """

    def __init__(self, model_path: str = None, sample_rate: int = None):
        self.sample_rate = sample_rate or config.VOICE_SAMPLE_RATE
        path = model_path or config.VAD_MODEL_PATH

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(self.sample_rate, dtype=np.int64)

        frame_ms = config.VOICE_FRAME_SAMPLES / self.sample_rate * 1000.0
        self._silence_frames_needed = max(1, int(config.VOICE_SILENCE_MS / frame_ms))
        self._max_frames = int(config.VOICE_MAX_UTTERANCE_SEC * 1000 / frame_ms)
        self._preroll_frames = max(1, int(_PREROLL_MS / frame_ms))

        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._preroll: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._silence_run = 0
        self.speech_started = False
        self.last_prob = 0.0

    def _probability(self, frame: np.ndarray) -> float:
        frame = frame.astype(np.float32)
        # Prepend the tail of the previous frame; without this the model
        # returns ~0 for everything (see module docstring).
        windowed = np.concatenate((self._context, frame))
        out, self._state = self._session.run(
            None,
            {
                "input": windowed.reshape(1, -1),
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._context = frame[-_CONTEXT_SAMPLES:].copy()
        return float(out[0][0])

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        """Returns the utterance audio once the user stops speaking, else None."""
        prob = self._probability(frame)
        self.last_prob = prob

        if not self.speech_started:
            # Keep a rolling pre-roll so the onset isn't lost to the detection
            # delay - without this the first syllable is routinely clipped.
            self._preroll.append(frame)
            if len(self._preroll) > self._preroll_frames:
                self._preroll.pop(0)

            if prob >= _SPEECH_THRESHOLD:
                self.speech_started = True
                self._buffer = list(self._preroll)
                self._preroll = []
                self._silence_run = 0
                logger.debug("speech start (p=%.2f)", prob)
            return None

        self._buffer.append(frame)

        if prob < _SILENCE_THRESHOLD:
            self._silence_run += 1
        else:
            self._silence_run = 0

        hit_cap = len(self._buffer) >= self._max_frames
        if self._silence_run >= self._silence_frames_needed or hit_cap:
            if hit_cap:
                logger.info("utterance hit %ds cap", config.VOICE_MAX_UTTERANCE_SEC)
            audio = np.concatenate(self._buffer)
            # Trim the trailing silence we waited through; Whisper does better
            # without a long dead tail.
            trim = self._silence_run * config.VOICE_FRAME_SAMPLES
            if not hit_cap and trim < audio.size:
                audio = audio[: audio.size - trim]
            self.reset()
            return audio

        return None
