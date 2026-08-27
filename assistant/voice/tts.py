"""Text-to-speech: pluggable engines plus cancellable playback.

Two backends behind `TTS_BACKEND`: Kokoro (local, default, no key) and
ElevenLabs (cloud, bring-your-own-key). A missing model file or API key
degrades to a clear message rather than raising at import.

`Speaker` is where barge-in lives. Sentences are synthesized and played on a
worker thread, and `stop()` both drains the queue and aborts audio already
handed to the sound card, so an interruption cuts within a block or two.
"""

import logging
import os
import queue
import threading

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)

# Playback block size. Small enough that stop() cuts promptly, large enough to
# avoid underruns: 1024 frames is ~43ms at 24kHz.
_BLOCK = 1024


class TTSUnavailable(Exception):
    """Raised at construction when a backend cannot be used."""


class TTSEngine:
    """Synthesizes text to mono float32 audio."""

    sample_rate: int = 24000

    def synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError


class KokoroTTS(TTSEngine):
    """Local Kokoro via onnxruntime. ~4-5x realtime on CPU."""

    sample_rate = 24000

    def __init__(self, model_dir: str = None, voice: str = None):
        model_dir = model_dir or config.KOKORO_MODEL_DIR
        self.voice = voice or config.KOKORO_VOICE

        model_path = os.path.join(model_dir, "kokoro-v1.0.onnx")
        voices_path = os.path.join(model_dir, "voices-v1.0.bin")

        missing = [p for p in (model_path, voices_path) if not os.path.exists(p)]
        if missing:
            raise TTSUnavailable(
                "Kokoro model files not found: "
                + ", ".join(os.path.basename(p) for p in missing)
                + f"\nDownload them into {model_dir} from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        # onnxruntime's default threading is counterproductive on a high core
        # count: 1.11s/sentence by default versus 0.76s pinned to 4 intra-op
        # threads. Capping also leaves headroom for the STT model.
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.KOKORO_THREADS
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._kokoro = Kokoro.from_session(session, voices_path)

        # Resolved once: passing the voice NAME makes kokoro-onnx re-read the
        # voices archive on every call, and its overlapping zip entries trip
        # Python 3.13's zip-bomb check after enough reads.
        self._style = self._kokoro.get_voice_style(self.voice)

        # The session and voice archive are not thread-safe.
        self._lock = threading.Lock()

        # First synthesis is several times slower than steady state.
        self.synthesize("ready")
        logger.info("TTS ready: kokoro (voice=%s)", self.voice)

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            samples, sample_rate = self._kokoro.create(text, voice=self._style)
        self.sample_rate = sample_rate
        return np.asarray(samples, dtype=np.float32)


class ElevenLabsTTS(TTSEngine):
    """Cloud ElevenLabs. Optional quality upgrade; requires the user's own key."""

    def __init__(self, api_key: str = None, voice_id: str = None, model: str = None):
        self.api_key = api_key if api_key is not None else config.ELEVENLABS_API_KEY
        self.voice_id = voice_id or config.ELEVENLABS_VOICE_ID
        self.model = model or config.ELEVENLABS_MODEL

        if not self.api_key:
            raise TTSUnavailable(
                "ELEVENLABS_API_KEY is not set. Either add it to assistant/.env "
                "or use the local backend with TTS_BACKEND=kokoro."
            )

        # PCM avoids a decode step but is gated to paid tiers; mp3 works on
        # every tier. Probed once at startup, so the fallback does not cost
        # latency on every sentence.
        self.sample_rate = 24000
        self._output_format = "pcm_24000"
        try:
            self.synthesize("Ready.")
        except Exception as e:
            logger.info("ElevenLabs PCM unavailable (%s); falling back to mp3", str(e)[:80])
            self._output_format = "mp3_44100_128"
            self.sample_rate = 44100
            self.synthesize("Ready.")

        logger.info("TTS ready: elevenlabs (%s, %s)", self.model, self._output_format)

    def synthesize(self, text: str) -> np.ndarray:
        import requests

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream",
            params={"output_format": self._output_format},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": self.model},
            timeout=30,
        )
        response.raise_for_status()

        if self._output_format.startswith("pcm_"):
            pcm = np.frombuffer(response.content, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0
        return self._decode_mp3(response.content)

    def _decode_mp3(self, data: bytes) -> np.ndarray:
        import io

        import av

        with av.open(io.BytesIO(data)) as container:
            chunks = [
                frame.to_ndarray().reshape(-1)
                for frame in container.decode(audio=0)
            ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        return audio


def build_engine(backend: str = None) -> TTSEngine | None:
    """Construct the configured engine, or None if unavailable.

    Never raises: voice output is a feature, not a prerequisite, so a missing
    key or model file disables speech and leaves the rest of the assistant
    working.
    """
    backend = (backend or config.TTS_BACKEND).lower()
    engines = {"kokoro": KokoroTTS, "elevenlabs": ElevenLabsTTS}

    if backend not in engines:
        logger.error("Unknown TTS_BACKEND %r; expected one of %s", backend, list(engines))
        return None

    try:
        return engines[backend]()
    except TTSUnavailable as e:
        logger.warning("TTS backend %r unavailable: %s", backend, e)
        return None
    except Exception:
        logger.exception("TTS backend %r failed to initialise", backend)
        return None


class Speaker:
    """Plays queued sentences, interruptibly.

    `say()` returns immediately. `stop()` cancels both the pending queue and
    audio already in the output buffer, which is what makes barge-in feel
    instant rather than "finishes the current sentence first".
    """

    def __init__(self, engine: TTSEngine | None, echo_guard=None):
        self.engine = engine
        self.echo_guard = echo_guard

        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._stream: sd.OutputStream | None = None
        self._stream_lock = threading.Lock()
        self._running = True

        self._worker = threading.Thread(target=self._run, name="tts-speaker", daemon=True)
        self._worker.start()

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    def say(self, text: str) -> None:
        text = text.strip()
        if not text or not self.enabled:
            return
        self._enqueue(("text", text, None))

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play ready-made audio (e.g. a UI cue) through the same path.

        Through the queue and the echo guard exactly as speech is, so the mic
        treats a cue as our own output rather than as the user talking.
        """
        if samples is None or samples.size == 0:
            return
        self._enqueue(("audio", samples, sample_rate))

    def _enqueue(self, item) -> None:
        self._cancel.clear()
        self._idle.clear()
        self._queue.put(item)

    def stop(self) -> None:
        """Cancel everything queued and cut audio already playing."""
        self._cancel.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        with self._stream_lock:
            if self._stream is not None:
                try:
                    # abort() discards buffered audio; stop() would drain it.
                    self._stream.abort()
                except Exception:
                    logger.debug("output stream abort failed", exc_info=True)
        if self.echo_guard is not None:
            self.echo_guard.reset()
        self._idle.set()

    def wait_until_idle(self, timeout: float = None) -> bool:
        return self._idle.wait(timeout=timeout)

    def close(self) -> None:
        self._running = False
        self.stop()
        self._worker.join(timeout=2.0)
        with self._stream_lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def _ensure_stream(self, sample_rate: int) -> sd.OutputStream:
        with self._stream_lock:
            needs_new = (
                self._stream is None
                or self._stream.samplerate != sample_rate
                or self._stream.stopped
            )
            if needs_new:
                if self._stream is not None:
                    self._stream.close()
                self._stream = sd.OutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32"
                )
                self._stream.start()
            return self._stream

    def _run(self) -> None:
        while self._running:
            try:
                kind, payload, sample_rate = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._queue.empty():
                    self._idle.set()
                continue

            if self._cancel.is_set():
                continue

            if kind == "text":
                try:
                    audio = self.engine.synthesize(payload)
                except Exception:
                    logger.exception("synthesis failed for %r", payload[:60])
                    continue
                sample_rate = self.engine.sample_rate
            else:
                audio = payload

            # Synthesis takes real time; the user may have interrupted during it.
            if self._cancel.is_set():
                continue

            self._play(audio, sample_rate)

            if self._queue.empty():
                self._idle.set()

    def _play(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            stream = self._ensure_stream(sample_rate)
        except Exception:
            logger.exception("could not open audio output")
            return

        for start in range(0, audio.size, _BLOCK):
            if self._cancel.is_set():
                return
            block = audio[start : start + _BLOCK]

            # Fed just before the samples go out, so mic frames match what is
            # actually being played.
            if self.echo_guard is not None:
                self.echo_guard.note_played(block, sample_rate)

            try:
                stream.write(block)
            except Exception:
                # abort() from stop() makes the in-flight write raise - the
                # expected path for an interruption, not an error.
                if not self._cancel.is_set():
                    logger.exception("audio write failed")
                return
