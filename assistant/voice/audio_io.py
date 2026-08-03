"""Shared microphone input.

Opening the mic twice is unreliable on Windows (WASAPI/MME will often fail the
second stream), and both the wake-word detector and the utterance capture need
frames at the same time. So there is exactly one input stream, and consumers
subscribe to it.

Frames are 512 samples of mono float32 at 16kHz (32ms). That size is not
arbitrary: Silero VAD requires exactly 512 samples per inference at 16kHz, so
using it as the universal frame size means no consumer has to re-block.
"""

import logging
import queue
import threading

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)

_INT16_SCALE = 1.0 / 32768.0


class AudioInput:
    """One 16kHz mono mic stream, fanned out to any number of subscribers."""

    def __init__(self, sample_rate: int = None, frame_samples: int = None):
        self.sample_rate = sample_rate or config.VOICE_SAMPLE_RATE
        self.frame_samples = frame_samples or config.VOICE_FRAME_SAMPLES

        self._subscribers: list = []
        self._lock = threading.Lock()
        self._stream = None
        # The PortAudio callback runs on a realtime thread; anything slow there
        # causes input overflow and dropped audio. It only enqueues, and a
        # dispatcher thread does the actual fan-out to (comparatively slow)
        # consumers like the VAD and wake-word models.
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._dispatcher: threading.Thread | None = None
        self._running = threading.Event()

    def subscribe(self, callback) -> None:
        """Register callback(frame: np.ndarray[float32, 512])."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("audio input status: %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            # Dropping is the correct failure mode here: blocking the realtime
            # thread would corrupt the whole stream, and stale frames are
            # useless for both endpointing and wake-word detection.
            logger.warning("audio queue full - dropping frame")

    def _dispatch_loop(self) -> None:
        while self._running.is_set():
            try:
                raw = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * _INT16_SCALE

            with self._lock:
                subscribers = list(self._subscribers)
            for callback in subscribers:
                try:
                    callback(frame)
                except Exception:
                    # One misbehaving consumer must not kill the audio path.
                    logger.exception("audio subscriber raised")

    def start(self) -> None:
        if self._stream is not None:
            return

        self._running.set()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="audio-dispatch", daemon=True
        )
        self._dispatcher.start()

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()
        logger.info(
            "microphone open (%dHz, %d-sample frames)", self.sample_rate, self.frame_samples
        )

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=1.0)
            self._dispatcher = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
