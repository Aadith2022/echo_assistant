"""Speech-to-text via faster-whisper (CTranslate2).

Two Windows-specific details are load-bearing:

1. CTranslate2 lazily loads `cublas64_12.dll` and cuDNN 9 at INFERENCE time via
   plain LoadLibrary, which searches PATH and ignores `os.add_dll_directory` -
   so without CUDA installed system-wide the model loads fine and then fails on
   the first transcribe. `_register_cuda_libraries()` prepends the NVIDIA pip
   wheel directories to PATH before faster_whisper is imported. Measured: 0.07s
   on cuda versus 2.44s on cpu for the same clip.

2. The first CUDA transcribe pays ~12s of context initialisation, which would
   otherwise land on the user's first spoken sentence.
"""

import glob
import logging
import os
import site
import time

import numpy as np

import config

logger = logging.getLogger(__name__)


def _register_cuda_libraries() -> None:
    """Put the pip-installed NVIDIA DLLs on PATH, before importing ctranslate2."""
    directories = []
    candidates = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidates.append(user_site)

    for packages_dir in candidates:
        directories.extend(glob.glob(os.path.join(packages_dir, "nvidia", "*", "bin")))

    if not directories:
        logger.debug("no NVIDIA pip libraries found; CUDA will be unavailable")
        return

    existing = os.environ.get("PATH", "")
    missing = [d for d in directories if d not in existing]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + existing
    logger.debug("registered %d NVIDIA library directories", len(directories))


_register_cuda_libraries()

from faster_whisper import WhisperModel  # noqa: E402  (must follow PATH setup)


class Transcriber:
    """Wraps a warm WhisperModel, preferring CUDA and falling back to CPU."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.VOICE_STT_MODEL
        requested = (device or config.VOICE_STT_DEVICE).lower()

        if requested == "auto":
            attempts = [("cuda", "float16"), ("cpu", "int8")]
        elif requested == "cuda":
            attempts = [("cuda", "float16"), ("cpu", "int8")]
        else:
            attempts = [("cpu", "int8")]

        self.model = None
        for dev, compute_type in attempts:
            try:
                model = WhisperModel(self.model_name, device=dev, compute_type=compute_type)
                self._warm_up(model)
            except Exception as e:
                logger.warning(
                    "STT on %s/%s unavailable (%s); trying next option",
                    dev,
                    compute_type,
                    str(e)[:120],
                )
                continue

            self.model = model
            self.device = dev
            self.compute_type = compute_type
            logger.info("STT ready: %s on %s/%s", self.model_name, dev, compute_type)
            break

        if self.model is None:
            raise RuntimeError(
                f"Could not initialise speech-to-text model '{self.model_name}' "
                "on any device."
            )

    def _warm_up(self, model) -> None:
        """Force CUDA context creation and kernel selection up front.

        This also doubles as the real check that the device works: a broken
        cuDNN/cuBLAS setup surfaces here, at construction, rather than on the
        user's first sentence.
        """
        start = time.monotonic()
        silence = np.zeros(config.VOICE_SAMPLE_RATE, dtype=np.float32)
        segments, _ = model.transcribe(silence, beam_size=1)
        list(segments)
        logger.debug("STT warmup took %.1fs", time.monotonic() - start)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe mono float32 16kHz audio to text."""
        if audio.size == 0:
            return ""

        start = time.monotonic()
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            beam_size=1,          # greedy: this is a latency-critical path
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,     # our own VAD already bounded the utterance
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "STT %.2fs for %.1fs of audio -> %r",
            time.monotonic() - start,
            audio.size / config.VOICE_SAMPLE_RATE,
            text[:80],
        )
        return text
