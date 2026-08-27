"""Voice session orchestration.

Voice mode does not lock you out of the keyboard: listeners run on background
threads while the main thread stays on `input()`, and both paths funnel into
`submit_turn()`.

    activation (hotkey or wake word)
      -> speaker.stop()                 # barge-in
      -> capture frames until VAD endpoint
      -> transcribe
      -> submit_turn(text, spoken=True)

The PortAudio callback and the AudioInput dispatcher must never block, so
transcription and the agent turn go to a separate worker. Only that worker and
the main thread call into `Agent`, and they serialize on `_turn_lock` -
`GeminiClient` mutates `last_interaction_id`, so two concurrent turns would
corrupt the interaction chain.
"""

import logging
import queue
import sys
import threading
import time

import config
from voice import cues
from voice.audio_io import AudioInput
from voice.echo_guard import EchoGuard
from voice.sentence_buffer import SentenceBuffer
from voice.stt import Transcriber
from voice.text_normalizer import for_speech
from voice.tts import Speaker, build_engine
from voice.vad import EndpointDetector

logger = logging.getLogger(__name__)

PROMPT = "You: "


class VoiceSession:
    """Runs the voice loop alongside the text REPL, sharing one Agent."""

    def __init__(self, agent, activation: str = None):
        self.agent = agent
        self.activation = (activation or config.VOICE_ACTIVATION).lower()

        self.echo_guard = EchoGuard()
        self.speaker = Speaker(build_engine(), echo_guard=self.echo_guard)
        self.sentence_buffer = SentenceBuffer(
            self._speak, idle_flush_seconds=config.VOICE_TTS_IDLE_FLUSH_MS / 1000.0
        )

        self.transcriber = Transcriber()
        self.vad = EndpointDetector()
        self.audio = AudioInput()

        self._capturing = threading.Event()
        self._activated_at = 0.0
        self._turn_lock = threading.Lock()
        self._work: queue.Queue = queue.Queue()
        self._running = True

        self._wake = None
        self._hotkey = None
        self._typing_listener = None

        self._worker = threading.Thread(
            target=self._work_loop, name="voice-turn", daemon=True
        )
        self._worker.start()

    # --- output ----------------------------------------------------------

    def write(self, text: str, end: str = "") -> None:
        """Write to the terminal without mangling a pending input prompt."""
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def _write_async(self, text: str) -> None:
        """Write from a background thread, redrawing the prompt afterwards."""
        sys.stdout.write("\r\033[K" + text + "\n" + PROMPT)
        sys.stdout.flush()

    def _speak(self, sentence: str) -> None:
        """Send one sentence to the voice, stripped of anything unspeakable.

        Only the speech path is normalized - what is printed to the terminal
        keeps its original formatting.
        """
        spoken = for_speech(sentence)
        if spoken:
            self.speaker.say(spoken)

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.audio.subscribe(self._on_frame)
        self.audio.start()

        if self.activation == "wakeword":
            from voice.wake_word import WakeWordDetector

            self._wake = WakeWordDetector(on_detect=self._activate)

        from voice.hotkey import HotkeyListener

        self._hotkey = HotkeyListener(on_press=self._activate)
        self._hotkey.start()

        self._start_typing_barge_in()

        if not self.speaker.enabled:
            self.write(
                "[voice] speech output is disabled - see the log for why. "
                "Listening still works.\n"
            )

    def stop(self) -> None:
        self._running = False
        if self._hotkey is not None:
            self._hotkey.stop()
        if self._typing_listener is not None:
            self._typing_listener.stop()
        self.audio.stop()
        self.speaker.close()

    def _start_typing_barge_in(self) -> None:
        """Cut speech on the first keystroke, not at end of line.

        This is a global listener, so a keypress in any application stops the
        assistant mid-sentence. That is intentional for a desktop assistant -
        you turned your attention elsewhere - but set
        VOICE_TYPING_BARGE_IN=false if it gets in the way.
        """
        if not config.VOICE_TYPING_BARGE_IN:
            return
        try:
            from pynput import keyboard

            def on_press(_key):
                if self.speaker.speaking:
                    self.speaker.stop()

            self._typing_listener = keyboard.Listener(on_press=on_press)
            self._typing_listener.daemon = True
            self._typing_listener.start()
        except Exception:
            logger.exception("could not start typing barge-in listener")

    # --- audio path ------------------------------------------------------

    def _on_frame(self, frame) -> None:
        """Called on the audio dispatch thread. Must stay fast."""
        # Drop our own speech (and cues) first, or the VAD endpoints on the
        # assistant's own audio and the wake word retriggers.
        if self.echo_guard.is_echo(frame):
            return

        if self._capturing.is_set():
            utterance = self.vad.feed(frame)
            if utterance is not None:
                self._capturing.clear()
                self._write_async("[listening complete]")
                self._work.put(utterance)
                return

            if not self.vad.speech_started:
                # Bounded patience for the user to BEGIN talking. Unbounded, a
                # capture the VAD never hears speech in listens forever with no
                # way back in - _activate() no-ops while already capturing.
                elapsed = time.monotonic() - self._activated_at
                if elapsed > config.VOICE_LISTEN_TIMEOUT_SEC:
                    self._capturing.clear()
                    self.vad.reset()
                    self._write_async("[no speech heard, standing down]")
                    self._play_cue(cues.cancelled)
            return

        if self._wake is not None and not self.speaker.speaking:
            self._wake.feed(frame)

    def _activate(self) -> None:
        """Begin capturing an utterance (hotkey pressed or wake word heard)."""
        if self._capturing.is_set():
            return
        self.speaker.stop()  # barge-in
        self.vad.reset()
        if self._wake is not None:
            self._wake.reset()
        self._activated_at = time.monotonic()
        self._capturing.set()
        self._write_async("[listening...]")
        self._play_cue(cues.listening)

    def _play_cue(self, cue_fn) -> None:
        if not config.VOICE_CUES_ENABLED or not self.speaker.enabled:
            return
        audio, sample_rate = cue_fn()
        self.speaker.play(audio, sample_rate)

    # --- turns -----------------------------------------------------------

    def _work_loop(self) -> None:
        while self._running:
            try:
                audio = self._work.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = self.transcriber.transcribe(audio)
            except Exception:
                logger.exception("transcription failed")
                continue

            if not text:
                self._write_async("[heard nothing]")
                continue

            self._write_async(f"You (voice): {text}")
            self.submit_turn(text, spoken=True)

    def submit_turn(self, text: str, spoken: bool) -> str:
        """Run one agent turn. Serialized: Agent is not thread-safe."""
        speak = spoken or config.VOICE_SPEAK_TEXT_REPLIES
        speak = speak and self.speaker.enabled

        with self._turn_lock:
            if speak:
                self.sentence_buffer.reset()

            def on_delta(chunk: str) -> None:
                self.write(chunk)
                if speak:
                    self.sentence_buffer.feed(chunk)

            previous_model = None
            if spoken and config.VOICE_MODEL:
                # TTFT dominates the spoken latency budget, so voice turns may
                # run on a faster model than typed ones.
                previous_model = self.agent.llm.model
                self.agent.llm.model = config.VOICE_MODEL

            try:
                reply = self.agent.respond(text, on_delta=on_delta, spoken=speak)
            finally:
                if previous_model is not None:
                    self.agent.llm.model = previous_model

            if speak:
                self.sentence_buffer.flush()
            self.write("\n")
            return reply
