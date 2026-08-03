import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GOOGLE_CREDENTIALS_PATH = os.path.join(
    BASE_DIR, os.getenv("GOOGLE_CREDENTIALS_PATH", "./google_credentials.json")
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_MODEL_FALLBACK = os.getenv("PRIVILEGED_LLM_FALLBACK", "gemini-3.5-flash")

CRITIC_ENABLED = os.getenv("CRITIC_ENABLED", "true").lower() == "true"
# Model tiering: the Critic is a bounded yes/no judgment, so it runs on the
# cheapest/fastest tier rather than the main reasoning model.
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gemini-3.1-flash-lite")

# Critic backend: "gemini" (cloud, default) or "ollama" (local Gemma, for
# private/desktop mode). Ollama path falls back to Gemini if unreachable.
CRITIC_BACKEND = os.getenv("CRITIC_BACKEND", "gemini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CRITIC_MODEL = os.getenv("OLLAMA_CRITIC_MODEL", "gemma3:4b")

# Hard ceiling (seconds) on any single Gemini API call, so a stalled request
# raises instead of retrying/hanging indefinitely inside the SDK.
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

MEMORY_DB_PATH = os.path.join(BASE_DIR, "memory_db")
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


# --- Phase 4: voice pipeline -------------------------------------------------
# Voice is opt-in; `python main.py --voice` overrides VOICE_ENABLED.
VOICE_ENABLED = _bool("VOICE_ENABLED", "false")

# Activation: "hotkey" (push-to-talk, default) or "wakeword" (always listening).
VOICE_ACTIVATION = os.getenv("VOICE_ACTIVATION", "hotkey")
VOICE_HOTKEY = os.getenv("VOICE_HOTKEY", "<ctrl>+<space>")
# "hey_jarvis" is a pretrained openWakeWord model name, not the product name -
# there is no "Echo" model yet (see voice/wake_word.py). Swappable to any of
# openWakeWord's other pretrained phrases via this env var.
VOICE_WAKE_WORD = os.getenv("VOICE_WAKE_WORD", "hey_jarvis")
VOICE_WAKE_THRESHOLD = float(os.getenv("VOICE_WAKE_THRESHOLD", "0.5"))

# Audio frame geometry. Silero VAD requires exactly 512 samples at 16kHz
# (32ms) per inference, so that is the frame size everything else works in.
VOICE_SAMPLE_RATE = 16000
VOICE_FRAME_SAMPLES = 512

# EchoGuard: the mic stays armed during playback so the user can interrupt by
# voice. Frames matching what we just played are dropped before reaching the
# VAD or wake-word model. Set VOICE_MIC_DURING_PLAYBACK=false to hard-gate the
# mic instead (barge-in then works only via hotkey/typing).
VOICE_MIC_DURING_PLAYBACK = _bool("VOICE_MIC_DURING_PLAYBACK", "true")
VOICE_ECHO_THRESHOLD = float(os.getenv("VOICE_ECHO_THRESHOLD", "0.6"))

# Speak replies to typed input too. Off by default: output mode follows input
# mode, so typing stays a quiet path.
VOICE_SPEAK_TEXT_REPLIES = _bool("VOICE_SPEAK_TEXT_REPLIES", "false")

# Short tones on activation (heard) and on giving up (not heard). Without the
# first you have no signal that the wake word registered before you start
# talking; without the second, a capture that never hears speech waits
# silently forever with no way to know it gave up.
VOICE_CUES_ENABLED = _bool("VOICE_CUES_ENABLED", "true")

# How long to wait, after activation, for the user to START speaking before
# giving up and returning to idle. Previously unbounded: if the mic never
# detected speech, the session stayed in "listening" forever, and the hotkey
# guard (_activate() no-ops while already capturing) meant there was no way
# back in except restarting. Generous by design - this is patience for the
# user to begin, not an end-of-utterance cutoff (that is VOICE_SILENCE_MS).
VOICE_LISTEN_TIMEOUT_SEC = float(os.getenv("VOICE_LISTEN_TIMEOUT_SEC", "8.0"))

# Cut speech on the first keystroke. This is a global listener, so a keypress
# in any application interrupts the assistant - intended for a desktop
# assistant, but set false if it gets in the way.
VOICE_TYPING_BARGE_IN = _bool("VOICE_TYPING_BARGE_IN", "true")

# Speak a trailing fragment once the token stream has been quiet this long.
# Gemini holds the stream open ~2.4s past the final token, so without this a
# reply ending without punctuation waits out that whole tail. Largest measured
# gap between real text deltas was 108ms, so 500ms cannot split live output.
VOICE_TTS_IDLE_FLUSH_MS = int(os.getenv("VOICE_TTS_IDLE_FLUSH_MS", "500"))

VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "small.en")
VOICE_STT_DEVICE = os.getenv("VOICE_STT_DEVICE", "auto")  # auto | cuda | cpu
VOICE_SILENCE_MS = int(os.getenv("VOICE_SILENCE_MS", "600"))
VOICE_MAX_UTTERANCE_SEC = int(os.getenv("VOICE_MAX_UTTERANCE_SEC", "30"))

# Optional cheaper/faster model for spoken turns. Gemini TTFT dominates the
# voice latency budget, so this is the only meaningful lever. Empty = use
# GEMINI_MODEL.
VOICE_MODEL = os.getenv("VOICE_MODEL", "")

VAD_MODEL_PATH = os.path.join(BASE_DIR, "models", "silero", "silero_vad.onnx")

TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro")  # kokoro | elevenlabs
KOKORO_MODEL_DIR = os.path.join(BASE_DIR, os.getenv("KOKORO_MODEL_DIR", "./models/kokoro"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
# onnxruntime intra-op threads for Kokoro. The default (all cores) is slower
# than a small cap on high-core machines - see the note in voice/tts.py.
KOKORO_THREADS = int(os.getenv("KOKORO_THREADS", "4"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# --- Phase 5: three-tier screen context ---
# Tier 2 (user-triggered screenshot) vision model. A bounded describe-what-you-
# see task doesn't need Pro-level reasoning, so this defaults to the same
# GA Flash tier already used for the Critic and Quarantined LLM roles.
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash")

# PyQt6 always-on-top overlay for security toasts (Critic VETO) and element
# highlighting. Off switch for headless/no-display environments, where Qt
# would otherwise fail to init.
OVERLAY_ENABLED = _bool("OVERLAY_ENABLED", "true")

# Tier 1 (uiautomation) element-tree bounds. Unbounded walks of a pathological
# window (e.g. a huge web page's DOM-backed accessibility tree) could return
# megabytes of text; these keep the output small enough to hand to the model.
ACCESSIBILITY_MAX_DEPTH = int(os.getenv("ACCESSIBILITY_MAX_DEPTH", "6"))
ACCESSIBILITY_MAX_ELEMENTS = int(os.getenv("ACCESSIBILITY_MAX_ELEMENTS", "150"))

# Ephemeral Tier-3 (task-scoped capture) storage. Written only during an active
# browser task and wiped when that task ends - see browser/task_runner.py.
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# --- Phase 6: browser automation ---------------------------------------------

# Gate 0 (no-browser path). Most "browse this page" requests are really "read
# this page", and a static fetch answers them for a fraction of the cost and
# with none of the attack surface of a live browser.
NO_BROWSER_GATE_ENABLED = _bool("NO_BROWSER_GATE_ENABLED", "true")
NO_BROWSER_TIMEOUT_SECONDS = int(os.getenv("NO_BROWSER_TIMEOUT_SECONDS", "15"))
# Jina Reader (r.jina.ai) renders a page server-side and returns markdown. It is
# the last rung of the gate before we pay for a browser - but it means sending
# the URL to a third party, so it is separately switchable.
JINA_READER_ENABLED = _bool("JINA_READER_ENABLED", "true")
JINA_READER_ENDPOINT = os.getenv("JINA_READER_ENDPOINT", "https://r.jina.ai/")
# Cap on extracted text handed to a model. Untrusted content is billed by the
# token and a hostile page can be arbitrarily long.
MAX_EXTRACTED_CHARS = int(os.getenv("MAX_EXTRACTED_CHARS", "12000"))

# Patchright stealth requires a *persistent context*: there is no
# browser.new_context() in this mode, so one profile is shared by every task.
# Default is a dedicated, logged-out profile - an injected page that hijacks the
# agent then has no sessions to abuse. Point this at your real Chrome profile
# only once the guardrails have been exercised, and note that Chrome locks a
# profile while it is open.
BROWSER_USER_DATA_DIR = os.path.join(
    BASE_DIR, os.getenv("BROWSER_USER_DATA_DIR", "./browser_profile")
)
# "chrome" (real Chrome, best stealth) or "chromium". Patchright's documented
# config is channel=chrome + headless=False + no_viewport=True with NO custom
# user agent - overriding those is what makes automation detectable.
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")
# Headless defeats Patchright's stealth patches, so it defaults off. The visible
# window is also the point on a desktop assistant: you can watch what Echo does
# and close the window to stop it.
BROWSER_HEADLESS = _bool("BROWSER_HEADLESS", "false")
BROWSER_TIMEOUT_SECONDS = int(os.getenv("BROWSER_TIMEOUT_SECONDS", "30"))
# Idle browsers hold ~300MB and a Chrome profile lock; shut down when unused.
BROWSER_IDLE_SHUTDOWN_SECONDS = int(os.getenv("BROWSER_IDLE_SHUTDOWN_SECONDS", "300"))

# Empty = launch a local browser. Set to a Steel Browser CDP endpoint
# (http://localhost:3000) to route through it for Cloudflare/DataDome/PerimeterX
# sites instead.
STEEL_BROWSER_ENDPOINT = os.getenv("STEEL_BROWSER_ENDPOINT", "")
RESIDENTIAL_PROXY_URL = os.getenv("RESIDENTIAL_PROXY_URL", "")

# Behavioural mimicry bounds (seconds) applied before each browser action.
BROWSER_MIN_ACTION_DELAY = float(os.getenv("BROWSER_MIN_ACTION_DELAY", "0.25"))
BROWSER_MAX_ACTION_DELAY = float(os.getenv("BROWSER_MAX_ACTION_DELAY", "0.9"))

# Quarantined LLM: sees all untrusted external content, has no tools, and can
# only emit schema-validated JSON. Flash tier - this is extraction, not
# reasoning, and it runs on every page.
QUARANTINED_MODEL = os.getenv("QUARANTINED_MODEL", "gemini-3.5-flash")
PROMPT_INJECTION_SCAN = _bool("PROMPT_INJECTION_SCAN", "true")

# Planner/Actor/Validator. The Planner decomposes the task and needs the
# strongest reasoning; Actor and Validator are bounded per-step judgments.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", GEMINI_MODEL)
ACTOR_MODEL = os.getenv("ACTOR_MODEL", "gemini-3.5-flash")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "gemini-3.5-flash")
BROWSER_MAX_STEPS = int(os.getenv("BROWSER_MAX_STEPS", "15"))
BROWSER_MAX_RETRIES_PER_STEP = int(os.getenv("BROWSER_MAX_RETRIES_PER_STEP", "2"))

# Stagehand-style DOM->action cache: a repeated workflow on an unchanged page
# replays its selectors with zero LLM calls.
BROWSER_DB_PATH = os.path.join(BASE_DIR, "memory_db", "browser.db")
ACTION_CACHE_ENABLED = _bool("ACTION_CACHE_ENABLED", "true")
ACTION_CACHE_TTL_DAYS = int(os.getenv("ACTION_CACHE_TTL_DAYS", "7"))

# Confirmation is reserved for genuinely irreversible actions. Most browser
# automation is MEDIUM (logged, auto-executed) - prompting on every click would
# defeat the point of an agent.
CONFIRM_HIGH_RISK = _bool("CONFIRM_HIGH_RISK", "true")

# Google Safe Browsing lookup for URLs found in extracted content. Optional:
# without a key, URL redaction still happens, only the reputation check is
# skipped.
SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY", "")
