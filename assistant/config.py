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
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gemini-3.1-flash-lite")

# "gemini" (cloud) or "ollama" (local, for private mode). Ollama falls back to
# Gemini if unreachable.
CRITIC_BACKEND = os.getenv("CRITIC_BACKEND", "gemini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CRITIC_MODEL = os.getenv("OLLAMA_CRITIC_MODEL", "gemma3:4b")

# Hard ceiling on any single Gemini call. Without it a stalled request retries
# inside the SDK indefinitely, which is indistinguishable from a hang.
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

MEMORY_DB_PATH = os.path.join(BASE_DIR, "memory_db")
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


# --- Voice -------------------------------------------------------------------

# Opt-in; `python main.py --voice` overrides this.
VOICE_ENABLED = _bool("VOICE_ENABLED", "false")

# "hotkey" (push-to-talk) or "wakeword" (always listening).
VOICE_ACTIVATION = os.getenv("VOICE_ACTIVATION", "hotkey")
VOICE_HOTKEY = os.getenv("VOICE_HOTKEY", "<ctrl>+<space>")
# A pretrained openWakeWord model name, not the product name - no "Echo" model
# exists yet. Swappable to any of openWakeWord's other pretrained phrases.
VOICE_WAKE_WORD = os.getenv("VOICE_WAKE_WORD", "hey_jarvis")
VOICE_WAKE_THRESHOLD = float(os.getenv("VOICE_WAKE_THRESHOLD", "0.5"))

# Silero VAD requires exactly 512 samples at 16kHz per inference, so that is
# the frame size the whole audio path works in.
VOICE_SAMPLE_RATE = 16000
VOICE_FRAME_SAMPLES = 512

# Keep the mic armed during playback so the user can interrupt by voice;
# EchoGuard drops frames matching what we just played. False hard-gates the mic
# instead, leaving only hotkey/typing barge-in.
VOICE_MIC_DURING_PLAYBACK = _bool("VOICE_MIC_DURING_PLAYBACK", "true")
VOICE_ECHO_THRESHOLD = float(os.getenv("VOICE_ECHO_THRESHOLD", "0.6"))

# Off by default: output mode follows input mode, so typing stays a quiet path.
VOICE_SPEAK_TEXT_REPLIES = _bool("VOICE_SPEAK_TEXT_REPLIES", "false")

# Short tones on activation and on giving up, so the user knows whether the
# assistant is listening.
VOICE_CUES_ENABLED = _bool("VOICE_CUES_ENABLED", "true")

# How long to wait for the user to START speaking before returning to idle.
# Distinct from VOICE_SILENCE_MS, which ends an utterance already in progress.
VOICE_LISTEN_TIMEOUT_SEC = float(os.getenv("VOICE_LISTEN_TIMEOUT_SEC", "8.0"))

# Global listener, so a keypress in any application cuts speech.
VOICE_TYPING_BARGE_IN = _bool("VOICE_TYPING_BARGE_IN", "true")

# Speak a trailing fragment once the token stream has been quiet this long.
# Gemini holds the stream open well past the final token, so without this a
# reply ending without punctuation waits out that tail. Must stay comfortably
# above the gap between real deltas (measured max 108ms) or it splits output.
VOICE_TTS_IDLE_FLUSH_MS = int(os.getenv("VOICE_TTS_IDLE_FLUSH_MS", "500"))

VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "small.en")
VOICE_STT_DEVICE = os.getenv("VOICE_STT_DEVICE", "auto")  # auto | cuda | cpu
VOICE_SILENCE_MS = int(os.getenv("VOICE_SILENCE_MS", "600"))
VOICE_MAX_UTTERANCE_SEC = int(os.getenv("VOICE_MAX_UTTERANCE_SEC", "30"))

# Optional cheaper model for spoken turns only. Empty = use GEMINI_MODEL.
VOICE_MODEL = os.getenv("VOICE_MODEL", "")

VAD_MODEL_PATH = os.path.join(BASE_DIR, "models", "silero", "silero_vad.onnx")

TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro")  # kokoro | elevenlabs
KOKORO_MODEL_DIR = os.path.join(BASE_DIR, os.getenv("KOKORO_MODEL_DIR", "./models/kokoro"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
# onnxruntime intra-op threads. The default (all cores) is slower than a small
# cap on high-core machines.
KOKORO_THREADS = int(os.getenv("KOKORO_THREADS", "4"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# --- Screen context ----------------------------------------------------------

VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash")

# PyQt6 always-on-top overlay for security toasts and element highlighting.
# Turn off for headless environments, where Qt would fail to initialise.
OVERLAY_ENABLED = _bool("OVERLAY_ENABLED", "true")

# Bounds on the uiautomation element-tree walk. A pathological window can
# otherwise return megabytes of text.
ACCESSIBILITY_MAX_DEPTH = int(os.getenv("ACCESSIBILITY_MAX_DEPTH", "6"))
ACCESSIBILITY_MAX_ELEMENTS = int(os.getenv("ACCESSIBILITY_MAX_ELEMENTS", "150"))

# Ephemeral: written only during an active browser task, wiped when it ends.
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# --- Browser automation ------------------------------------------------------

# Gate 0. Most "browse this page" requests are really "read this page", and a
# static fetch answers them for a fraction of the cost and attack surface.
NO_BROWSER_GATE_ENABLED = _bool("NO_BROWSER_GATE_ENABLED", "true")
NO_BROWSER_TIMEOUT_SECONDS = int(os.getenv("NO_BROWSER_TIMEOUT_SECONDS", "15"))
# Jina Reader renders a page server-side and returns markdown - the last rung
# before paying for a browser. Separately switchable because it discloses the
# URL to a third party.
JINA_READER_ENABLED = _bool("JINA_READER_ENABLED", "true")
JINA_READER_ENDPOINT = os.getenv("JINA_READER_ENDPOINT", "https://r.jina.ai/")
# Untrusted content is billed by the token and a hostile page can be any length.
MAX_EXTRACTED_CHARS = int(os.getenv("MAX_EXTRACTED_CHARS", "12000"))

# Patchright stealth requires a persistent context - there is no new_context()
# in this mode - so one profile is shared by every task. The default is a
# dedicated logged-out profile: an injected page that hijacks the agent then
# has no sessions to abuse. Chrome locks a profile while it is open.
BROWSER_USER_DATA_DIR = os.path.join(
    BASE_DIR, os.getenv("BROWSER_USER_DATA_DIR", "./browser_profile")
)
# Patchright's documented config is channel=chrome + headless=False +
# no_viewport=True with NO custom user agent. Overriding those is what makes
# automation detectable.
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "chrome")
BROWSER_HEADLESS = _bool("BROWSER_HEADLESS", "false")
BROWSER_TIMEOUT_SECONDS = int(os.getenv("BROWSER_TIMEOUT_SECONDS", "30"))
# Much shorter than the navigation budget above: waiting 30s to discover an
# element is not clickable makes a bad choice expensive, when re-observing and
# picking again is cheap.
BROWSER_ACTION_TIMEOUT_SECONDS = int(os.getenv("BROWSER_ACTION_TIMEOUT_SECONDS", "8"))

# Wait for network-idle before observing. `domcontentloaded` fires before a JS
# app has content, so without this the agent reasons about "Loading results...".
# A ceiling, not a requirement - polling sites never go idle.
BROWSER_SETTLE_MS = int(os.getenv("BROWSER_SETTLE_MS", "4000"))

# Not cosmetic. no_viewport=True makes the viewport follow the window, and
# Chrome's ~800x600 default makes responsive sites serve their compact layout,
# hiding controls a desktop user would see.
BROWSER_WINDOW_SIZE = os.getenv("BROWSER_WINDOW_SIZE", "1920,1080")

# Wipe cookies and site storage at the start of each task. Off by default: one
# shared profile means tasks inherit stale state, but clearing it also signs the
# user out of everything, and being signed in is what makes an assistant useful.
# Tests force it on.
BROWSER_CLEAR_SITE_DATA = _bool("BROWSER_CLEAR_SITE_DATA", "false")
# Idle browsers hold ~300MB and the Chrome profile lock.
BROWSER_IDLE_SHUTDOWN_SECONDS = int(os.getenv("BROWSER_IDLE_SHUTDOWN_SECONDS", "300"))

# Empty = launch a local browser. Set to a Steel Browser CDP endpoint to route
# through it for sites behind aggressive bot management.
STEEL_BROWSER_ENDPOINT = os.getenv("STEEL_BROWSER_ENDPOINT", "")
RESIDENTIAL_PROXY_URL = os.getenv("RESIDENTIAL_PROXY_URL", "")

# Behavioural mimicry bounds (seconds) applied before each browser action.
BROWSER_MIN_ACTION_DELAY = float(os.getenv("BROWSER_MIN_ACTION_DELAY", "0.25"))
BROWSER_MAX_ACTION_DELAY = float(os.getenv("BROWSER_MAX_ACTION_DELAY", "0.9"))

# The browser roles send a whole page state rather than a chat turn, which on a
# dense application is large enough to push a single call past the 30s budget.
BROWSER_MODEL_TIMEOUT_SECONDS = int(os.getenv("BROWSER_MODEL_TIMEOUT_SECONDS", "90"))

# Sees all untrusted external content, has no tools, emits only schema-valid
# JSON. Flash tier - this is extraction, not reasoning, and it runs per page.
QUARANTINED_MODEL = os.getenv("QUARANTINED_MODEL", "gemini-3.5-flash")
PROMPT_INJECTION_SCAN = _bool("PROMPT_INJECTION_SCAN", "true")

# The Planner decomposes the task and needs the strongest reasoning; Actor and
# Validator make bounded per-step judgments.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", GEMINI_MODEL)
ACTOR_MODEL = os.getenv("ACTOR_MODEL", "gemini-3.5-flash")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "gemini-3.5-flash")
BROWSER_MAX_STEPS = int(os.getenv("BROWSER_MAX_STEPS", "15"))

# Times a task may revise its remaining steps after one misses its goal. The
# Planner writes the whole plan before seeing a page, so on a long task reality
# diverges from the plan. Bounded because each re-plan is also a fresh chance
# for a page to influence planning.
BROWSER_MAX_REPLANS_PER_TASK = int(os.getenv("BROWSER_MAX_REPLANS_PER_TASK", "2"))
# Re-attempts after a step fails with a TRANSIENT error - a timed-out call or a
# 5xx, not a refusal or an uncooperative page. The step re-observes first, so a
# retry resumes from the page's actual state.
BROWSER_MAX_RETRIES_PER_STEP = int(os.getenv("BROWSER_MAX_RETRIES_PER_STEP", "2"))

# Runaway guard on observe->act rounds within one step, not the usual limit -
# that is BROWSER_MAX_STALLS_PER_STEP. Any fixed per-step count is wrong
# somewhere, because the Planner cannot know how many rounds a step takes on a
# site it has not seen, and it is wrong in the direction that kills working
# steps. Raising it is close to free: it is a ceiling, not a quota.
BROWSER_MAX_ITERATIONS_PER_STEP = int(os.getenv("BROWSER_MAX_ITERATIONS_PER_STEP", "20"))

# Consecutive rounds with no visible effect before a step is abandoned. The
# primary control on step length: keep working while the page keeps responding.
# Not too tight - focusing a field or opening a portal-rendered menu
# legitimately changes nothing measurable.
BROWSER_MAX_STALLS_PER_STEP = int(os.getenv("BROWSER_MAX_STALLS_PER_STEP", "4"))

# Critic vetoes tolerated per step. Separate from the iteration budget on
# purpose: a veto is not an attempt at the page, so charging it to the same
# budget lets the safety layer starve the step it supervises. Still capped, or a
# Critic that refuses everything loops forever.
BROWSER_MAX_VETOES_PER_STEP = int(os.getenv("BROWSER_MAX_VETOES_PER_STEP", "3"))

# Facts one browser task established, available to the next. Session-scoped, so
# it holds the task in hand rather than a figure that was true last week. URLs
# and hostnames are stripped before the Planner sees them, so a page cannot
# nominate a domain for a later task's Origin Set. See browser/ledger.py.
BROWSER_LEDGER_ENABLED = _bool("BROWSER_LEDGER_ENABLED", "true")

# Independent steps pinned to different sites run concurrently. ~82% of a step
# is waiting on a model, so those waits overlap even though the browser thread
# still serialises page operations.
BROWSER_PARALLEL_ENABLED = _bool("BROWSER_PARALLEL_ENABLED", "true")

# Each branch is a real Chrome tab plus its own model calls, so this is not
# free to raise.
BROWSER_MAX_PARALLEL_BRANCHES = int(os.getenv("BROWSER_MAX_PARALLEL_BRANCHES", "3"))

# DOM->action cache: a repeated workflow on an unchanged page replays its
# selectors with zero LLM calls.
BROWSER_DB_PATH = os.path.join(BASE_DIR, "memory_db", "browser.db")
ACTION_CACHE_ENABLED = _bool("ACTION_CACHE_ENABLED", "true")
ACTION_CACHE_TTL_DAYS = int(os.getenv("ACTION_CACHE_TTL_DAYS", "7"))

# Confirmation is reserved for genuinely irreversible actions. Prompting on
# every click would defeat the point of an agent.
CONFIRM_HIGH_RISK = _bool("CONFIRM_HIGH_RISK", "true")

# Optional: without a key, URL redaction still happens and only the reputation
# check is skipped.
SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY", "")
