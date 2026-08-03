import json
import logging
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types
import config
from tools.search import web_search
from tools.weather import get_weather
from tools.calendar_tool import get_upcoming_events
from tools.memory_tool import remember_fact, recall_memories
from tools.screen_tool import get_screen_context, see_screen
from guardrails.user_alignment_critic import Critic
from guardrails.audit_log import log_decision
from guardrails.risk_classifier import classify, needs_critic
from vision.overlay import overlay

logger = logging.getLogger(__name__)

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for current information and return a summary with sources.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city or location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name, e.g. 'London'."}},
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_upcoming_events",
        "description": "List the user's upcoming Google Calendar events.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Number of events to return."}
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "Save a durable fact about the user (name, preferences, personal details) "
            "to long-term memory so it can be recalled in future conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The fact to remember, as a plain sentence."}},
            "required": ["text"],
        },
    },
    {
        "type": "function",
        "name": "get_screen_context",
        "description": (
            "See what window/application the user currently has focused and its UI "
            "elements (buttons, text, fields), as a fast text description. No screenshot "
            "is taken. Use this first for anything about what the user is looking at."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "see_screen",
        "description": (
            "Take a screenshot of the user's focused window and visually analyze it. "
            "Only use this when get_screen_context's text description wouldn't be enough "
            "- e.g. the user is asking about an image, video, or visual layout on screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to look for or answer about the screen. Leave empty for a general description.",
                }
            },
            "required": [],
        },
    },
]

# recall_memories is intentionally NOT registered as a callable tool: relevant
# memories are already auto-injected into every turn's prompt in send(), so
# exposing it to the model only invites a redundant Critic + API round trip for
# context it already has.
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_weather": get_weather,
    "get_upcoming_events": get_upcoming_events,
    "remember_fact": remember_fact,
    "get_screen_context": get_screen_context,
    "see_screen": see_screen,
}

# Tools whose result text carries nothing the model needs to see - a turn made
# up entirely of these can skip the result-submission round trip.
TRIVIAL_RESULT_TOOLS = {"remember_fact"}


@dataclass
class _StreamedCall:
    """A function_call reassembled from streamed step/delta events."""

    id: str
    name: str
    raw_arguments: str = ""
    arguments: dict = field(default_factory=dict)
    type: str = "function_call"

    def finalize(self) -> None:
        try:
            self.arguments = json.loads(self.raw_arguments) if self.raw_arguments else {}
        except json.JSONDecodeError:
            logger.error("Could not parse streamed arguments for %s: %r", self.name, self.raw_arguments)
            self.arguments = {}


@dataclass
class _StreamedInteraction:
    """Mirrors the non-streaming interaction shape the tool loop expects."""

    id: str | None
    steps: list
    output_text: str

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Use the available tools when you need "
    "current information (web search, weather, calendar). Call remember_fact when the "
    "user shares a durable personal fact worth keeping (name, preferences, personal "
    "details). When the user asks what they're looking at or doing on their screen, "
    "call get_screen_context first - it's fast and text-only. Only call see_screen, "
    "which takes an actual screenshot, if get_screen_context's element tree wouldn't "
    "answer the question (e.g. an image, video, or visual layout). Otherwise answer "
    "directly."
)

# Spoken turns get different style rules. Markdown is meaningless out loud - a
# text-to-speech voice reads "**97%**" with the asterisks - and a bulleted
# report that scans well on screen is exhausting to listen to.
VOICE_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nThis reply will be read aloud by a text-to-speech voice, so write it to "
    "be *heard*, not read:\n"
    "- Plain conversational prose only. Never use markdown: no asterisks, bold, "
    "bullet points, numbered lists, headings, tables or code blocks.\n"
    "- Be brief - usually one to three sentences. Lead with the answer. If there "
    "is more worth saying, offer it ('want the details?') instead of reciting it.\n"
    "- Sound like a person talking. Contractions are good; section headers are not.\n"
    "- Write units and symbols the way you would say them: 'twenty-two degrees', "
    "not '22°C'.\n"
    "- Never read out URLs. Name the source instead.\n"
    "- No emoji."
)


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_SECONDS * 1000),
        )
        self.model = config.GEMINI_MODEL
        self.last_interaction_id = None
        self.critic = Critic() if config.CRITIC_ENABLED else None

    def _create_interaction(self, **kwargs):
        """Wrap interactions.create with timing/logging so a slow or stalled
        call is visible instead of silently hanging."""
        start = time.monotonic()
        logger.info("Gemini interactions.create started (model=%s)", self.model)
        try:
            interaction = self.client.interactions.create(model=self.model, **kwargs)
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Gemini interactions.create failed after %.1fs: %s", elapsed, e)
            raise
        elapsed = time.monotonic() - start
        logger.info("Gemini interactions.create finished in %.1fs", elapsed)
        return interaction

    def _stream_interaction(self, on_delta, **kwargs):
        """Stream an interaction, emitting text tokens to on_delta as they arrive.

        The streamed `interaction.completed` event carries only the interaction
        id - its `steps` and `output_text` are empty - so we reassemble both
        from the event stream: text from `text` deltas, and function calls from
        `step.start` (name/id) plus `arguments_delta` fragments. The returned
        object mirrors the non-streaming interaction shape so the tool loop is
        identical either way. Only text is streamed; tool calls still run
        through the normal loop, so streaming never races action execution or
        bypasses Critic review.
        """
        start = time.monotonic()
        interaction_id = None
        text_parts = []
        calls = []

        for event in self.client.interactions.create(model=self.model, stream=True, **kwargs):
            etype = event.event_type

            if etype == "step.start":
                step = getattr(event, "step", None)
                if getattr(step, "type", None) == "function_call":
                    calls.append(_StreamedCall(id=step.id, name=step.name))

            elif etype == "step.delta":
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", None)
                if dtype == "text":
                    text_parts.append(delta.text)
                    on_delta(delta.text)
                elif dtype == "arguments_delta" and calls:
                    calls[-1].raw_arguments += delta.arguments or ""

            elif etype == "interaction.completed":
                interaction_id = event.interaction.id

        for call in calls:
            call.finalize()

        logger.info("Gemini stream finished in %.1fs", time.monotonic() - start)
        return _StreamedInteraction(
            id=interaction_id,
            steps=calls,
            output_text="".join(text_parts),
        )

    def _next_interaction(self, on_delta, **kwargs):
        """Stream when a delta callback is supplied, otherwise do a blocking call."""
        if on_delta:
            return self._stream_interaction(on_delta, **kwargs)
        return self._create_interaction(**kwargs)

    def send(self, message: str, on_delta=None, spoken: bool = False) -> str:
        user_intent = message
        system_prompt = VOICE_SYSTEM_PROMPT if spoken else SYSTEM_PROMPT

        memories = recall_memories(message, k=3)
        augmented_message = f"Relevant memories about the user:\n{memories}\n\nUser message: {message}"

        interaction = self._next_interaction(
            on_delta,
            input=augmented_message,
            tools=TOOL_DECLARATIONS,
            system_instruction=system_prompt,
            previous_interaction_id=self.last_interaction_id,
        )

        while True:
            function_calls = [s for s in interaction.steps if s.type == "function_call"]
            if not function_calls:
                break

            results = []
            for call in function_calls:
                tier = classify(call.name, call.arguments)

                # LOW-risk actions are reversible/read-only and auto-execute:
                # skipping the Critic here removes a full round trip from the
                # common path. Everything else is reviewed.
                if self.critic is not None and needs_critic(tier):
                    verdict = self.critic.review(user_intent, call.name, call.arguments)
                    log_decision(user_intent, call.name, call.arguments, verdict.decision, verdict.reason)

                    if verdict.decision == "VETO":
                        try:
                            overlay.show_toast(
                                f"Blocked: {call.name} - {verdict.reason}", level="veto"
                            )
                        except Exception:
                            logger.exception("Overlay toast failed; continuing without it")
                        results.append(
                            {
                                "type": "function_result",
                                "name": call.name,
                                "call_id": call.id,
                                "result": [
                                    {
                                        "type": "text",
                                        "text": f"[blocked by User Alignment Critic] {verdict.reason}",
                                    }
                                ],
                            }
                        )
                        continue

                func = TOOL_FUNCTIONS[call.name]
                output = func(**call.arguments)
                results.append(
                    {
                        "type": "function_result",
                        "name": call.name,
                        "call_id": call.id,
                        "result": [{"type": "text", "text": str(output)}],
                    }
                )

            # Trivial-output short-circuit: when every call was a fire-and-forget
            # tool whose result carries no information the model needs to phrase
            # (e.g. remember_fact), skip the result-submission round trip and
            # reply directly. Trade-off: this turn isn't added to the interaction
            # chain, so the exchange itself isn't in history - but the stored
            # fact is, and gets auto-injected on the next turn.
            if all(c.name in TRIVIAL_RESULT_TOOLS for c in function_calls):
                reply = "Got it - I'll remember that."
                if on_delta:
                    on_delta(reply)
                return reply

            # system_instruction is interaction-scoped, so it has to be resent
            # on every hop - otherwise a tool turn silently reverts to the
            # written-style prompt and speaks markdown again.
            interaction = self._next_interaction(
                on_delta,
                input=results,
                tools=TOOL_DECLARATIONS,
                system_instruction=system_prompt,
                previous_interaction_id=interaction.id,
            )

        self.last_interaction_id = interaction.id
        # When streaming, text has already been emitted token-by-token as it
        # arrived; return it for callers that want the full string.
        return interaction.output_text
