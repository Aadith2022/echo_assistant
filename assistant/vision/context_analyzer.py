"""Tier 2 screen context: user-triggered screenshot + Gemini vision.

Only runs when the user explicitly asks ("what's on my screen?"). The
screenshot is captured, sent for analysis, and discarded - never written to
disk. Uses its own isolated genai.Client (no tools attached), the same
isolation pattern as guardrails/user_alignment_critic.py, so a vision call
can never itself trigger a tool call.
"""

import base64
import logging

from google import genai
from google.genai import types

import config
from vision.screen_capture import capture_foreground_window

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Describe what's currently visible on the user's screen in a few sentences. "
    "Focus on the application, its content, and anything notable or actionable."
)


class ContextAnalyzer:
    def __init__(self):
        self.client = genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_SECONDS * 1000),
        )
        self.model = config.VISION_MODEL

    def analyze_screen(self, question: str = "") -> str:
        try:
            image_bytes = capture_foreground_window()
        except Exception as e:
            logger.exception("Screen capture failed")
            return f"Could not capture the screen: {e}"

        prompt = question.strip() or _DEFAULT_PROMPT

        interaction = self.client.interactions.create(
            model=self.model,
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                    "mime_type": "image/png",
                },
            ],
        )
        return interaction.output_text


_analyzer = ContextAnalyzer()


def analyze_screen(question: str = "") -> str:
    return _analyzer.analyze_screen(question)
