from llm.gemini_client import GeminiClient


class Agent:
    def __init__(self):
        self.llm = GeminiClient()

    def respond(self, user_input: str, on_delta=None, spoken: bool = False) -> str:
        """Run one turn.

        `spoken` selects the voice system prompt, which asks for short
        conversational prose instead of markdown - a reply written for the
        screen is read aloud with its asterisks intact.
        """
        return self.llm.send(user_input, on_delta=on_delta, spoken=spoken)
