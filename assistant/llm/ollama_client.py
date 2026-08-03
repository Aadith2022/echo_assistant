"""Minimal Ollama client for the local Critic backend (private/desktop mode).

Uses Ollama's REST API with schema-forced JSON output so a local Gemma model
returns the same structured Verdict the cloud Critic does. Talks over `requests`
(already a project dependency) to avoid pulling in the `ollama` package.
"""

import requests

import config


def chat_structured(system_prompt: str, user_prompt: str, schema: dict,
                    model: str | None = None, base_url: str | None = None,
                    timeout: float = 30.0) -> str:
    """Send a chat request and force the response to match `schema`.

    Returns the model's message content as a JSON string matching the schema.
    Raises on connection error / non-200 so the caller can fall back.
    """
    model = model or config.OLLAMA_CRITIC_MODEL
    base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": schema,   # Ollama constrains output to this JSON schema
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
