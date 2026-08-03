import argparse
import logging

import config
from agent import Agent
from vision.overlay import overlay


# Our own packages - these are the ones worth hearing from under --verbose.
_OWN_LOGGERS = ("agent", "llm", "voice", "memory", "guardrails", "tools")

# Third-party loggers that are pure noise at INFO. Raising sentence_transformers
# above INFO also suppresses its "Batches: 100%|...|" progress bar, which it
# shows based on its own effective log level - so --verbose was inadvertently
# turning that bar on.
_NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "faster_whisper": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "chromadb": logging.WARNING,
    "openwakeword": logging.WARNING,
    "numba": logging.WARNING,
    # espeak routinely emits "words count mismatch" on ordinary text; it is
    # advisory and there is nothing to act on.
    "phonemizer": logging.ERROR,
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if verbose:
        for name in _OWN_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
    for name, level in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(level)


def run_text(agent: Agent) -> None:
    print("Echo - text mode (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            agent.respond(user_input, on_delta=lambda t: print(t, end="", flush=True))
            print()
        except Exception as e:
            print(f"\n[error] {e}")


def run_voice(agent: Agent) -> None:
    from voice.session import VoiceSession

    print("Echo - voice mode (loading models...)")
    session = VoiceSession(agent)
    session.start()

    if session.activation == "wakeword":
        activation = f"say '{config.VOICE_WAKE_WORD.replace('_', ' ')}'"
    else:
        activation = f"press {config.VOICE_HOTKEY}"
    print(f"Ready. To speak, {activation}. You can also just type. ('exit' to quit)")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)
            try:
                session.submit_turn(user_input, spoken=False)
            except Exception as e:
                print(f"\n[error] {e}")
    finally:
        session.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo - personal AI assistant")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable the voice pipeline (typing still works alongside it).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log pipeline detail.")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    overlay.start()
    try:
        agent = Agent()
        if args.voice or config.VOICE_ENABLED:
            run_voice(agent)
        else:
            run_text(agent)
    finally:
        overlay.stop()


if __name__ == "__main__":
    main()
