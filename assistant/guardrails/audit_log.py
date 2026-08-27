import json
import os
import threading
from datetime import datetime, timezone

import config

LOG_PATH = os.path.join(config.BASE_DIR, "logs", "critic_decisions.jsonl")

# Everything that isn't a Critic verdict - injection detections, Origin Set
# blocks, browser actions, confirmations. Separate, so the Critic's decision
# history stays a greppable record of one thing.
EVENT_LOG_PATH = os.path.join(config.BASE_DIR, "logs", "events.jsonl")

# Several threads write here at once, and single-line append writes are
# usually atomic but not guaranteed.
_write_lock = threading.Lock()


def _append(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(entry, default=str) + "\n"
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def log_event(kind: str, **fields) -> None:
    """Record a non-Critic audit event."""
    _append(
        EVENT_LOG_PATH,
        {"timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields},
    )


def log_decision(user_intent: str, action_name: str, action_args: dict, decision: str, reason: str) -> None:
    _append(
        LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_intent": user_intent,
            "action_name": action_name,
            "action_args": action_args,
            "decision": decision,
            "reason": reason,
        },
    )
