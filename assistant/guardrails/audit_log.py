import json
import os
from datetime import datetime, timezone

import config

LOG_PATH = os.path.join(config.BASE_DIR, "logs", "critic_decisions.jsonl")


def log_decision(user_intent: str, action_name: str, action_args: dict, decision: str, reason: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_intent": user_intent,
        "action_name": action_name,
        "action_args": action_args,
        "decision": decision,
        "reason": reason,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
