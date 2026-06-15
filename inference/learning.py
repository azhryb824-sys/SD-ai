import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEARNING_DIR = ROOT / "data" / "learning"
CONVERSATIONS_PATH = LEARNING_DIR / "conversations.jsonl"
FEEDBACK_PATH = LEARNING_DIR / "feedback.jsonl"
LEARNED_EXAMPLES_PATH = LEARNING_DIR / "approved_examples.jsonl"

_lock = threading.Lock()
_pending_interactions = {}


def redact_sensitive_text(text):
    redacted = str(text)
    redacted = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[بريد محجوب]",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)",
        "[رقم محجوب]",
        redacted,
    )
    redacted = re.sub(
        r"\b(?:\d[ -]?){13,19}\b",
        "[بيانات رقمية محجوبة]",
        redacted,
    )
    return redacted[:5000]


def _append_jsonl(path, payload):
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_interaction(message, response, history, session_id, consent):
    if not consent:
        return None

    interaction_id = uuid.uuid4().hex
    payload = {
        "interaction_id": interaction_id,
        "session_id": re.sub(r"[^\w-]", "", session_id or "")[:80],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": redact_sensitive_text(message),
        "response": redact_sensitive_text(response),
        "history": [
            {
                "role": item.get("role"),
                "content": redact_sensitive_text(item.get("content", "")),
            }
            for item in history[-6:]
            if item.get("role") in {"user", "assistant"}
        ],
    }
    with _lock:
        _append_jsonl(CONVERSATIONS_PATH, payload)
        _pending_interactions[interaction_id] = payload
        if len(_pending_interactions) > 1000:
            oldest_id = next(iter(_pending_interactions))
            _pending_interactions.pop(oldest_id, None)
    return interaction_id


def submit_feedback(interaction_id, helpful):
    with _lock:
        interaction = _pending_interactions.get(interaction_id)
        if not interaction:
            return False, False

        feedback = {
            "interaction_id": interaction_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "helpful": bool(helpful),
        }
        _append_jsonl(FEEDBACK_PATH, feedback)

        learned = False
        if helpful:
            example = {
                "prompt": interaction["prompt"],
                "response": interaction["response"],
                "approved_at": feedback["created_at"],
                "source": "user_feedback",
            }
            _append_jsonl(LEARNED_EXAMPLES_PATH, example)
            learned = True

        _pending_interactions.pop(interaction_id, None)
        return True, learned


def load_learned_examples(limit=500):
    if not LEARNED_EXAMPLES_PATH.exists():
        return []
    examples = []
    try:
        with LEARNED_EXAMPLES_PATH.open(encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("prompt") and item.get("response"):
                    examples.append(item)
    except OSError:
        return []
    return examples[-limit:]
