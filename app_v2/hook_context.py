from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_content_text(item) for item in value)))
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _content_text(value["content"])
    return ""


def _role_messages(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        role = value.get("role")
        if role in {"user", "assistant"} and "content" in value:
            content = _content_text(value.get("content")).strip()
            if content:
                found.append((str(role), content))
            return found
        for nested in value.values():
            found.extend(_role_messages(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_role_messages(nested))
    return found


def latest_assistant_head(
    path: str | Path | None,
    *,
    max_chars: int = 100,
    max_bytes: int = 1_048_576,
) -> str:
    """Return only the bounded beginning of the latest assistant message.

    Codex transcript JSONL is treated as a best-effort input: malformed or
    unavailable records fail closed to an empty working-context candidate.
    The transcript text is never persisted by this helper.
    """

    if not isinstance(path, (str, Path)) or not path or max_chars <= 0:
        return ""
    transcript = Path(path)
    try:
        with transcript.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - max_bytes))
            raw = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""

    if size > max_bytes and "\n" in raw:
        raw = raw.split("\n", 1)[1]

    messages: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for message in _role_messages(record):
            if not messages or messages[-1] != message:
                messages.append(message)

    for role, content in reversed(messages):
        if role == "assistant":
            return content[:max_chars]
    return ""
