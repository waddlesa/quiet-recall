from __future__ import annotations

import hashlib
import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic IDs without changing canonical source text."""
    value = unicodedata.normalize("NFKC", text)
    return SPACE_RE.sub(" ", value).strip()


def normalize_key(text: str) -> str:
    return PUNCT_RE.sub("-", normalize_text(text).casefold()).strip("-")


def digest_id(prefix: str, *parts: str, length: int = 24) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts).encode("utf-8")
    return f"{prefix}://{hashlib.sha256(payload).hexdigest()[:length]}"


def chunk_id(source_id: str, heading_path: str, excerpt: str) -> str:
    return digest_id("chunk", source_id, heading_path, normalize_text(excerpt))


def event_id(date_key: str, title_key: str, entity_ids: list[str]) -> tuple[str, str]:
    entity_key = ",".join(sorted(set(entity_ids)))
    event_key = "|".join((normalize_key(date_key), normalize_key(title_key), entity_key))
    return digest_id("event", event_key), event_key
