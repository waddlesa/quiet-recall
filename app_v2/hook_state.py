from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import unicodedata


MODE_MARKER = re.compile(r"^\s*#(研究|日常)(?=\s|$)")


@dataclass(frozen=True)
class HookTurn:
    mode: str | None
    query: str
    marker_only: bool


class HookModeStore:
    """Persist only the manual daily/research switch for a Codex chat.

    Raw session identifiers and prompts are never stored.  A missing or broken
    state store yields ``mode=None``: ordinary recall remains available, while
    the v2 core refuses every vault search for that turn.
    """

    def __init__(self, database_path: Path, salt_path: Path) -> None:
        self.database_path = database_path
        self.salt_path = salt_path
        self._lock = threading.RLock()
        self._salt = self._load_or_create_salt()

    def _load_or_create_salt(self) -> bytes:
        if self.salt_path.is_file():
            return self.salt_path.read_bytes()
        self.salt_path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_bytes(32)
        temporary = self.salt_path.with_suffix(self.salt_path.suffix + ".tmp")
        temporary.write_bytes(value)
        temporary.replace(self.salt_path)
        return value

    def _session_hash(self, session_id: str) -> str:
        return hashlib.sha256(self._salt + session_id.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hook_mode (
                session_hash TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                mode_date TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    @staticmethod
    def _local_date() -> str:
        return datetime.now(timezone(timedelta(hours=8))).date().isoformat()

    def resolve(self, session_id: str | None, prompt: str) -> HookTurn:
        normalized = unicodedata.normalize("NFKC", prompt)
        match = MODE_MARKER.match(normalized)
        marker_mode = None
        query = prompt
        if match:
            marker_mode = "research" if match.group(1) == "研究" else "daily"
            query = normalized[match.end() :].lstrip()

        if not session_id:
            return HookTurn(marker_mode, query, bool(match and not query))

        local_date = self._local_date()
        now = datetime.now(timezone.utc).isoformat()
        session_hash = self._session_hash(session_id)
        try:
            with self._lock, closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT mode, mode_date FROM hook_mode WHERE session_hash=?",
                    (session_hash,),
                ).fetchone()
                mode = "daily"
                if row is not None and str(row[1]) == local_date:
                    mode = str(row[0])
                if marker_mode is not None:
                    mode = marker_mode
                connection.execute(
                    """
                    INSERT INTO hook_mode(session_hash, mode, mode_date, updated_at)
                    VALUES(?,?,?,?)
                    ON CONFLICT(session_hash) DO UPDATE SET
                        mode=excluded.mode,
                        mode_date=excluded.mode_date,
                        updated_at=excluded.updated_at
                    """,
                    (session_hash, mode, local_date, now),
                )
                connection.commit()
            return HookTurn(mode, query, bool(match and not query))
        except (OSError, sqlite3.Error, ValueError):
            # Privacy side fails closed in the service because mode=None denies
            # vault access.  Ordinary retrieval remains product-fail-open.
            return HookTurn(None, query, bool(match and not query))
