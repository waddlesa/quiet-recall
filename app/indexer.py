from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable

import numpy as np

from .config import ProjectConfig
from .embedding_backend import EmbeddingBackend
from .entity_resolver import EntityResolver
from .ids import chunk_id, event_id, normalize_key, normalize_text
from .markdown_parser import MarkdownChunk, parse_frontmatter, parse_markdown
from .source_registry import SourceRecord, SourceRegistry


INDEX_VERSION = "3"
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})(?:日)?|(?<!\d)(\d{2})-(\d{2})(?!\d)")


MIGRATION = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_file (
    source_id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    namespace TEXT NOT NULL,
    canonical_group TEXT NULL,
    role TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_retrieval (
    source_id TEXT PRIMARY KEY REFERENCES source_file(source_id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK(mode IN ('full_text', 'strict_surface')),
    surface TEXT NOT NULL,
    cues_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event (
    event_id TEXT PRIMARY KEY,
    event_key TEXT UNIQUE NOT NULL,
    date_key TEXT NULL,
    title_key TEXT NOT NULL,
    entity_key TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES source_file(source_id) ON DELETE CASCADE,
    event_id TEXT NULL REFERENCES event(event_id),
    heading_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    excerpt TEXT NOT NULL,
    normalized_hash TEXT NOT NULL,
    entity_ids_json TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    index_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunk(source_id);
CREATE INDEX IF NOT EXISTS idx_chunk_event ON chunk(event_id);
CREATE TABLE IF NOT EXISTS entity (
    entity_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    identity_card TEXT NULL
);
CREATE TABLE IF NOT EXISTS entity_alias (
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    alias_type TEXT NOT NULL CHECK(alias_type IN ('strong', 'ambiguous')),
    PRIMARY KEY(alias, entity_id, alias_type)
);
"""


@dataclass(frozen=True)
class PreparedChunk:
    chunk_id: str
    source_id: str
    event_id: str | None
    event_key: str | None
    date_key: str
    title_key: str
    entity_ids: tuple[str, ...]
    heading_path: str
    start_line: int
    end_line: int
    excerpt: str
    normalized_hash: str

    @property
    def embed_text(self) -> str:
        return f"{self.heading_path}。{self.excerpt}"


@dataclass(frozen=True)
class SyncReport:
    dry_run: bool
    total_sources: int
    changed_sources: int
    unchanged_sources: int
    removed_sources: int
    total_chunks: int
    encoded_chunks: int
    reused_chunks: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _date_key(text: str) -> str:
    match = DATE_RE.search(text)
    if not match:
        return ""
    if match.group(1):
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return f"{match.group(4)}-{match.group(5)}"


def _retrieval_policy(path: Path) -> tuple[str, str, str]:
    frontmatter = parse_frontmatter(path)
    raw = frontmatter.get("retrieval", {})
    if raw in ({}, None):
        return "full_text", "", "[]"
    if not isinstance(raw, dict):
        raise ValueError(f"retrieval frontmatter must be a mapping: {path}")
    mode = str(raw.get("mode", "full_text")).strip()
    if mode not in {"full_text", "strict_surface"}:
        raise ValueError(f"unknown retrieval mode {mode!r}: {path}")
    raw_cues = raw.get("cues", [])
    if not isinstance(raw_cues, list):
        raise ValueError(f"retrieval.cues must be a list: {path}")
    cues = tuple(dict.fromkeys(str(item).strip() for item in raw_cues if str(item).strip()))
    if mode == "strict_surface" and not cues:
        raise ValueError(f"strict_surface requires at least one retrieval cue: {path}")
    return mode, "；".join(cues), json.dumps(cues, ensure_ascii=False)


def _split_long_chunk(chunk: MarkdownChunk, size: int = 480, overlap: int = 80) -> Iterable[MarkdownChunk]:
    text = chunk.text
    if len(text) <= size:
        yield chunk
        return
    step = size - overlap
    for segment, start in enumerate(range(0, len(text), step)):
        excerpt = text[start : start + size]
        if not excerpt:
            continue
        yield MarkdownChunk(
            heading_path=f"{chunk.heading_path} / segment-{segment + 1}",
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=excerpt,
            block_type=chunk.block_type,
        )


class MemoryIndexer:
    def __init__(
        self,
        config: ProjectConfig,
        backend: EmbeddingBackend,
        database_path: Path | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.database_path = database_path or config.project_root / "state" / "memory-index.sqlite3"
        self.registry = SourceRegistry(config)
        self.entities = EntityResolver(config.entities)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(MIGRATION)
        return connection

    def _prepare(self, source: SourceRecord) -> list[PreparedChunk]:
        report = parse_markdown(source.path)
        prepared: list[PreparedChunk] = []
        for block in report.chunks:
            event_entities = self.entities.find_strong_entities(block.heading_path + "\n" + block.text)
            event_title = block.heading_path.split(" / ")[-1]
            event_date = _date_key(block.heading_path + " " + block.text[:120])
            can_name_event = bool(
                event_date
                or event_entities
                or normalize_key(event_title) != normalize_key(source.path.name)
            )
            for item in _split_long_chunk(block):
                excerpt = normalize_text(item.text)
                if not excerpt:
                    continue
                entities = self.entities.find_strong_entities(item.heading_path + "\n" + excerpt)
                stable_event_id: str | None = None
                stable_event_key: str | None = None
                if can_name_event:
                    stable_event_id, stable_event_key = event_id(
                        event_date, event_title, list(event_entities)
                    )
                prepared.append(
                    PreparedChunk(
                        chunk_id=chunk_id(source.source_id, item.heading_path, excerpt),
                        source_id=source.source_id,
                        event_id=stable_event_id,
                        event_key=stable_event_key,
                        date_key=event_date,
                        title_key=normalize_key(event_title),
                        entity_ids=entities,
                        heading_path=item.heading_path,
                        start_line=item.start_line,
                        end_line=item.end_line,
                        excerpt=excerpt,
                        normalized_hash=_sha256_bytes(excerpt.encode("utf-8")),
                    )
                )
        return prepared

    def _sync_entities(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM entity_alias")
        connection.execute("DELETE FROM entity")
        for entity_id, item in self.config.entities.get("entities", {}).items():
            connection.execute(
                "INSERT INTO entity(entity_id, display_name, kind, identity_card) VALUES(?,?,?,?)",
                (
                    entity_id,
                    str(item.get("display_name", entity_id)),
                    str(item.get("kind", "unknown")),
                    item.get("identity_card"),
                ),
            )
            for alias_type, field in (("strong", "strong_aliases"), ("ambiguous", "ambiguous_aliases")):
                for alias in item.get(field, []):
                    connection.execute(
                        "INSERT INTO entity_alias(alias, entity_id, alias_type) VALUES(?,?,?)",
                        (str(alias), entity_id, alias_type),
                    )

    def sync(self, dry_run: bool = False) -> SyncReport:
        sources = [record for record in self.registry.records if record.parse]
        existing_hashes: dict[str, str] = {}
        existing_model = ""
        existing_index_version = ""
        if self.database_path.is_file():
            with closing(sqlite3.connect(self.database_path)) as read_connection:
                read_connection.row_factory = sqlite3.Row
                try:
                    existing_hashes = {
                        row["source_id"]: row["content_hash"]
                        for row in read_connection.execute("SELECT source_id, content_hash FROM source_file")
                    }
                    row = read_connection.execute(
                        "SELECT value FROM index_meta WHERE key='embedding_model'"
                    ).fetchone()
                    existing_model = str(row[0]) if row else ""
                    row = read_connection.execute(
                        "SELECT value FROM index_meta WHERE key='index_version'"
                    ).fetchone()
                    existing_index_version = str(row[0]) if row else ""
                except sqlite3.Error:
                    existing_hashes = {}
                    existing_model = ""
                    existing_index_version = ""

        source_hashes = {source.source_id: _source_hash(source.path) for source in sources}
        model_changed = bool(existing_model and existing_model != self.backend.name)
        index_changed = bool(existing_index_version and existing_index_version != INDEX_VERSION)
        changed = [
            source
            for source in sources
            if model_changed
            or index_changed
            or existing_hashes.get(source.source_id) != source_hashes[source.source_id]
        ]
        stale = set(existing_hashes) - set(source_hashes)
        if dry_run:
            return SyncReport(
                dry_run=True,
                total_sources=len(sources),
                changed_sources=len(changed),
                unchanged_sources=len(sources) - len(changed),
                removed_sources=len(stale),
                total_chunks=0,
                encoded_chunks=0,
                reused_chunks=0,
            )

        connection = self._connect()
        encoded = 0
        reused = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._sync_entities(connection)
            for source_id in stale:
                connection.execute("DELETE FROM source_file WHERE source_id=?", (source_id,))

            for source in changed:
                prepared = self._prepare(source)
                previous = {
                    row["chunk_id"]: (bytes(row["embedding"]), int(row["embedding_dim"]), row["embedding_model"])
                    for row in connection.execute(
                        "SELECT chunk_id, embedding, embedding_dim, embedding_model FROM chunk WHERE source_id=?",
                        (source.source_id,),
                    )
                }
                missing = [
                    item
                    for item in prepared
                    if item.chunk_id not in previous or previous[item.chunk_id][2] != self.backend.name
                ]
                vectors = self.backend.encode_documents([item.embed_text for item in missing])
                encoded_vectors = {
                    item.chunk_id: vectors[index].astype(np.float32, copy=False)
                    for index, item in enumerate(missing)
                }

                now = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT INTO source_file(source_id,path,namespace,canonical_group,role,content_hash,mtime_ns,indexed_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(source_id) DO UPDATE SET
                      path=excluded.path, namespace=excluded.namespace,
                      canonical_group=excluded.canonical_group, role=excluded.role,
                      content_hash=excluded.content_hash, mtime_ns=excluded.mtime_ns,
                      indexed_at=excluded.indexed_at
                    """,
                    (
                        source.source_id,
                        str(source.path),
                        source.namespace,
                        source.canonical_group,
                        source.role,
                        source_hashes[source.source_id],
                        source.path.stat().st_mtime_ns,
                        now,
                    ),
                )
                connection.execute("DELETE FROM chunk WHERE source_id=?", (source.source_id,))
                for item in prepared:
                    if item.event_id and item.event_key:
                        connection.execute(
                            "INSERT OR IGNORE INTO event(event_id,event_key,date_key,title_key,entity_key) VALUES(?,?,?,?,?)",
                            (
                                item.event_id,
                                item.event_key,
                                item.date_key or None,
                                item.title_key,
                                ",".join(item.entity_ids),
                            ),
                        )
                    cached = previous.get(item.chunk_id)
                    if item.chunk_id in encoded_vectors:
                        vector = encoded_vectors[item.chunk_id]
                        blob = vector.tobytes()
                        dimension = int(vector.shape[0])
                        encoded += 1
                    elif cached:
                        blob, dimension, _ = cached
                        reused += 1
                    else:
                        raise RuntimeError(f"missing embedding for {item.chunk_id}")
                    connection.execute(
                        """
                        INSERT INTO chunk(
                          chunk_id,source_id,event_id,heading_path,start_line,end_line,excerpt,
                          normalized_hash,entity_ids_json,embedding,embedding_dim,embedding_model,index_version
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            item.chunk_id,
                            item.source_id,
                            item.event_id,
                            item.heading_path,
                            item.start_line,
                            item.end_line,
                            item.excerpt,
                            item.normalized_hash,
                            json.dumps(item.entity_ids, ensure_ascii=False),
                            blob,
                            dimension,
                            self.backend.name,
                            INDEX_VERSION,
                        ),
                    )

            # Retrieval policy is source metadata, not memory prose. Refresh
            # it for every registered source so older indexes gain the policy
            # table without re-encoding unchanged body chunks.
            for source in sources:
                mode, surface, cues_json = _retrieval_policy(source.path)
                connection.execute(
                    """
                    INSERT INTO source_retrieval(source_id,mode,surface,cues_json)
                    VALUES(?,?,?,?)
                    ON CONFLICT(source_id) DO UPDATE SET
                      mode=excluded.mode, surface=excluded.surface,
                      cues_json=excluded.cues_json
                    """,
                    (source.source_id, mode, surface, cues_json),
                )

            connection.execute(
                "INSERT INTO index_meta(key,value) VALUES('embedding_model',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.backend.name,),
            )
            connection.execute(
                "INSERT INTO index_meta(key,value) VALUES('index_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (INDEX_VERSION,),
            )
            connection.execute(
                "DELETE FROM event WHERE event_id NOT IN (SELECT DISTINCT event_id FROM chunk WHERE event_id IS NOT NULL)"
            )
            connection.commit()
            total_chunks = int(connection.execute("SELECT COUNT(*) FROM chunk").fetchone()[0])
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return SyncReport(
            dry_run=False,
            total_sources=len(sources),
            changed_sources=len(changed),
            unchanged_sources=len(sources) - len(changed),
            removed_sources=len(stale),
            total_chunks=total_chunks,
            encoded_chunks=encoded,
            reused_chunks=reused,
        )

    def status(self) -> dict[str, object]:
        if not self.database_path.is_file():
            return {"exists": False, "path": str(self.database_path)}
        try:
            with closing(sqlite3.connect(self.database_path)) as connection:
                return {
                    "exists": True,
                    "path": str(self.database_path),
                    "sources": int(connection.execute("SELECT COUNT(*) FROM source_file").fetchone()[0]),
                    "chunks": int(connection.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]),
                    "events": int(connection.execute("SELECT COUNT(*) FROM event").fetchone()[0]),
                    "entities": int(connection.execute("SELECT COUNT(*) FROM entity").fetchone()[0]),
                    "embedding_model": connection.execute(
                        "SELECT value FROM index_meta WHERE key='embedding_model'"
                    ).fetchone()[0],
                    "integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
                }
        except sqlite3.Error as exc:
            return {"exists": True, "path": str(self.database_path), "error": type(exc).__name__}
