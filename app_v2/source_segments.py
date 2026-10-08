from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import tempfile

from app.config import ProjectConfig
from app.ids import chunk_id, normalize_text
from app.indexer import _split_long_chunk
from app.markdown_parser import parse_markdown
from app.source_registry import SourceRegistry

from .candidates import index_build_fingerprint


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class SegmentBuildReport:
    index_build_id: str
    source_count: int
    chunk_count: int
    output_path: Path


class CanonicalSegmentStore:
    """Build-time projection from indexed chunk IDs to canonical segment text.

    The reader deliberately does not parse Markdown or reconstruct segment
    offsets.  This sidecar is derived once with the same parser/splitter used by
    the frozen index, and every generated ID is checked against that index.
    """

    def __init__(self, path: Path, index_build_id: str, segments: dict[str, str]) -> None:
        self.path = path
        self.index_build_id = index_build_id
        self._segments = segments

    def get(self, chunk_id_value: str) -> str | None:
        return self._segments.get(chunk_id_value)

    @classmethod
    def load(cls, path: Path, *, expected_build_id: str) -> "CanonicalSegmentStore":
        connection = sqlite3.connect(path)
        try:
            meta = dict(connection.execute("SELECT key,value FROM segment_meta").fetchall())
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("canonical segment sidecar schema mismatch")
            if meta.get("index_build_id") != expected_build_id:
                raise ValueError("canonical segment sidecar build mismatch")
            segments = dict(
                connection.execute(
                    "SELECT chunk_id,canonical_text FROM canonical_segment"
                ).fetchall()
            )
        finally:
            connection.close()
        return cls(path, expected_build_id, segments)

    @classmethod
    def ensure_current(
        cls,
        config: ProjectConfig,
        *,
        index_path: Path,
        sidecar_path: Path | None = None,
    ) -> "CanonicalSegmentStore":
        output = sidecar_path or config.project_root / "state" / "v2" / "canonical-segments.sqlite3"
        build_id = index_build_fingerprint(index_path)
        if output.is_file():
            try:
                return cls.load(output, expected_build_id=build_id)
            except (OSError, sqlite3.DatabaseError, ValueError):
                pass
        cls.build(config, index_path=index_path, output_path=output)
        return cls.load(output, expected_build_id=build_id)

    @classmethod
    def build(
        cls,
        config: ProjectConfig,
        *,
        index_path: Path,
        output_path: Path,
    ) -> SegmentBuildReport:
        build_id = index_build_fingerprint(index_path)
        connection = sqlite3.connect(index_path)
        connection.row_factory = sqlite3.Row
        try:
            source_rows = connection.execute(
                """
                SELECT source_id,path,content_hash
                FROM source_file
                WHERE role='canonical'
                ORDER BY source_id
                """
            ).fetchall()
            chunk_rows = connection.execute(
                """
                SELECT chunk_id,source_id,heading_path,start_line,end_line,excerpt
                FROM chunk
                WHERE source_id IN (
                    SELECT source_id FROM source_file WHERE role='canonical'
                )
                ORDER BY source_id,start_line,chunk_id
                """
            ).fetchall()
        finally:
            connection.close()

        expected = {str(row["chunk_id"]): row for row in chunk_rows}
        records = {
            record.source_id: record
            for record in SourceRegistry(config).records
            if record.role == "canonical"
        }
        generated: dict[str, tuple[str, str, int, int, str]] = {}
        for source in source_rows:
            source_id_value = str(source["source_id"])
            record = records.get(source_id_value)
            if record is None:
                raise ValueError(f"canonical source missing from registry: {source_id_value}")
            if record.path.resolve() != Path(str(source["path"])).resolve():
                raise ValueError(f"canonical source path mismatch: {source_id_value}")
            for block in parse_markdown(record.path).chunks:
                for item in _split_long_chunk(block):
                    excerpt = normalize_text(item.text)
                    if not excerpt:
                        continue
                    identifier = chunk_id(source_id_value, item.heading_path, excerpt)
                    generated[identifier] = (
                        source_id_value,
                        item.heading_path,
                        item.start_line,
                        item.end_line,
                        item.text,
                    )

        missing = sorted(set(expected) - set(generated))
        unexpected = sorted(set(generated) - set(expected))
        mismatched: list[str] = []
        for identifier in sorted(set(expected).intersection(generated)):
            row = expected[identifier]
            source_id_value, heading, start, end, canonical_text = generated[identifier]
            if (
                source_id_value != str(row["source_id"])
                or heading != str(row["heading_path"])
                or start != int(row["start_line"])
                or end != int(row["end_line"])
                or normalize_text(canonical_text) != str(row["excerpt"])
            ):
                mismatched.append(identifier)
        if missing or unexpected or mismatched:
            raise ValueError(
                "canonical segment projection does not match frozen index: "
                f"missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix="canonical-segments-", suffix=".sqlite3", dir=output_path.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            sidecar = sqlite3.connect(temporary_path)
            try:
                sidecar.executescript(
                    """
                    CREATE TABLE segment_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE canonical_segment (
                        chunk_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        heading_path TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        canonical_text TEXT NOT NULL
                    );
                    """
                )
                sidecar.executemany(
                    "INSERT INTO segment_meta(key,value) VALUES (?,?)",
                    (
                        ("schema_version", SCHEMA_VERSION),
                        ("index_build_id", build_id),
                        ("built_at", datetime.now(timezone.utc).isoformat()),
                        ("source_count", str(len(source_rows))),
                        ("chunk_count", str(len(generated))),
                    ),
                )
                sidecar.executemany(
                    """
                    INSERT INTO canonical_segment(
                        chunk_id,source_id,heading_path,start_line,end_line,canonical_text
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (
                        (identifier, *generated[identifier])
                        for identifier in sorted(generated)
                    ),
                )
                sidecar.commit()
            finally:
                sidecar.close()
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return SegmentBuildReport(
            index_build_id=build_id,
            source_count=len(source_rows),
            chunk_count=len(generated),
            output_path=output_path,
        )
