from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ProjectConfig


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: Path
    relative_path: str
    namespace: str
    role: str
    canonical_group: str | None
    auto_recall: bool | str
    parse: bool


class SourceRegistry:
    """Expands the allowlist and applies source role gates before parsing."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._records = tuple(self._expand())

    def _expand(self) -> Iterable[SourceRecord]:
        roots = self.config.roots
        for rule in self.config.sources.get("sources", []):
            root_name = str(rule.get("root", "memory"))
            root = roots[root_name]
            if "path" in rule:
                paths = [(root / str(rule["path"])).resolve()]
            else:
                exclude_globs = tuple(str(item) for item in rule.get("exclude_globs", []))
                paths = sorted(
                    path.resolve()
                    for path in root.glob(str(rule["path_glob"]))
                    if path.is_file()
                    and not any(path.relative_to(root).match(pattern) for pattern in exclude_globs)
                )
            for path in paths:
                suffix = ""
                if len(paths) > 1:
                    suffix = ":" + path.relative_to(root).as_posix()
                yield SourceRecord(
                    source_id=str(rule["id"]) + suffix,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    namespace=str(rule["namespace"]),
                    role=str(rule["role"]),
                    canonical_group=rule.get("canonical_group"),
                    auto_recall=rule.get("auto_recall", False),
                    parse=bool(rule.get("parse", True)),
                )

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return self._records

    def select(self, mode: str = "ordinary") -> tuple[SourceRecord, ...]:
        if mode not in {"ordinary", "historical", "verification", "all_registered"}:
            raise ValueError(f"unknown source mode: {mode}")
        selected: list[SourceRecord] = []
        for record in self._records:
            if record.role in {"protected", "routing_index"}:
                continue
            if mode == "ordinary":
                if record.role == "canonical" and record.auto_recall is True:
                    selected.append(record)
            elif mode == "historical":
                if (
                    record.role == "historical_snapshot"
                    or (record.role == "canonical" and record.auto_recall is True)
                ):
                    selected.append(record)
            elif mode == "verification":
                if (
                    record.role in {"historical_snapshot", "evidence_fallback"}
                    or (record.role == "canonical" and record.auto_recall is True)
                ):
                    selected.append(record)
            else:
                selected.append(record)
        return tuple(selected)
