from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


ALLOWED_ROLES = {
    "canonical",
    "historical_snapshot",
    "routing_index",
    "protected",
    "evidence_fallback",
}


class ConfigError(ValueError):
    """Raised when configuration is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    sources_path: Path
    entities_path: Path
    policies_path: Path
    recall_aliases_path: Path
    sources: dict[str, Any]
    entities: dict[str, Any]
    policies: dict[str, Any]
    recall_aliases: dict[str, Any]

    @property
    def roots(self) -> dict[str, Path]:
        raw = self.sources.get("roots", {})
        resolved: dict[str, Path] = {}
        for name, value in raw.items():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            resolved[name] = path.resolve()
        return resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping")
    return value


def _load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "events": []}
    return _load_yaml(path)


def load_project_config(project_root: Path | None = None) -> ProjectConfig:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    config_dir = root / "config"
    sources_path = config_dir / "sources.yaml"
    entities_path = config_dir / "entities.yaml"
    policies_path = config_dir / "policies.yaml"
    recall_aliases_path = config_dir / "recall-aliases.yaml"
    return ProjectConfig(
        project_root=root,
        sources_path=sources_path,
        entities_path=entities_path,
        policies_path=policies_path,
        recall_aliases_path=recall_aliases_path,
        sources=_load_yaml(sources_path),
        entities=_load_yaml(entities_path),
        policies=_load_yaml(policies_path),
        recall_aliases=_load_optional_yaml(recall_aliases_path),
    )


def _matches_known_namespace(namespace: str, known: set[str]) -> bool:
    if namespace in known:
        return True
    return any(item.endswith(".*") and namespace.startswith(item[:-1]) for item in known)


def validate_config(config: ProjectConfig) -> list[str]:
    errors: list[str] = []
    source_rules = config.sources.get("sources")
    if not isinstance(source_rules, list) or not source_rules:
        errors.append("sources.yaml: sources must be a non-empty list")
        source_rules = []

    known_namespaces = set(config.policies.get("namespaces", {}).keys())
    if not known_namespaces:
        errors.append("policies.yaml: namespaces must define at least one route")

    roots = config.roots
    if not roots:
        errors.append("sources.yaml: roots must define at least one root")
    for name, root in roots.items():
        if not root.is_dir():
            errors.append(f"sources.yaml: root {name!r} does not exist: {root}")

    seen_ids: set[str] = set()
    seen_files: dict[Path, str] = {}
    canonical_groups: dict[str, list[str]] = {}
    covered_files: set[Path] = set()

    for index, rule in enumerate(source_rules):
        label = f"sources[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{label}: rule must be a mapping")
            continue
        source_id = str(rule.get("id", "")).strip()
        if not source_id:
            errors.append(f"{label}: missing id")
            continue
        if source_id in seen_ids:
            errors.append(f"{label}: duplicate source id {source_id!r}")
        seen_ids.add(source_id)

        root_name = str(rule.get("root", "memory"))
        root = roots.get(root_name)
        if root is None:
            errors.append(f"{source_id}: unknown root {root_name!r}")
            continue

        has_path = "path" in rule
        has_glob = "path_glob" in rule
        if has_path == has_glob:
            errors.append(f"{source_id}: define exactly one of path or path_glob")
            continue

        namespace = str(rule.get("namespace", "")).strip()
        if not namespace or not _matches_known_namespace(namespace, known_namespaces):
            errors.append(f"{source_id}: unknown namespace {namespace!r}")

        role = str(rule.get("role", "")).strip()
        if role not in ALLOWED_ROLES:
            errors.append(f"{source_id}: unknown role {role!r}")

        canonical_group = rule.get("canonical_group")
        if role == "canonical" and canonical_group:
            canonical_groups.setdefault(str(canonical_group), []).append(source_id)

        matches: list[Path]
        if has_path:
            candidate = (root / str(rule["path"])).resolve()
            matches = [candidate] if candidate.is_file() else []
            if not matches:
                errors.append(f"{source_id}: source file does not exist: {candidate}")
        else:
            exclude_globs = rule.get("exclude_globs", [])
            if not isinstance(exclude_globs, list) or any(
                not isinstance(item, str) or not item.strip() for item in exclude_globs
            ):
                errors.append(f"{source_id}: exclude_globs must be a list of non-empty strings")
                exclude_globs = []
            matches = sorted(
                path.resolve()
                for path in root.glob(str(rule["path_glob"]))
                if path.is_file()
                and not any(
                    path.relative_to(root).match(str(pattern)) for pattern in exclude_globs
                )
            )
            if not matches:
                errors.append(f"{source_id}: path_glob matched no files: {rule['path_glob']}")

        for path in matches:
            if path in seen_files:
                errors.append(
                    f"{source_id}: file also matched by {seen_files[path]!r}: {path}"
                )
            else:
                seen_files[path] = source_id
            covered_files.add(path)

    alias_rules = config.recall_aliases.get("events", [])
    if not isinstance(alias_rules, list):
        errors.append("recall-aliases.yaml: events must be a list")
        alias_rules = []
    for index, rule in enumerate(alias_rules):
        label = f"recall-aliases.events[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{label}: rule must be a mapping")
            continue
        source_id = str(rule.get("source_id", "")).strip()
        heading_contains = str(rule.get("heading_contains", "")).strip()
        aliases = rule.get("aliases", [])
        if source_id not in seen_ids:
            errors.append(f"{label}: unknown source_id {source_id!r}")
        if not heading_contains:
            errors.append(f"{label}: heading_contains is required")
        if not isinstance(aliases, list) or not aliases:
            errors.append(f"{label}: aliases must be a non-empty list")
        elif any(not str(alias).strip() for alias in aliases):
            errors.append(f"{label}: aliases cannot contain empty values")

    for group, ids in canonical_groups.items():
        if len(ids) > 1:
            errors.append(f"canonical group {group!r} has multiple current sources: {ids}")

    entity_defs = config.entities.get("entities")
    if not isinstance(entity_defs, dict):
        errors.append("entities.yaml: entities must be a mapping")
        entity_defs = {}
    strong_aliases: dict[str, str] = {}
    for entity_id, entity in entity_defs.items():
        if not isinstance(entity, dict):
            errors.append(f"entity {entity_id!r}: definition must be a mapping")
            continue
        if "namespaces" in entity or "namespace" in entity:
            errors.append(
                f"entity {entity_id!r}: namespaces belong to memory sources, not entities"
            )
        aliases = entity.get("strong_aliases", [])
        if not isinstance(aliases, list):
            errors.append(f"entity {entity_id!r}: strong_aliases must be a list")
            continue
        for alias in aliases:
            normalized = str(alias).strip().casefold()
            if not normalized:
                errors.append(f"entity {entity_id!r}: empty strong alias")
            elif normalized in strong_aliases and strong_aliases[normalized] != entity_id:
                errors.append(
                    f"strong alias {alias!r} belongs to both {strong_aliases[normalized]!r} and {entity_id!r}"
                )
            else:
                strong_aliases[normalized] = str(entity_id)

    validation = config.policies.get("validation", {})
    models = config.policies.get("models", {})
    if validation.get("require_runtime_models") and models.get("backend", "bge") == "bge":
        for field in ("embedding_path", "reranker_path"):
            value = models.get(field) if isinstance(models, dict) else None
            path = Path(str(value)).expanduser() if value else Path()
            if value and not path.is_absolute():
                path = config.project_root / path
            if not value or not path.is_dir():
                errors.append(f"policies.yaml: model path {field!r} does not exist: {value}")

    thresholds = config.policies.get("thresholds", {})
    required_thresholds = {
        "ordinary_event",
        "continuity_query",
        "past_event_anchor",
        "past_fact_question",
        "stable_personal_fact",
        "explicit_recall",
        "health_private",
        "triggered_protocol",
    }
    if validation.get("require_frozen_thresholds"):
        for key in sorted(required_thresholds):
            value = thresholds.get(key) if isinstance(thresholds, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"policies.yaml: threshold {key!r} must be finite after T08")

    cooldown = config.policies.get("cooldown", {})
    if validation.get("require_cooldown"):
        for key in ("identity_min_user_turns", "fallback_min_user_turns", "fallback_reconsider_turns"):
            value = cooldown.get(key) if isinstance(cooldown, dict) else None
            if not isinstance(value, int) or value <= 0:
                errors.append(f"policies.yaml: cooldown {key!r} must be a positive integer")

    coverage = config.sources.get("coverage", {})
    excluded = coverage.get("excluded_sources", []) if isinstance(coverage, dict) else []
    excluded_files: set[Path] = set()
    root = roots.get("memory")
    if root and root.is_dir():
        for item in excluded:
            if not isinstance(item, dict) or not item.get("path") or not item.get("reason"):
                errors.append("coverage.excluded_sources entries require path and reason")
                continue
            path = (root / str(item["path"])).resolve()
            if not path.is_file():
                errors.append(f"declared excluded source does not exist: {path}")
            excluded_files.add(path)
        all_markdown = {path.resolve() for path in root.rglob("*.md")}
        undeclared = sorted(all_markdown - covered_files - excluded_files)
        for path in undeclared:
            errors.append(f"memory source is neither registered nor explicitly excluded: {path}")

    return errors
