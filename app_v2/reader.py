from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import secrets
from typing import Callable, Iterable

from app.card_builder import estimate_tokens, truncate_text

from .candidates import Candidate, ChunkRef, ScopeCatalog


@dataclass(frozen=True)
class ReadPolicy:
    max_reads: int = 2
    metadata_tokens: int = 200
    total_content_tokens: int = 960
    content_tokens: int = 800
    future_read_reserve: int = 256


@dataclass
class _Capability:
    candidate: Candidate
    scope: str
    origin: str
    used: bool = False


def memory_id(candidate: Candidate) -> str:
    analysis_key = "|".join(
        (
            candidate.index_build_id,
            candidate.namespace,
            candidate.relative_path,
            " / ".join(candidate.headings),
            candidate.display_date,
        )
    )
    digest = hashlib.sha256(analysis_key.encode("utf-8")).hexdigest()[:24]
    return f"v2-node-{digest}"


def catalog_handle(candidate: Candidate) -> str:
    """Stable selector that is usable only after registration in a live turn."""

    analysis_key = "|".join(
        (
            candidate.index_build_id,
            candidate.namespace,
            candidate.relative_path,
            candidate.build_id,
        )
    )
    digest = hashlib.sha256(analysis_key.encode("utf-8")).hexdigest()[:24]
    return f"v2-catalog-{digest}"


class TurnReadSession:
    """In-memory, single-turn capability broker and canonical reader."""

    def __init__(
        self,
        *,
        memory_root: Path,
        scopes: ScopeCatalog,
        index_build_id: str,
        mode: str | None,
        policy: ReadPolicy | None = None,
        token_factory: Callable[[], str] | None = None,
        diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.memory_root = memory_root.resolve()
        self.scopes = scopes
        self.index_build_id = index_build_id
        self.mode = mode if mode in {"daily", "research"} else None
        self.policy = policy or ReadPolicy()
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._diagnostic_sink = diagnostic_sink
        self._capabilities: dict[str, _Capability] = {}
        # The random capability remains internal. The model sees only a short,
        # turn-local selector; the 192-bit turn handle is still required by the
        # service before this session can be reached. This keeps the directory
        # compact without weakening the cross-turn boundary.
        self._aliases: dict[str, str] = {}
        self._expired: set[str] = set()
        self._closed = False
        self._reads = 0
        self._used_content_tokens = 0
        self._used_metadata_tokens = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.policy.total_content_tokens - self._used_content_tokens)

    @property
    def remaining_metadata_tokens(self) -> int:
        return max(0, self.policy.metadata_tokens - self._used_metadata_tokens)

    def register_metadata_tokens(self, tokens: int) -> None:
        self._used_metadata_tokens = min(
            self.policy.metadata_tokens,
            self._used_metadata_tokens + max(0, int(tokens)),
        )

    def revoke_capabilities(self, handles: set[str]) -> None:
        for handle in handles:
            internal = self._aliases.pop(handle, handle)
            if internal in self._capabilities:
                self._expired.update({handle, internal})
                self._capabilities.pop(internal, None)

    def _new_alias(self, internal: str) -> str:
        """Return a compact opaque selector for one live turn.

        The selector is not an authorization secret by itself. Authorization
        is the pair (192-bit turn handle, selector); the full random capability
        never enters the prompt.
        """

        alias = f"m{hashlib.sha256(internal.encode('utf-8')).hexdigest()[:10]}"
        if alias in self._aliases or alias in self._expired:
            raise RuntimeError("capability_alias_collision")
        self._aliases[alias] = internal
        return alias

    def _deny_reason(self, scope: str) -> str | None:
        if self._closed:
            return "turn_closed"
        if self.mode == "research":
            return "research_mode"
        if scope != "ordinary" and self.mode is None:
            return "vault_mode_unavailable"
        return None

    def issue_directory(
        self,
        candidates: Iterable[Candidate],
        *,
        scope: str,
        origin: str,
        stable_handles: bool = False,
    ) -> dict[str, object]:
        authorization = self.authorize_search(scope=scope, origin=origin)
        if authorization["status"] != "ok":
            return {**authorization, "directory": []}
        allowed_namespaces = self.scopes.namespaces_for(scope)
        if stable_handles and (scope != "ordinary" or origin != "automatic"):
            return {
                "status": "denied",
                "error": "stable_catalog_requires_ordinary_automatic",
                "scope": scope,
                "directory": [],
            }

        directory: list[dict[str, object]] = []
        for candidate in candidates:
            if candidate.index_build_id != self.index_build_id:
                return {
                    "status": "error",
                    "error": "index_build_mismatch",
                    "scope": scope,
                    "directory": [],
                }
            if candidate.namespace not in allowed_namespaces:
                return {
                    "status": "error",
                    "error": "scope_namespace_mismatch",
                    "scope": scope,
                    "namespace": candidate.namespace,
                    "directory": [],
                }
            handle = catalog_handle(candidate) if stable_handles else self._token_factory()
            if stable_handles and handle in self._capabilities:
                return {
                    "status": "error",
                    "error": "catalog_handle_collision",
                    "scope": scope,
                    "directory": [],
                }
            while not stable_handles and (
                handle in self._capabilities or handle in self._expired
            ):
                handle = self._token_factory()
            self._capabilities[handle] = _Capability(candidate, scope, origin)
            exposed_handle = handle if stable_handles else self._new_alias(handle)
            directory.append(
                {
                    "capability_handle": exposed_handle,
                    "memory_id": memory_id(candidate),
                    "namespace": candidate.namespace,
                    "hierarchy": candidate.hierarchy,
                    "document_title": candidate.source_title,
                    "leaf_title": candidate.leaf_title,
                    "date": candidate.display_date,
                    "matched_cues": list(candidate.matched_cues),
                }
            )
        return {"status": "ok", "scope": scope, "directory": directory}

    def authorize_search(self, *, scope: str, origin: str) -> dict[str, object]:
        try:
            self.scopes.namespaces_for(scope)
        except ValueError:
            return {"status": "error", "error": "unknown_scope", "scope": scope}
        denied = self._deny_reason(scope)
        if denied:
            return {"status": "denied", "error": denied, "scope": scope}
        if scope != "ordinary" and origin not in {"explicit_search", "user_explicit_recall"}:
            return {
                "status": "denied",
                "error": "vault_requires_explicit_search",
                "scope": scope,
            }
        return {"status": "ok", "scope": scope}

    def _source_path(self, candidate: Candidate) -> Path | None:
        path = (self.memory_root / candidate.relative_path).resolve()
        return path if path.is_relative_to(self.memory_root) else None

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _stale_index_error(self, candidate: Candidate) -> dict[str, object]:
        diagnostic = {
            "event": "stale_index",
            "source": candidate.relative_path,
            "index_build_id": candidate.index_build_id,
        }
        if self._diagnostic_sink is not None:
            try:
                self._diagnostic_sink(diagnostic)
            except Exception:
                # Diagnostics must never weaken the read boundary.
                pass
        return {
            "status": "error",
            "error": "stale_index",
            "model_message": "该记忆刚被修改，但派生索引尚未更新；不要据此判断为记忆缺失或内容错误。",
            "recovery": "rebuild_memory_index",
            "diagnostic": diagnostic,
        }

    def _navigation_indexes(self, candidate: Candidate) -> list[Path]:
        direct_parent = Path(candidate.relative_path).parent
        found: list[Path] = []
        current = direct_parent
        while current != Path("."):
            path = (self.memory_root / current / "index.md").resolve()
            if path.is_relative_to(self.memory_root) and path.is_file():
                found.append(path)
            current = current.parent
        found.reverse()
        if len(found) <= 2:
            return found
        return [found[0], found[-1]]

    @staticmethod
    def _compact_index(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            kept.append(stripped)
        return "\n".join(kept)

    def _navigation_context(self, candidate: Candidate, budget: int) -> tuple[str, list[str]]:
        if budget <= 0:
            return "", []
        paths = self._navigation_indexes(candidate)
        parts: list[str] = []
        sources: list[str] = []
        for index, path in enumerate(paths):
            try:
                compact = self._compact_index(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError):
                continue
            current = "\n\n".join(parts)
            remaining = budget - estimate_tokens(current + ("\n\n" if parts else ""))
            if remaining <= 0:
                break
            remaining_indexes = max(1, len(paths) - index)
            share = max(1, remaining // remaining_indexes)
            part = truncate_text(compact, share)
            if not part:
                continue
            parts.append(part)
            sources.append(path.relative_to(self.memory_root).as_posix())
        context = "\n\n".join(parts)
        return truncate_text(context, budget), sources

    @staticmethod
    def _ranked_chunks(candidate: Candidate) -> list[ChunkRef]:
        return sorted(
            candidate.chunks,
            key=lambda item: (-item.vector_score, item.start_line, item.chunk_id),
        )

    def _content(
        self, candidate: Candidate, budget: int
    ) -> tuple[str, list[str], list[str], list[str], list[str], list[str], ChunkRef | None]:
        ranked = self._ranked_chunks(candidate)
        parts: list[str] = []
        included: list[str] = []
        partial: list[str] = []
        consumed: set[str] = set()
        empty: list[str] = []
        unavailable: list[str] = []
        first: ChunkRef | None = None

        for chunk in ranked:
            heading = chunk.heading_path.split(" / ")[-1]
            canonical_text = chunk.canonical_text
            if canonical_text is None:
                unavailable.append(chunk.chunk_id)
                continue
            if not canonical_text.strip():
                empty.append(chunk.chunk_id)
                continue
            part = f"{heading}\n{canonical_text}" if heading else canonical_text
            current = "\n\n".join(parts)
            separator = "\n\n" if parts else ""
            remaining = budget - estimate_tokens(current + separator)
            if remaining <= 0:
                break
            if estimate_tokens(part) <= remaining:
                parts.append(part)
                included.append(chunk.chunk_id)
                consumed.add(chunk.chunk_id)
                first = first or chunk
                continue
            if remaining >= 20:
                parts.append(truncate_text(part, remaining))
                included.append(chunk.chunk_id)
                partial.append(chunk.chunk_id)
                consumed.add(chunk.chunk_id)
                first = first or chunk
            break

        ignored = set(consumed).union(empty, unavailable)
        omitted = [chunk.chunk_id for chunk in ranked if chunk.chunk_id not in ignored]
        return "\n\n".join(parts), included, partial, omitted, empty, unavailable, first

    def read(self, capability_handle: str) -> dict[str, object]:
        if capability_handle in self._expired:
            return {"status": "error", "error": "expired_handle"}
        internal_handle = self._aliases.get(capability_handle, capability_handle)
        if internal_handle in self._expired:
            return {"status": "error", "error": "expired_handle"}
        capability = self._capabilities.get(internal_handle)
        if capability is None:
            return {"status": "error", "error": "invalid_handle"}
        if capability.used:
            return {"status": "error", "error": "used_handle"}
        denied = self._deny_reason(capability.scope)
        if denied:
            return {"status": "denied", "error": denied}
        if self._reads >= self.policy.max_reads:
            return {"status": "error", "error": "read_limit_exceeded"}
        if self.remaining_tokens <= 0:
            return {"status": "error", "error": "token_budget_exhausted"}

        candidate = capability.candidate
        if candidate.index_build_id != self.index_build_id:
            return {"status": "error", "error": "index_build_mismatch"}
        if candidate.namespace not in self.scopes.namespaces_for(capability.scope):
            return {"status": "error", "error": "scope_namespace_mismatch"}
        path = self._source_path(candidate)
        if path is None:
            return {"status": "error", "error": "source_forbidden"}
        if not path.is_file():
            return {"status": "error", "error": "source_unavailable"}
        try:
            if self._sha256(path) != candidate.source_hash:
                return self._stale_index_error(candidate)
        except OSError:
            return {"status": "error", "error": "source_unavailable"}
        try:
            path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            return {"status": "error", "error": "source_corrupt"}
        except OSError:
            return {"status": "error", "error": "source_unavailable"}

        unused_followups = sum(
            not item.used for token, item in self._capabilities.items()
            if token != internal_handle
        )
        remaining_read_slots = min(
            max(0, self.policy.max_reads - self._reads - 1),
            unused_followups,
        )
        reserved = min(
            self.remaining_tokens,
            self.policy.future_read_reserve * remaining_read_slots,
        )
        available_this_read = max(0, self.remaining_tokens - reserved)
        navigation_budget = self.remaining_metadata_tokens
        parent_context, parent_sources = self._navigation_context(candidate, navigation_budget)
        parent_tokens = estimate_tokens(parent_context)
        self._used_metadata_tokens += parent_tokens
        content_budget = min(
            self.policy.content_tokens,
            available_this_read,
        )
        if content_budget <= 0:
            return {"status": "error", "error": "token_budget_exhausted"}
        content, included, partial, omitted, empty, unavailable, first = self._content(
            candidate, content_budget
        )
        if unavailable:
            return {
                "status": "error",
                "error": "segment_unavailable",
                "unavailable_chunk_ids": unavailable,
                "model_message": "该记忆的可读段索引不完整；不要把它解释为记忆内容缺失。",
                "recovery": "rebuild_canonical_segment_sidecar",
            }
        if not content or first is None:
            return {
                "status": "error",
                "error": "empty_source_content",
                "empty_chunk_ids": empty,
            }

        content_tokens = estimate_tokens(content)
        spent = content_tokens
        capability.used = True
        self._reads += 1
        self._used_content_tokens += spent
        source = f"{candidate.relative_path}:{first.start_line}"
        return {
            "status": "ok",
            "memory_id": memory_id(candidate),
            "index_build_id": candidate.index_build_id,
            "namespace": candidate.namespace,
            "scope": capability.scope,
            "parent_context": parent_context,
            "parent_source": parent_sources[-1] if parent_sources else None,
            "navigation_context": parent_context,
            "navigation_sources": parent_sources,
            "leaf_title": candidate.leaf_title,
            "content": content,
            "source": source,
            "included_chunk_ids": included,
            "partial_chunk_ids": partial,
            "omitted_chunk_ids": omitted,
            "empty_chunk_ids": empty,
            "truncated": bool(partial or omitted),
            "tokens": {
                "parent": parent_tokens,
                "metadata": parent_tokens,
                "metadata_spent_this_turn": self._used_metadata_tokens,
                "metadata_remaining_this_turn": self.remaining_metadata_tokens,
                "content": content_tokens,
                "spent_this_read": spent,
                "spent_this_turn": self._used_content_tokens,
                "remaining_this_turn": self.remaining_tokens,
                "reserved_for_future_reads": reserved,
                "available_this_read": available_this_read,
            },
        }

    def end_turn(self) -> None:
        self._expired.update(self._capabilities)
        self._expired.update(self._aliases)
        self._capabilities.clear()
        self._aliases.clear()
        self._closed = True
