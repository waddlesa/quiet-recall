from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Callable, Protocol

from app.cli import model_backends
from app.config import ProjectConfig, load_project_config
from .advisory import render_turn_plan_packet
from .candidates import CleanCandidateEngine, ScopeCatalog
from .reader import ReadPolicy, TurnReadSession
from .source_segments import CanonicalSegmentStore
from .turn_policy import is_explicit_recall_request, route_explicit_scope


class CandidateEngine(Protocol):
    def search(self, prompt: str, *, scope: str = "ordinary", **kwargs: object) -> dict[str, object]: ...

    def search_hybrid(
        self, prompt: str, *, scope: str = "ordinary", **kwargs: object
    ) -> dict[str, object]: ...

    def catalog(self, prompt: str, *, scope: str = "ordinary") -> list[object]: ...

    def build_fingerprint(self) -> str: ...


@dataclass
class _ServiceSession:
    reader: TurnReadSession
    mode: str | None
    expires_at: float
    created_at: float


class MemoryToolService:
    """Resident Phase-3 core shared by CLI/HTTP/MCP adapters.

    The service owns model instances and turn-scoped capability registries.  An
    adapter may forward arguments and results, but must not duplicate policy.
    """

    def __init__(
        self,
        *,
        engine: CandidateEngine,
        scopes: ScopeCatalog,
        memory_root: Path,
        ttl_seconds: int = 600,
        max_sessions: int = 64,
        clock: Callable[[], float] | None = None,
        diagnostic_path: Path | None = None,
        turn_token_factory: Callable[[], str] | None = None,
        retrieval_mode: str = "dense_bge",
    ) -> None:
        self.engine = engine
        self.scopes = scopes
        self.memory_root = memory_root.resolve()
        self.index_build_id = engine.build_fingerprint()
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock or time.monotonic
        self._turn_token_factory = turn_token_factory or (lambda: secrets.token_urlsafe(24))
        if retrieval_mode not in {"dense_bge", "hybrid_rrf"}:
            raise ValueError("invalid_retrieval_mode")
        self.retrieval_mode = retrieval_mode
        self._diagnostic_path = diagnostic_path
        self._sessions: dict[str, _ServiceSession] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_project(
        cls, project_root: Path, *, retrieval_mode: str = "dense_bge"
    ) -> "MemoryToolService":
        root = project_root.resolve()
        config: ProjectConfig = load_project_config(root)
        scopes = ScopeCatalog.load(root / "config" / "v2" / "scopes.yaml")
        database_path = root / "state" / "memory-index.sqlite3"
        segment_store = CanonicalSegmentStore.ensure_current(
            config, index_path=database_path
        )
        embedder, reranker = model_backends(config)
        engine = CleanCandidateEngine(
            config,
            scopes,
            embedder,
            reranker,
            database_path=database_path,
            segment_store=segment_store,
        )
        return cls(
            engine=engine,
            scopes=scopes,
            memory_root=config.roots["memory"],
            diagnostic_path=root / "state" / "v2" / "service-diagnostics.jsonl",
            retrieval_mode=retrieval_mode,
        )

    @staticmethod
    def _safe_id(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _emit(self, event: str, *, turn_handle: str | None = None, **values: object) -> None:
        if self._diagnostic_path is None:
            return
        payload = {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **({"turn": self._safe_id(turn_handle)} if turn_handle else {}),
            **values,
        }
        self._diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        with self._diagnostic_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _prune(self) -> None:
        now = self._clock()
        expired = [key for key, item in self._sessions.items() if item.expires_at <= now]
        for key in expired:
            self._sessions.pop(key).reader.end_turn()
            self._emit("turn_expired", turn_handle=key)
        if len(self._sessions) <= self.max_sessions:
            return
        oldest = sorted(self._sessions, key=lambda key: self._sessions[key].created_at)
        for key in oldest[: len(self._sessions) - self.max_sessions]:
            self._sessions.pop(key).reader.end_turn()
            self._emit("turn_evicted", turn_handle=key)

    def _new_turn(self, mode: str | None) -> tuple[str, _ServiceSession]:
        token = self._turn_token_factory()
        while token in self._sessions:
            token = self._turn_token_factory()
        now = self._clock()

        def diagnostic_sink(payload: dict[str, object]) -> None:
            self._emit(
                "reader_diagnostic",
                turn_handle=token,
                code=payload.get("event"),
                source=payload.get("source"),
                index_build_id=payload.get("index_build_id"),
            )

        record = _ServiceSession(
            reader=TurnReadSession(
                memory_root=self.memory_root,
                scopes=self.scopes,
                index_build_id=self.index_build_id,
                mode=mode,
                policy=ReadPolicy(
                    metadata_tokens=self.scopes.metadata_policy.total_tokens,
                    total_content_tokens=4096,
                    content_tokens=3072,
                    future_read_reserve=512,
                ),
                diagnostic_sink=diagnostic_sink,
            ),
            mode=mode,
            expires_at=now + self.ttl_seconds,
            created_at=now,
        )
        self._sessions[token] = record
        self._emit("turn_started", turn_handle=token, mode=mode or "missing")
        return token, record

    def _turn(self, turn_handle: str) -> _ServiceSession | None:
        self._prune()
        record = self._sessions.get(turn_handle)
        if record is not None:
            record.expires_at = self._clock() + self.ttl_seconds
        return record

    @staticmethod
    def _normalize_mode(mode: str | None) -> str | None:
        if mode in {None, "missing"}:
            return None
        if mode in {"daily", "research"}:
            return mode
        raise ValueError("invalid_mode")

    def _retrieve(
        self,
        query: str,
        *,
        scope: str,
        apply_evidence_floor: bool,
        working_context: str = "",
    ) -> tuple[list[object], dict[str, object]]:
        if self.retrieval_mode == "hybrid_rrf":
            result = self.engine.search_hybrid(
                query,
                scope=scope,
                apply_evidence_floor=apply_evidence_floor,
                working_context=working_context,
            )
        else:
            result = self.engine.search(
                query,
                scope=scope,
                apply_evidence_floor=apply_evidence_floor,
                working_context=working_context,
            )
        competition = result.get("context_competitor")
        return list(result["directory"]), (
            dict(competition) if isinstance(competition, dict) else {}
        )

    def _packetize(
        self,
        *,
        token: str,
        record: _ServiceSession,
        directories: list[tuple[str, list[dict[str, object]]]],
        explicit_recall: bool,
    ) -> tuple[str, int, list[dict[str, object]]]:
        remaining_budget = record.reader.remaining_metadata_tokens
        issued_handles = {
            str(item.get("capability_handle", ""))
            for _, directory in directories
            for item in directory
        }
        if not issued_handles and not explicit_recall:
            return "", 0, []
        if remaining_budget <= 0:
            record.reader.revoke_capabilities(issued_handles)
            return "", 0, []
        active_vaults = [
            scope for scope, directory in directories if scope != "ordinary" and directory
        ]
        descriptions = None
        if explicit_recall:
            # A uniquely routed vault is already present in the packet.  When
            # an explicit recollection request is oblique and no vault could be
            # routed safely, expose only the small scope catalog so the model
            # can choose an explicit search without guessing hidden names.
            described_vaults = active_vaults or sorted(self.scopes.vaults)
            descriptions = {
                scope: self.scopes.descriptions.get(scope, "")
                for scope in described_vaults
            }
        packet = render_turn_plan_packet(
            turn_handle=token,
            directories=directories,
            descriptions=descriptions,
            explicit_recall=explicit_recall,
            token_budget=min(
                remaining_budget, self.scopes.metadata_policy.directory_tokens
            ),
        )
        visible = set(packet.visible_handles)
        record.reader.revoke_capabilities(issued_handles - visible)
        record.reader.register_metadata_tokens(packet.estimated_tokens)
        flattened = [
            item
            for _, directory in directories
            for item in directory
            if str(item.get("capability_handle", "")) in visible
        ]
        return packet.text, packet.estimated_tokens, flattened

    def plan_turn(
        self,
        *,
        query: str,
        mode: str | None = "daily",
        working_context: str = "",
    ) -> dict[str, object]:
        """Create one coherent memory plan for a user turn.

        Ordinary candidates use the low evidence floor. A vault is considered
        only when the user explicitly refers to previously shared facts and one
        scope has uniquely matching catalog cues.
        """

        query = str(query).strip()
        if not query:
            return {"status": "error", "error": "empty_query"}
        if len(query) > 8000:
            return {"status": "error", "error": "query_too_long"}
        bounded_context = str(working_context).strip()[-100:]
        try:
            normalized_mode = self._normalize_mode(mode)
        except ValueError:
            return {"status": "error", "error": "invalid_mode"}

        with self._lock:
            self._prune()
            token, record = self._new_turn(normalized_mode)
            authorization = record.reader.authorize_search(
                scope="ordinary", origin="automatic"
            )
            if authorization["status"] != "ok":
                return {
                    **authorization,
                    "turn_handle": token,
                    "directory": [],
                    "advisory_text": "",
                    "metadata_tokens": 0,
                }

            started = time.perf_counter()
            explicit = is_explicit_recall_request(query)
            ordinary_candidates, context_competitor = self._retrieve(
                query,
                scope="ordinary",
                apply_evidence_floor=True,
                working_context="" if explicit else bounded_context,
            )
            ordinary_issued = record.reader.issue_directory(
                ordinary_candidates, scope="ordinary", origin="automatic"
            )
            directories: list[tuple[str, list[dict[str, object]]]] = [
                ("ordinary", list(ordinary_issued.get("directory", [])))
            ]

            routed_scope = None
            if explicit:
                routed_scope = route_explicit_scope(
                    query,
                    self.scopes.routing_cues or {},
                    self.scopes.routing_subjects or {},
                )
            if routed_scope:
                vault_candidates, _ = self._retrieve(
                    query,
                    scope=routed_scope,
                    apply_evidence_floor=False,
                )
                vault_issued = record.reader.issue_directory(
                    vault_candidates,
                    scope=routed_scope,
                    origin="user_explicit_recall",
                )
                if vault_issued.get("status") == "ok":
                    directories.insert(
                        0, (routed_scope, list(vault_issued.get("directory", [])))
                    )

            advisory, metadata_tokens, directory = self._packetize(
                token=token,
                record=record,
                directories=directories,
                explicit_recall=explicit,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._emit(
                "turn_planned",
                turn_handle=token,
                explicit_recall=explicit,
                routed_scope=routed_scope,
                candidate_count=len(directory),
                context_competitor_used=bool(context_competitor.get("used")),
                context_competitor_won=bool(context_competitor.get("won")),
                metadata_tokens=metadata_tokens,
                latency_ms=round(elapsed_ms, 3),
            )
            return {
                "status": "ok",
                "turn_handle": token,
                "explicit_recall": explicit,
                "routed_scope": routed_scope,
                "directory": directory,
                "advisory_text": advisory,
                "metadata_tokens": metadata_tokens,
                "latency_ms": round(elapsed_ms, 3),
                "context_competitor": context_competitor,
            }

    def search(
        self,
        *,
        query: str,
        scope: str = "ordinary",
        mode: str | None = "daily",
        origin: str = "explicit_search",
        turn_handle: str | None = None,
        directory_source: str = "ranked",
    ) -> dict[str, object]:
        query = str(query).strip()
        if not query:
            return {"status": "error", "error": "empty_query"}
        if len(query) > 8000:
            return {"status": "error", "error": "query_too_long"}
        if origin not in {"automatic", "explicit_search", "user_explicit_recall"}:
            return {"status": "error", "error": "invalid_origin"}
        if directory_source not in {"ranked", "static_full"}:
            return {"status": "error", "error": "invalid_directory_source"}
        if directory_source == "static_full" and (
            scope != "ordinary" or origin != "automatic"
        ):
            return {
                "status": "denied",
                "error": "static_catalog_requires_ordinary_automatic",
                "scope": scope,
                "directory": [],
            }
        with self._lock:
            self._prune()
            if turn_handle:
                record = self._turn(turn_handle)
                if record is None:
                    return {"status": "error", "error": "invalid_turn_handle"}
                if mode == "inherit":
                    normalized_mode = record.mode
                else:
                    try:
                        normalized_mode = self._normalize_mode(mode)
                    except ValueError:
                        return {"status": "error", "error": "invalid_mode"}
                    if record.mode != normalized_mode:
                        return {"status": "error", "error": "mode_mismatch"}
                token = turn_handle
            else:
                if mode == "inherit":
                    return {"status": "error", "error": "mode_required"}
                try:
                    normalized_mode = self._normalize_mode(mode)
                except ValueError:
                    return {"status": "error", "error": "invalid_mode"}
                token, record = self._new_turn(normalized_mode)

            authorization = record.reader.authorize_search(scope=scope, origin=origin)
            if authorization["status"] != "ok":
                self._emit(
                    "search_denied",
                    turn_handle=token,
                    scope=scope,
                    reason=authorization.get("error"),
                )
                return {**authorization, "turn_handle": token, "directory": []}

            started = time.perf_counter()
            if directory_source == "static_full":
                candidates = self.engine.catalog(query, scope=scope)
            else:
                candidates, _ = self._retrieve(
                    query,
                    scope=scope,
                    apply_evidence_floor=(scope == "ordinary" and origin == "automatic"),
                )
            issued = record.reader.issue_directory(
                candidates,
                scope=scope,
                origin=origin,
                stable_handles=directory_source == "static_full",
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            directory = list(issued.get("directory", []))
            advisory_text = ""
            metadata_tokens = 0
            if directory_source == "ranked" and issued.get("status") == "ok":
                advisory_text, metadata_tokens, directory = self._packetize(
                    token=token,
                    record=record,
                    directories=[(scope, directory)],
                    explicit_recall=origin == "user_explicit_recall",
                )
            self._emit(
                "search_completed",
                turn_handle=token,
                scope=scope,
                origin=origin,
                directory_source=directory_source,
                candidate_count=len(directory),
                latency_ms=round(elapsed_ms, 3),
            )
            return {
                **issued,
                "turn_handle": token,
                "scope_description": self.scopes.descriptions.get(scope, ""),
                "directory_source": directory_source,
                # `_packetize` revokes every capability that did not fit in the
                # visible metadata budget.  Never leak the pre-packetization
                # directory from `issued`: those selectors are deliberately
                # expired and would make an immediate search -> read fail.
                "directory": directory,
                "advisory_text": advisory_text,
                "metadata_tokens": metadata_tokens,
                "latency_ms": round(elapsed_ms, 3),
            }

    def read(self, *, turn_handle: str, capability_handle: str) -> dict[str, object]:
        with self._lock:
            record = self._turn(turn_handle)
            if record is None:
                return {"status": "error", "error": "invalid_turn_handle"}
            result = record.reader.read(capability_handle)
            self._emit(
                "read_completed" if result.get("status") == "ok" else "read_rejected",
                turn_handle=turn_handle,
                status=result.get("status"),
                error=result.get("error"),
                namespace=result.get("namespace"),
                spent=(result.get("tokens") or {}).get("spent_this_read")
                if isinstance(result.get("tokens"), dict)
                else None,
            )
            return result

    def end_turn(self, *, turn_handle: str) -> dict[str, object]:
        with self._lock:
            record = self._sessions.pop(turn_handle, None)
            if record is None:
                return {"status": "error", "error": "invalid_turn_handle"}
            record.reader.end_turn()
            self._emit("turn_ended", turn_handle=turn_handle)
            return {"status": "ok"}

    def health(self) -> dict[str, object]:
        with self._lock:
            self._prune()
            return {
                "status": "ok",
                "service": "quiet-recall",
                "index_build_id": self.index_build_id,
                "resident_models": True,
                "retrieval_mode": self.retrieval_mode,
                "metadata_token_budget": self.scopes.metadata_policy.total_tokens,
                "directory_token_budget": self.scopes.metadata_policy.directory_tokens,
                "active_turns": len(self._sessions),
                "scopes": {
                    "ordinary": self.scopes.descriptions.get("ordinary", ""),
                    **{
                        name: self.scopes.descriptions.get(name, "")
                        for name in sorted(self.scopes.vaults)
                    },
                },
            }
