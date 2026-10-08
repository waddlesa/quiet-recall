from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable

import numpy as np
import yaml

from app.config import ProjectConfig
from app.embedding_backend import EmbeddingBackend, RerankBackend
from app_v2.lexical import bm25_scores, matched_cues, reciprocal_rank
from app_v2.turn_policy import EvidenceFloor, MetadataPolicy, normalize_text


@dataclass(frozen=True)
class ScopeCatalog:
    ordinary: frozenset[str]
    vaults: dict[str, frozenset[str]]
    descriptions: dict[str, str]
    dense_pool_k: int
    directory_k: int
    parent_cap: int
    evidence_floor: EvidenceFloor = EvidenceFloor()
    metadata_policy: MetadataPolicy = MetadataPolicy()
    routing_cues: dict[str, tuple[str, ...]] | None = None
    routing_subjects: dict[str, str] | None = None

    @classmethod
    def load(cls, path: Path) -> "ScopeCatalog":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ordinary = raw.get("ordinary", {})
        vaults_raw = raw.get("vaults", {})
        descriptions = {
            "ordinary": str(ordinary.get("description", "")),
            **{
                str(name): str(item.get("description", ""))
                for name, item in vaults_raw.items()
            },
        }
        policy = raw.get("candidate_policy", {})
        floor = policy.get("evidence_floor", {})
        strict = policy.get("strict_surface", {})
        metadata = raw.get("metadata_policy", {})
        return cls(
            ordinary=frozenset(str(item) for item in ordinary.get("namespaces", [])),
            vaults={
                str(name): frozenset(str(value) for value in item.get("namespaces", []))
                for name, item in vaults_raw.items()
            },
            descriptions=descriptions,
            dense_pool_k=int(policy.get("dense_pool_k", 32)),
            directory_k=int(policy.get("directory_k", 8)),
            parent_cap=int(policy.get("parent_cap", 3)),
            evidence_floor=EvidenceFloor(
                lexical_min=float(floor.get("lexical_min", 8.0)),
                rerank_min=float(floor.get("rerank_min", 0.5)),
                vector_min=float(floor.get("vector_min", 0.60)),
                strict_lexical_min=float(strict.get("lexical_min", 8.0)),
                strict_rerank_min=float(strict.get("rerank_min", 0.5)),
                strict_cue_min=int(strict.get("matched_cue_min", 2)),
            ),
            metadata_policy=MetadataPolicy(
                total_tokens=int(metadata.get("total_tokens", 200)),
                directory_tokens=int(metadata.get("directory_tokens", 140)),
            ),
            routing_cues={
                str(name): tuple(str(cue) for cue in item.get("routing_cues", []))
                for name, item in vaults_raw.items()
            },
            routing_subjects={
                str(name): str(item.get("routing_subject", "named"))
                for name, item in vaults_raw.items()
            },
        )

    def namespaces_for(self, scope: str) -> frozenset[str]:
        if scope == "ordinary":
            return self.ordinary
        if scope not in self.vaults:
            raise ValueError(f"unknown v2 scope: {scope}")
        return self.vaults[scope]


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    heading_path: str
    start_line: int
    end_line: int
    excerpt: str
    vector_score: float
    canonical_text: str | None = None


@dataclass(frozen=True)
class Candidate:
    index_build_id: str
    build_id: str
    source_id: str
    event_id: str | None
    relative_path: str
    namespace: str
    event_date: str
    headings: tuple[str, ...]
    excerpts: tuple[str, ...]
    chunks: tuple[ChunkRef, ...]
    source_hash: str
    navigation_key: str
    navigation_title: str
    vector_score: float
    retrieval_mode: str = "full_text"
    retrieval_surface: str = ""
    retrieval_cues: tuple[str, ...] = ()
    lexical_score: float = 0.0
    rerank_score: float = float("-inf")
    fusion_score: float = 0.0
    matched_cues: tuple[str, ...] = ()

    @property
    def parent_key(self) -> str:
        return self.navigation_key

    @property
    def leaf_title(self) -> str:
        heading = next((item for item in self.headings if item and item != "(untitled)"), "")
        return heading.split(" / ")[-1] if heading else Path(self.relative_path).stem

    @property
    def source_title(self) -> str:
        heading = next((item for item in self.headings if item and item != "(untitled)"), "")
        return heading.split(" / ")[0] if heading else Path(self.relative_path).stem

    @property
    def display_date(self) -> str:
        if self.retrieval_mode == "strict_surface":
            return ""
        dates = re.findall(r"20\d{2}[./年-]\d{1,2}[./月-]\d{1,2}日?", " ".join(self.headings))
        if dates:
            parts = re.findall(r"\d+", dates[-1])
            if len(parts) >= 3:
                return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        return self.event_date

    @property
    def hierarchy(self) -> str:
        values = [
            self.navigation_title,
            self.source_title,
            self.leaf_title,
            self.display_date,
        ]
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            normalized = re.sub(r"[\s./年月日_-]+", "", value).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(value)
        return " > ".join(result)

    @property
    def document(self) -> str:
        if self.retrieval_mode == "strict_surface":
            return f"严格召回线索：{self.retrieval_surface}"
        return f"记忆类别：{self.namespace}。层级：{self.hierarchy}。内容：{' '.join(self.excerpts)}"

    @property
    def content_text(self) -> str:
        return "\n".join(self.excerpts)


def _candidate_admitted(candidate: Candidate, query: str, floor: EvidenceFloor) -> bool:
    if candidate.retrieval_mode != "strict_surface":
        return floor.admits(
            lexical=candidate.lexical_score,
            rerank=candidate.rerank_score,
            vector=candidate.vector_score,
        )
    normalized_query = normalize_text(query)
    exact_cue = any(
        normalize_text(cue) and normalize_text(cue) in normalized_query
        for cue in candidate.retrieval_cues
    )
    return floor.admits_strict(
        lexical=candidate.lexical_score,
        rerank=candidate.rerank_score,
        matched_cue_count=len(candidate.matched_cues),
        exact_cue=exact_cue,
    )


def apply_directory_policy(
    ranked: Iterable[Candidate], *, directory_k: int, parent_cap: int
) -> tuple[list[Candidate], list[Candidate]]:
    """Preserve rank while selecting unique leaves with a navigation-tree cap."""

    selected: list[Candidate] = []
    cap_rejected: list[Candidate] = []
    seen_leaves: set[str] = set()
    parent_counts: dict[str, int] = {}
    for candidate in ranked:
        if candidate.relative_path in seen_leaves:
            continue
        seen_leaves.add(candidate.relative_path)
        parent = candidate.parent_key
        if parent_counts.get(parent, 0) >= parent_cap:
            cap_rejected.append(candidate)
            continue
        selected.append(candidate)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        if len(selected) >= directory_k:
            break
    return selected, cap_rejected


def apply_context_cutoff(
    candidates: Iterable[Candidate],
    *,
    context_score: float | None,
    minimum_distinct_leaves: int = 2,
) -> tuple[list[Candidate], list[Candidate]]:
    """Use working context as a cutoff while preserving a small recall floor.

    This turns working context into a true position in the semantic ranking:
    if it would rank third, the two stronger admitted memories remain.  When
    context is not first, at least two distinct leaves survive so a near-tie
    cannot erase the correct second card.  Candidate order is preserved so
    the existing fusion and tree-cap policy still decide display order.
    """

    items = list(candidates)
    if context_score is None:
        return items, []
    kept = [item for item in items if item.rerank_score > context_score]
    if not kept:
        return [], items

    kept_leaves = {item.relative_path for item in kept}
    for item in items:
        if len(kept_leaves) >= minimum_distinct_leaves:
            break
        if item.relative_path in kept_leaves:
            continue
        kept.append(item)
        kept_leaves.add(item.relative_path)
    kept_ids = {id(item) for item in kept}
    suppressed = [item for item in items if id(item) not in kept_ids]
    return kept, suppressed


def context_rank(
    candidates: Iterable[Candidate], *, context_score: float | None
) -> int | None:
    """Return the one-based reranker position of working context."""

    if context_score is None:
        return None
    return 1 + sum(item.rerank_score > context_score for item in candidates)


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^#\s+(.+?)\s*$", line)
            if match:
                return match.group(1).strip()
    except OSError:
        pass
    return ""


def navigation_metadata(root: Path, relative_path: str) -> tuple[str, str]:
    """Use the nearest ancestor with index.md; otherwise use the direct parent."""

    direct_parent = Path(relative_path).parent
    current = direct_parent
    while current != Path("."):
        index_path = root / current / "index.md"
        if index_path.is_file():
            return current.as_posix(), _first_heading(index_path) or current.name
        current = current.parent
    fallback_key = direct_parent.as_posix() if direct_parent != Path(".") else "<root>"
    fallback_title = direct_parent.name if direct_parent != Path(".") else "记忆"
    return fallback_key, fallback_title


def index_build_fingerprint(database_path: Path) -> str:
    connection = sqlite3.connect(database_path)
    try:
        meta = connection.execute("SELECT key,value FROM index_meta ORDER BY key").fetchall()
        sources = connection.execute(
            "SELECT source_id,content_hash FROM source_file WHERE role='canonical' ORDER BY source_id"
        ).fetchall()
        try:
            retrieval = connection.execute(
                "SELECT source_id,mode,surface,cues_json FROM source_retrieval ORDER BY source_id"
            ).fetchall()
        except sqlite3.Error:
            retrieval = []
    finally:
        connection.close()
    payload = json.dumps(
        {"meta": meta, "sources": sources, "retrieval": retrieval},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class CleanCandidateEngine:
    """Phase-1 engine with no v1 route, alias, threshold, or entity logic."""

    def __init__(
        self,
        config: ProjectConfig,
        scopes: ScopeCatalog,
        embedder: EmbeddingBackend,
        reranker: RerankBackend,
        database_path: Path | None = None,
        segment_store: object | None = None,
    ) -> None:
        self.config = config
        self.scopes = scopes
        self.embedder = embedder
        self.reranker = reranker
        self.database_path = database_path or config.project_root / "state" / "memory-index.sqlite3"
        self.segment_store = segment_store
        self._index_build_id = index_build_fingerprint(self.database_path)

    def _rows(self, scope: str) -> list[sqlite3.Row]:
        namespaces = sorted(self.scopes.namespaces_for(scope))
        if not namespaces or not self.database_path.is_file():
            return []
        placeholders = ",".join("?" for _ in namespaces)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(
                f"""
                SELECT c.*, s.path, s.namespace, s.role, s.content_hash,
                       COALESCE(sr.mode, 'full_text') AS retrieval_mode,
                       COALESCE(sr.surface, '') AS retrieval_surface,
                       COALESCE(sr.cues_json, '[]') AS retrieval_cues_json,
                       COALESCE(e.date_key, '') AS event_date
                FROM chunk c
                JOIN source_file s ON s.source_id=c.source_id
                LEFT JOIN source_retrieval sr ON sr.source_id=s.source_id
                LEFT JOIN event e ON e.event_id=c.event_id
                WHERE s.role='canonical' AND s.namespace IN ({placeholders})
                """,
                namespaces,
            ).fetchall()
        finally:
            connection.close()

    def build_fingerprint(self) -> str:
        return self._index_build_id

    def _group(self, prompt: str, scope: str) -> list[Candidate]:
        rows = self._rows(scope)
        if not rows:
            return []
        query_vector = self.embedder.encode_query(prompt).astype(np.float32, copy=False)
        grouped: dict[str, list[tuple[sqlite3.Row, float]]] = {}
        for row in rows:
            dimension = int(row["embedding_dim"])
            vector = np.frombuffer(row["embedding"], dtype=np.float32, count=dimension)
            if vector.shape[0] != query_vector.shape[0]:
                continue
            # A strict-surface file is an authored source-level summary.  Its
            # internal headings are reading sections, not independent catalog
            # entries competing under the same cue card.
            memory_key = (
                "strict_surface"
                if str(row["retrieval_mode"]) == "strict_surface"
                else str(row["event_id"] or row["chunk_id"])
            )
            key = f"{row['source_id']}::{memory_key}"
            grouped.setdefault(key, []).append((row, float(np.dot(vector, query_vector))))

        root = self.config.roots["memory"].resolve()
        index_build_id = self.build_fingerprint()
        candidates: list[Candidate] = []
        for build_id, items in grouped.items():
            ordered = sorted(items, key=lambda item: int(item[0]["start_line"]))
            first = ordered[0][0]
            path = Path(str(first["path"])).resolve()
            relative_path = path.relative_to(root).as_posix()
            navigation_key, navigation_title = navigation_metadata(root, relative_path)
            headings = tuple(dict.fromkeys(str(item[0]["heading_path"]) for item in ordered))
            excerpts = tuple(dict.fromkeys(str(item[0]["excerpt"]) for item in ordered))
            chunks = tuple(
                ChunkRef(
                    chunk_id=str(row["chunk_id"]),
                    heading_path=str(row["heading_path"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    excerpt=str(row["excerpt"]),
                    vector_score=score,
                    canonical_text=(
                        self.segment_store.get(str(row["chunk_id"]))
                        if self.segment_store is not None
                        else None
                    ),
                )
                for row, score in ordered
            )
            candidates.append(
                Candidate(
                    index_build_id=index_build_id,
                    build_id=build_id,
                    source_id=str(first["source_id"]),
                    event_id=(
                        None
                        if str(first["retrieval_mode"]) == "strict_surface"
                        else (str(first["event_id"]) if first["event_id"] else None)
                    ),
                    relative_path=relative_path,
                    namespace=str(first["namespace"]),
                    event_date=(
                        ""
                        if str(first["retrieval_mode"]) == "strict_surface"
                        else str(first["event_date"])
                    ),
                    headings=headings,
                    excerpts=excerpts,
                    chunks=chunks,
                    source_hash=str(first["content_hash"]),
                    navigation_key=navigation_key,
                    navigation_title=navigation_title,
                    vector_score=max(score for _, score in items),
                    retrieval_mode=str(first["retrieval_mode"]),
                    retrieval_surface=str(first["retrieval_surface"]),
                    retrieval_cues=tuple(json.loads(str(first["retrieval_cues_json"]))),
                )
            )
        return candidates

    def search(
        self,
        prompt: str,
        *,
        scope: str = "ordinary",
        recent_turns: Iterable[str] = (),
        dense_pool_k: int | None = None,
        directory_k: int | None = None,
        parent_cap: int | None = None,
        apply_evidence_floor: bool = False,
        working_context: str = "",
    ) -> dict[str, object]:
        recent = [str(item).strip() for item in recent_turns if str(item).strip()]
        query = (
            "\n".join([*(f"近期用户消息：{item}" for item in recent), f"当前消息：{prompt}"])
            if recent
            else prompt
        )
        all_candidates = sorted(
            self._group(query, scope), key=lambda item: item.vector_score, reverse=True
        )
        pool_size = dense_pool_k or self.scopes.dense_pool_k
        dense_pool = all_candidates[:pool_size]
        dense_pool = [
            replace(item, matched_cues=matched_cues(query, item.document))
            for item in dense_pool
        ]
        bounded_context = str(working_context).strip()[-100:]
        rerank_documents = [item.document for item in dense_pool]
        if bounded_context:
            rerank_documents.append(bounded_context)
        scores = self.reranker.score(query, rerank_documents)
        memory_scores = scores[: len(dense_pool)]
        context_score = float(scores[-1]) if bounded_context else None
        reranked = sorted(
            (
                replace(item, rerank_score=float(score))
                for item, score in zip(dense_pool, memory_scores)
            ),
            key=lambda item: (item.rerank_score, item.vector_score),
            reverse=True,
        )
        admitted = (
            [
                item
                for item in reranked
                if _candidate_admitted(item, query, self.scopes.evidence_floor)
            ]
            if apply_evidence_floor
            else reranked
        )
        context_position = context_rank(reranked, context_score=context_score)
        context_outranking_count = (
            sum(item.rerank_score > context_score for item in admitted)
            if context_score is not None
            else None
        )
        admitted, context_suppressed = apply_context_cutoff(
            admitted, context_score=context_score
        )
        context_won = bool(context_position == 1)
        evidence_rejected = [item for item in reranked if item not in admitted and item not in context_suppressed]
        directory, cap_rejected = apply_directory_policy(
            admitted,
            directory_k=directory_k or self.scopes.directory_k,
            parent_cap=parent_cap or self.scopes.parent_cap,
        )
        return {
            "scope": scope,
            "all_dense": all_candidates,
            "dense_pool": dense_pool,
            "reranked": reranked,
            "admitted": admitted,
            "evidence_rejected": evidence_rejected,
            "directory": directory,
            "cap_rejected": cap_rejected,
            "context_suppressed": context_suppressed,
            "context_competitor": {
                "used": bool(bounded_context),
                "won": context_won,
                "rank": context_position,
                "score": context_score,
                "top_memory_score": reranked[0].rerank_score if reranked else None,
                "memories_before": context_outranking_count,
                "kept_count": len(admitted),
                "suppressed_count": len(context_suppressed),
            },
        }

    def search_hybrid(
        self,
        prompt: str,
        *,
        scope: str = "ordinary",
        recent_turns: Iterable[str] = (),
        dense_pool_k: int | None = None,
        lexical_pool_k: int | None = None,
        directory_k: int | None = None,
        parent_cap: int | None = None,
        rrf_k: int = 60,
        apply_evidence_floor: bool = False,
        working_context: str = "",
    ) -> dict[str, object]:
        """Dense + character n-gram BM25 candidates, fused by equal-weight RRF."""

        recent = [str(item).strip() for item in recent_turns if str(item).strip()]
        query = (
            "\n".join([*(f"近期用户消息：{item}" for item in recent), f"当前消息：{prompt}"])
            if recent
            else prompt
        )
        grouped = self._group(query, scope)
        documents = [
            item.document
            for item in grouped
        ]
        lexical_values = bm25_scores(query, documents)
        scored = [
            replace(
                candidate,
                lexical_score=float(score),
                matched_cues=matched_cues(query, document),
            )
            for candidate, score, document in zip(grouped, lexical_values, documents)
        ]
        dense_ranked = sorted(scored, key=lambda item: item.vector_score, reverse=True)
        lexical_ranked = sorted(
            scored,
            key=lambda item: (item.lexical_score, item.vector_score),
            reverse=True,
        )
        dense_size = dense_pool_k or self.scopes.dense_pool_k
        lexical_size = lexical_pool_k or dense_size
        pool_by_id: dict[str, Candidate] = {}
        for candidate in dense_ranked[:dense_size] + lexical_ranked[:lexical_size]:
            pool_by_id[candidate.build_id] = candidate
        pool = list(pool_by_id.values())
        bounded_context = str(working_context).strip()[-100:]
        rerank_documents = [item.document for item in pool]
        if bounded_context:
            rerank_documents.append(bounded_context)
        rerank_values = self.reranker.score(query, rerank_documents)
        memory_rerank_values = rerank_values[: len(pool)]
        context_score = float(rerank_values[-1]) if bounded_context else None
        reranked = sorted(
            (
                replace(item, rerank_score=float(score))
                for item, score in zip(pool, memory_rerank_values)
            ),
            key=lambda item: (item.rerank_score, item.vector_score),
            reverse=True,
        )
        lexical_ranks = {item.build_id: rank for rank, item in enumerate(lexical_ranked, 1)}
        rerank_ranks = {item.build_id: rank for rank, item in enumerate(reranked, 1)}
        fused = sorted(
            (
                replace(
                    item,
                    fusion_score=(
                        (
                            reciprocal_rank(lexical_ranks[item.build_id], k=rrf_k)
                            if item.lexical_score > 0
                            else 0.0
                        )
                        + reciprocal_rank(rerank_ranks[item.build_id], k=rrf_k)
                    ),
                )
                for item in reranked
            ),
            key=lambda item: (
                item.fusion_score,
                item.rerank_score,
                item.lexical_score,
                item.vector_score,
            ),
            reverse=True,
        )
        admitted = (
            [
                item
                for item in fused
                if _candidate_admitted(item, query, self.scopes.evidence_floor)
            ]
            if apply_evidence_floor
            else fused
        )
        context_position = context_rank(reranked, context_score=context_score)
        context_outranking_count = (
            sum(item.rerank_score > context_score for item in admitted)
            if context_score is not None
            else None
        )
        admitted, context_suppressed = apply_context_cutoff(
            admitted, context_score=context_score
        )
        context_won = bool(context_position == 1)
        evidence_rejected = [item for item in fused if item not in admitted and item not in context_suppressed]
        directory, cap_rejected = apply_directory_policy(
            admitted,
            directory_k=directory_k or self.scopes.directory_k,
            parent_cap=parent_cap or self.scopes.parent_cap,
        )
        return {
            "scope": scope,
            "all_dense": dense_ranked,
            "dense_pool": dense_ranked[:dense_size],
            "all_lexical": lexical_ranked,
            "candidate_pool": pool,
            "reranked": reranked,
            "fused": fused,
            "admitted": admitted,
            "evidence_rejected": evidence_rejected,
            "directory": directory,
            "cap_rejected": cap_rejected,
            "rrf_k": rrf_k,
            "context_suppressed": context_suppressed,
            "context_competitor": {
                "used": bool(bounded_context),
                "won": context_won,
                "rank": context_position,
                "score": context_score,
                "top_memory_score": reranked[0].rerank_score if reranked else None,
                "memories_before": context_outranking_count,
                "kept_count": len(admitted),
                "suppressed_count": len(context_suppressed),
            },
        }

    def catalog(self, prompt: str, *, scope: str = "ordinary") -> list[Candidate]:
        """Return every readable node in a stable order.

        The prompt is used only to score chunks inside each node so a later
        read still returns the most relevant segments first.  It never changes
        catalog membership or order, keeping the rendered catalog cacheable.
        """

        return sorted(
            self._group(prompt, scope),
            key=lambda item: (
                item.relative_path.casefold(),
                min((chunk.start_line for chunk in item.chunks), default=0),
                item.build_id,
            ),
        )


def candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "index_build_id": candidate.index_build_id,
        "build_id": candidate.build_id,
        "source_path": candidate.relative_path,
        "source_id": candidate.source_id,
        "event_id": candidate.event_id,
        "namespace": candidate.namespace,
        "event_date": candidate.event_date,
        "display_date": candidate.display_date,
        "navigation_key": candidate.navigation_key,
        "navigation_title": candidate.navigation_title,
        "source_title": candidate.source_title,
        "leaf_title": candidate.leaf_title,
        "hierarchy": candidate.hierarchy,
        "headings": list(candidate.headings),
        "retrieval_mode": candidate.retrieval_mode,
        "vector_score": candidate.vector_score,
        "lexical_score": candidate.lexical_score,
        "rerank_score": candidate.rerank_score,
        "fusion_score": candidate.fusion_score,
        "matched_cues": list(candidate.matched_cues),
    }
