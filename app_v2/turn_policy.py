from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EvidenceFloor:
    """Low admission floor for the automatic ordinary-memory directory.

    The three signals are alternatives rather than cumulative requirements.
    This keeps unusual but strong lexical, semantic, or reranker evidence while
    dropping candidates that are weak on every channel.
    """

    lexical_min: float = 8.0
    rerank_min: float = 0.5
    vector_min: float = 0.60
    strict_lexical_min: float = 8.0
    strict_rerank_min: float = 0.5
    strict_cue_min: int = 2

    def admits(self, *, lexical: float, rerank: float, vector: float) -> bool:
        return (
            lexical >= self.lexical_min
            or rerank >= self.rerank_min
            or vector >= self.vector_min
        )

    def admits_strict(
        self,
        *,
        lexical: float,
        rerank: float,
        matched_cue_count: int,
        exact_cue: bool,
    ) -> bool:
        """Admit a broad summary only through its deliberately narrow surface.

        Body-vector similarity is intentionally absent. An exact authored cue
        is sufficient; otherwise at least two query/surface cues plus one
        normal lexical or reranker signal are required.
        """

        if exact_cue:
            return True
        if matched_cue_count < self.strict_cue_min:
            return False
        return lexical >= self.strict_lexical_min or rerank >= self.strict_rerank_min


@dataclass(frozen=True)
class MetadataPolicy:
    """Shared non-body context budget for one assistant turn."""

    total_tokens: int = 200
    directory_tokens: int = 140


# This is an intent vocabulary, not a topic or per-memory alias list. It only
# recognizes that the user is explicitly asking for previously shared facts.
_EXPLICIT_RECALL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?:我)?(?:之前|以前|曾经)?(?:已经)?(?:告诉|跟你说|和你说|给你讲|跟你讲)过",
        r"(?:你)?还记得",
        r"(?:你(?:之前|以前|曾经)|(?:之前|以前|曾经)你).{0,16}(?:的|过)",
        r"你(?:应该|本来就|不是)?知道我(?:有过|曾经|以前)",
        r"你(?:应该|本来就)知道",
        r"(?:我们|咱们)(?:之前|以前|曾经)(?:聊|说|讲|弄)过",
        r"(?:这|那)(?:部分|件事|段)?(?:你)?(?:有|留了|存了)?记忆",
        r"(?:do)?youremember",
        r"i(?:already)?toldyou",
        r"we(?:previously|once)?(?:talked|spoke)about",
        r"what(?:do)?yourememberabout",
    )
)

_QUOTED_OR_CODE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`]*`"),
    re.compile(r"“[^”]*”"),
    re.compile(r"‘[^’]*’"),
    re.compile(r'"[^"]*"'),
)

_SELF_ROUTE_PREFIX = (
    r"(?:我(?:自己)?(?:以前|之前|曾经|那次|这次|那段|当时|高中|初中|"
    r"有过|有|得过|被诊断为|经历过|出现过|发作过|去|的|又|最近|曾|复发|发生){0,4}"
    r"|(?:my|i(?:have|had|was|am)){1,3})"
)


def _unquoted_text(text: str) -> str:
    value = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith(">")
    )
    for pattern in _QUOTED_OR_CODE_PATTERNS:
        value = pattern.sub(" ", value)
    return value


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def is_explicit_recall_request(text: str) -> bool:
    normalized = normalize_text(_unquoted_text(text))
    return any(pattern.search(normalized) for pattern in _EXPLICIT_RECALL_PATTERNS)


def _self_owned_cue(normalized: str, cue: str) -> bool:
    return bool(re.search(f"{_SELF_ROUTE_PREFIX}{re.escape(cue)}", normalized))


def route_explicit_scope(
    text: str,
    routing_cues: Mapping[str, Sequence[str]],
    routing_subjects: Mapping[str, str] | None = None,
) -> str | None:
    """Select a vault only when one scope has uniquely strongest direct cues.

    The caller must first establish explicit recall intent. This function does
    not open a vault for an ordinary topic mention.
    """

    normalized = normalize_text(_unquoted_text(text))
    scored: list[tuple[int, int, str]] = []
    for scope, cues in routing_cues.items():
        matched = {
            normalize_text(cue)
            for cue in cues
            if cue
            and normalize_text(cue) in normalized
            and (
                (routing_subjects or {}).get(scope) != "self"
                or _self_owned_cue(normalized, normalize_text(cue))
            )
        }
        scored.append((sum(len(cue) for cue in matched), len(matched), scope))
    scored.sort(reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None
    return scored[0][2]
