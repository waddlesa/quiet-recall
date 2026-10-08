from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable, Mapping

from app.card_builder import estimate_tokens, truncate_text


@dataclass(frozen=True)
class AdvisoryPacket:
    text: str
    estimated_tokens: int
    visible_handles: tuple[str, ...]


def _candidate_line(item: Mapping[str, object], scope: str) -> str:
    handle = escape(str(item.get("capability_handle", "")))
    hierarchy = escape(str(item.get("hierarchy", "")))
    scope_attribute = f' s="{escape(scope)}"' if scope != "ordinary" else ""
    return f'<c h="{handle}"{scope_attribute}>{hierarchy}</c>'


def _candidate_line_with_hierarchy(
    item: Mapping[str, object], scope: str, hierarchy: str
) -> str:
    handle = escape(str(item.get("capability_handle", "")))
    scope_attribute = f' s="{escape(scope)}"' if scope != "ordinary" else ""
    return f'<c h="{handle}"{scope_attribute}>{escape(hierarchy)}</c>'


def _compact_candidate_line(item: Mapping[str, object], scope: str, label: str) -> str:
    handle = escape(str(item.get("capability_handle", "")))
    prefix = f"{escape(scope)}/" if scope != "ordinary" else ""
    return f"{prefix}{handle}:{escape(label)}"


def _candidate_label(item: Mapping[str, object], scope: str) -> str:
    """Use the leaf as the cheap decision surface; expand parents on read.

    Vault directories retain their hierarchy because choosing the wrong sibling
    there is costlier and their candidate sets are deliberately small.
    """

    hierarchy = str(item.get("hierarchy", ""))
    document = str(item.get("document_title", "")).strip()
    leaf = str(item.get("leaf_title", "")).strip()
    if scope == "ordinary" and (document or leaf):
        return document or leaf
    return hierarchy or leaf


def render_turn_plan_packet(
    *,
    turn_handle: str,
    directories: Iterable[tuple[str, Iterable[Mapping[str, object]]]],
    descriptions: Mapping[str, str] | None = None,
    explicit_recall: bool = False,
    token_budget: int = 200,
) -> AdvisoryPacket:
    """Render the complete non-body memory packet under one hard budget.

    Candidate order is preserved. Handles that do not fit are deliberately not
    exposed; callers should revoke them from the live turn registry.
    """

    lines = [
        f'<memory_index v="2" turn="{escape(turn_handle)}">',
        "<rule>每行是句柄:标题。相关时先读，勿凭标题作答。</rule>",
    ]
    if explicit_recall:
        lines.append("<r>追问旧事，先读后答。</r>")

    if descriptions:
        compact = "；".join(
            f"{name}:{description}"
            for name, description in descriptions.items()
            if name != "ordinary"
        )
        if compact:
            scope_line = f"<scopes>{escape(compact)}</scopes>"
            trial = "\n".join([*lines, scope_line, "</memory_index>"])
            if estimate_tokens(trial) <= token_budget:
                lines.append(scope_line)

    handles: list[str] = []
    for scope, directory in directories:
        for item in directory:
            label = _candidate_label(item, scope)
            line = _compact_candidate_line(item, scope, label)
            trial = "\n".join([*lines, line, "</memory_index>"])
            if estimate_tokens(trial) > token_budget:
                fitted = ""
                # Prefer a shortened high-ranked title over silently skipping
                # it and showing a lower-ranked candidate.
                for label_budget in range(
                    min(estimate_tokens(label), token_budget), 3, -1
                ):
                    shortened = truncate_text(label, label_budget)
                    candidate_line = _compact_candidate_line(item, scope, shortened)
                    candidate_trial = "\n".join(
                        [*lines, candidate_line, "</memory_index>"]
                    )
                    if estimate_tokens(candidate_trial) <= token_budget:
                        fitted = candidate_line
                        break
                if not fitted:
                    break
                line = fitted
            lines.append(line)
            handles.append(str(item.get("capability_handle", "")))

    lines.append("</memory_index>")
    text = "\n".join(lines)
    if estimate_tokens(text) > token_budget:
        # Defensive fallback for very small custom budgets. It intentionally
        # exposes no handles rather than returning malformed or over-budget XML.
        minimal = (
            f'<memory_index v="2" turn="{escape(turn_handle)}">'
            f"{truncate_text('候选元数据预算不足，本轮不提供记忆候选。', max(1, token_budget // 3))}"
            "</memory_index>"
        )
        if estimate_tokens(minimal) > token_budget:
            return AdvisoryPacket("", 0, ())
        return AdvisoryPacket(minimal, estimate_tokens(minimal), ())
    return AdvisoryPacket(text, estimate_tokens(text), tuple(handles))


def render_candidate_index(
    *,
    turn_handle: str,
    directory: Iterable[Mapping[str, object]],
    scope: str = "ordinary",
    token_budget: int = 200,
) -> str:
    return render_turn_plan_packet(
        turn_handle=turn_handle,
        directories=[(scope, directory)],
        token_budget=token_budget,
    ).text


def render_scope_catalog(descriptions: Mapping[str, str]) -> str:
    values = "；".join(
        f"{escape(str(name))}:{escape(str(description))}"
        for name, description in descriptions.items()
    )
    return f"<memory_scopes>{values}</memory_scopes>"


def render_static_candidate_catalog(
    *, directory: Iterable[Mapping[str, object]], scope: str = "ordinary"
) -> str:
    """Legacy experiment renderer; static-full is not part of the live plan."""

    lines = [f'<memory_candidate_catalog advisory="true" version="2" scope="{escape(scope)}">']
    for item in directory:
        lines.append(_candidate_line(item, scope))
    lines.append("</memory_candidate_catalog>")
    return "\n".join(lines)


def render_turn_handle(turn_handle: str) -> str:
    return f"<memory_turn_handle>{escape(turn_handle)}</memory_turn_handle>"
