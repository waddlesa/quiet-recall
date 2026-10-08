from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntityResolution:
    entity_ids: tuple[str, ...]
    ambiguous: dict[str, str]
    evidence: dict[str, tuple[str, ...]]

    @property
    def abstained(self) -> bool:
        return any(value == "unknown" for value in self.ambiguous.values())


class EntityResolver:
    """Resolve registered names while abstaining on ambiguous nicknames."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.entities = config.get("entities", {})
        self.disambiguation = config.get("disambiguation", {})
        self.strong_aliases: list[tuple[str, str]] = []
        self.ambiguous_aliases: dict[str, set[str]] = {}
        for entity_id, item in self.entities.items():
            for alias in item.get("strong_aliases", []):
                self.strong_aliases.append((str(alias), str(entity_id)))
            for alias in item.get("ambiguous_aliases", []):
                self.ambiguous_aliases.setdefault(str(alias), set()).add(str(entity_id))
        self.strong_aliases.sort(key=lambda pair: len(pair[0]), reverse=True)

    def find_strong_entities(self, text: str) -> tuple[str, ...]:
        found = {entity_id for alias, entity_id in self.strong_aliases if alias in text}
        return tuple(sorted(found))

    def _ambiguous_decision(self, alias: str, context: str) -> tuple[str, tuple[str, ...]]:
        shen_cues = tuple(str(cue) for cue in self.disambiguation.get("shen_huai_cues", []))
        cat_cues = tuple(str(cue) for cue in self.disambiguation.get("real_cat_cues", []))
        strong_names = tuple(name for name, entity_id in self.strong_aliases if entity_id == "shen-huai")
        shen_hits = tuple(cue for cue in shen_cues + strong_names if cue in context)
        cat_hits_list = [cue for cue in cat_cues if cue in context]
        if "监控找猫" in cat_cues and "监控" in context and "找猫" in context:
            cat_hits_list.append("监控找猫")
        cat_hits = tuple(cat_hits_list)
        if len(shen_hits) > len(cat_hits):
            candidates = self.ambiguous_aliases.get(alias, set())
            if len(candidates) == 1:
                return next(iter(candidates)), shen_hits
        if len(cat_hits) > len(shen_hits):
            return "real_cat", cat_hits
        return "unknown", tuple(sorted(set(shen_hits + cat_hits)))

    def resolve(self, turns: list[str] | tuple[str, ...]) -> EntityResolution:
        if not turns:
            return EntityResolution((), {}, {})
        current = str(turns[-1])
        strong_current = set(self.find_strong_entities(current))
        ambiguous: dict[str, str] = {}
        evidence: dict[str, tuple[str, ...]] = {}

        window = self.disambiguation.get("recent_turn_window", [2, 4])
        max_turns = int(window[-1]) if isinstance(window, list) and window else 4
        context = "\n".join(str(turn) for turn in turns[-max_turns:])
        for alias in self.ambiguous_aliases:
            # A longer strong alias has already resolved the subject.
            if alias not in current or strong_current:
                continue
            decision, hits = self._ambiguous_decision(alias, context)
            ambiguous[alias] = decision
            evidence[alias] = hits
            if decision not in {"unknown", "real_cat"}:
                strong_current.add(decision)

        return EntityResolution(tuple(sorted(strong_current)), ambiguous, evidence)

    def identity_card(self, entity_id: str) -> str | None:
        item = self.entities.get(entity_id)
        return str(item.get("identity_card")) if isinstance(item, dict) and item.get("identity_card") else None
