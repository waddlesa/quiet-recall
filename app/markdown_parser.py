from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
MARKUP_ONLY_RE = re.compile(r"^[\s|:_*#>`~=-]+$")
YAML_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*:")


@dataclass(frozen=True)
class MarkdownChunk:
    heading_path: str
    start_line: int
    end_line: int
    text: str
    block_type: str


@dataclass(frozen=True)
class ParseReport:
    path: Path
    chunks: tuple[MarkdownChunk, ...]
    skipped: dict[str, int]


def _frontmatter_bounds(lines: list[str]) -> tuple[int, int] | None:
    if not lines or lines[0].strip() != "---":
        return None
    for index, candidate in enumerate(lines[1:], start=2):
        if candidate.strip() not in {"---", "..."}:
            continue
        body = lines[1 : index - 1]
        # A leading horizontal rule is ordinary Markdown. Treat it as YAML
        # frontmatter only when the delimited body actually looks like YAML.
        if any(YAML_FIELD_RE.match(item.strip()) for item in body):
            return 1, index
        return None
    return None


def parse_frontmatter(path: Path) -> dict[str, object]:
    """Read optional YAML metadata without exposing it as memory prose."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    bounds = _frontmatter_bounds(lines)
    if bounds is None:
        return {}
    start, end = bounds
    value = yaml.safe_load("\n".join(lines[start : end - 1])) or {}
    if not isinstance(value, dict):
        raise ValueError(f"frontmatter must be a mapping: {path}")
    return {str(key): item for key, item in value.items()}


def _clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_markdown(path: Path) -> ParseReport:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    headings: list[str] = []
    chunks: list[MarkdownChunk] = []
    skipped = {"blank": 0, "code": 0, "markup_only": 0}
    block_lines: list[str] = []
    block_start = 0
    block_type = "paragraph"
    in_fence = False
    fence_token = ""
    bounds = _frontmatter_bounds(lines)
    frontmatter_end = bounds[1] if bounds else 0

    def heading_path() -> str:
        return " / ".join(headings) if headings else path.name

    def flush(end_line: int) -> None:
        nonlocal block_lines, block_start, block_type
        if not block_lines:
            return
        text = _clean_inline("\n".join(block_lines))
        if not text or MARKUP_ONLY_RE.fullmatch(text):
            skipped["markup_only"] += 1
        else:
            chunks.append(
                MarkdownChunk(
                    heading_path=heading_path(),
                    start_line=block_start,
                    end_line=end_line,
                    text=text,
                    block_type=block_type,
                )
            )
        block_lines = []
        block_start = 0
        block_type = "paragraph"

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if frontmatter_end and line_number <= frontmatter_end:
            skipped["code"] += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            flush(line_number - 1)
            token = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_token = token
            elif token == fence_token:
                in_fence = False
                fence_token = ""
            skipped["code"] += 1
            continue
        if in_fence:
            skipped["code"] += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_number - 1)
            level = len(heading.group(1))
            title = _clean_inline(heading.group(2))
            headings[:] = headings[: level - 1]
            while len(headings) < level - 1:
                headings.append("(untitled)")
            headings.append(title)
            continue

        if not stripped:
            flush(line_number - 1)
            skipped["blank"] += 1
            continue

        is_list = bool(LIST_RE.match(line))
        next_type = "list" if is_list else "paragraph"
        if block_lines and next_type != block_type:
            flush(line_number - 1)
        if not block_lines:
            block_start = line_number
            block_type = next_type
        block_lines.append(line)

    flush(len(lines))
    return ParseReport(path=path, chunks=tuple(chunks), skipped=skipped)
