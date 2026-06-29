"""
app/tools/markdown_chunker.py — структурный чанкинг Markdown для RAG.

Parent = раздел уровня parent_level (по умолчанию H3 / ###).
Child = подразделы, таблицы, code blocks, prose (с fallback recursive split).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ContentType = Literal["prose", "table", "code", "mixed"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_HAS_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class ChildChunk:
    content: str
    content_type: ContentType
    heading_path: list[str] = field(default_factory=list)


@dataclass
class StructuralSection:
    parent_text: str
    children: list[ChildChunk]
    heading_path: list[str] = field(default_factory=list)
    heading_level: int = 0
    content_types: list[str] = field(default_factory=list)
    chunking: str = "markdown_structural"


@dataclass
class _Block:
    kind: Literal["heading", "code", "table", "prose", "hr"]
    text: str = ""
    level: int = 0
    title: str = ""


def has_markdown_headings(text: str) -> bool:
    return bool(_HAS_HEADING_RE.search(text))


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end() :]


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and "|" in s[1:-1]


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    inner = s.strip("|").replace("|", "").strip()
    return bool(inner) and all(c in "-: " for c in inner)


def _parse_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = _HEADING_RE.match(line)
        if hm:
            blocks.append(
                _Block("heading", level=len(hm.group(1)), title=hm.group(2).strip())
            )
            i += 1
            continue
        if _FENCE_RE.match(line):
            lang = _FENCE_RE.match(line).group(1) or ""
            buf = [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip().startswith("```") and len(buf) > 1:
                    i += 1
                    break
                i += 1
            blocks.append(_Block("code", text="\n".join(buf), title=lang))
            continue
        if line.strip() == "---":
            blocks.append(_Block("hr"))
            i += 1
            continue
        if _is_table_line(line):
            buf = [line]
            i += 1
            while i < len(lines) and (_is_table_line(lines[i]) or _is_table_separator(lines[i])):
                buf.append(lines[i])
                i += 1
            blocks.append(_Block("table", text="\n".join(buf)))
            continue
        if not line.strip():
            i += 1
            continue
        buf = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (
                _HEADING_RE.match(nxt)
                or _FENCE_RE.match(nxt)
                or nxt.strip() == "---"
                or _is_table_line(nxt)
            ):
                break
            buf.append(nxt)
            i += 1
        prose = "\n".join(buf).strip()
        if prose:
            blocks.append(_Block("prose", text=prose))
    return blocks


def _breadcrumb(path: list[str]) -> str:
    if not path:
        return ""
    return " > ".join(path) + "\n"


def _split_prose(
    text: str,
    path: list[str],
    child_size: int,
    child_overlap: int,
) -> list[ChildChunk]:
    prefix = _breadcrumb(path)
    body = text.strip()
    if not body:
        return []
    if len(prefix) + len(body) <= child_size:
        return [ChildChunk(prefix + body, "prose", list(path))]
    from app.tools.doc_indexer import recursive_chunk

    parts = recursive_chunk(body, child_size - len(prefix), child_overlap)
    return [ChildChunk(prefix + p, "prose", list(path)) for p in parts if p.strip()]


def _blocks_to_children(
    blocks: list[_Block],
    base_path: list[str],
    child_size: int,
    child_overlap: int,
    subsection_level: int = 4,
) -> list[ChildChunk]:
    children: list[ChildChunk] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "heading" and b.level >= subsection_level:
            sub_path = base_path + [b.title]
            sub_blocks: list[_Block] = []
            i += 1
            while i < len(blocks):
                nb = blocks[i]
                if nb.kind == "heading" and nb.level >= subsection_level:
                    break
                sub_blocks.append(nb)
                i += 1
            children.extend(
                _blocks_to_children(sub_blocks, sub_path, child_size, child_overlap, subsection_level)
            )
            continue
        if b.kind == "code":
            path = list(base_path)
            children.append(
                ChildChunk(_breadcrumb(path) + b.text, "code", path)
            )
            i += 1
            continue
        if b.kind == "table":
            path = list(base_path)
            children.append(
                ChildChunk(_breadcrumb(path) + b.text, "table", path)
            )
            i += 1
            continue
        if b.kind == "prose":
            children.extend(
                _split_prose(b.text, base_path, child_size, child_overlap)
            )
            i += 1
            continue
        i += 1
    return children


def _section_content_types(children: list[ChildChunk]) -> list[str]:
    seen: list[str] = []
    for c in children:
        if c.content_type not in seen:
            seen.append(c.content_type)
    return seen or ["prose"]


def _render_parent_body(title: str, level: int, blocks: list[_Block]) -> str:
    hashes = "#" * level
    parts = [f"{hashes} {title}"] if title else []
    for b in blocks:
        if b.kind == "heading":
            parts.append(f"{'#' * b.level} {b.title}")
        elif b.kind in ("code", "table", "prose"):
            parts.append(b.text)
    return "\n\n".join(parts).strip()


def markdown_structural_chunk(
    text: str,
    *,
    parent_level: int = 3,
    parent_max_chars: int = 1500,
    child_size: int = 800,
    child_overlap: int = 80,
) -> list[StructuralSection]:
    """
    Разбить Markdown на parent-child секции.
    Если заголовков нет — пустой список (вызывающий код использует recursive fallback).
    """
    if not has_markdown_headings(text):
        return []

    _, body = _parse_frontmatter(text)
    blocks = _parse_blocks(body)
    sections: list[StructuralSection] = []
    heading_stack: list[tuple[int, str]] = []

    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind != "heading":
            i += 1
            continue

        while heading_stack and heading_stack[-1][0] >= b.level:
            heading_stack.pop()
        heading_stack.append((b.level, b.title))

        if b.level > parent_level:
            i += 1
            continue

        if b.level < parent_level:
            i += 1
            continue

        path = [t for _, t in heading_stack]
        section_blocks: list[_Block] = []
        i += 1
        while i < len(blocks):
            nb = blocks[i]
            if nb.kind == "heading" and nb.level <= parent_level:
                break
            section_blocks.append(nb)
            i += 1

        parent_text = _render_parent_body(b.title, b.level, section_blocks)
        children = _blocks_to_children(section_blocks, path, child_size, child_overlap)

        if not children and parent_text.strip():
            children = _split_prose(parent_text, path, child_size, child_overlap)

        if parent_text.strip() and children:
            sections.append(
                StructuralSection(
                    parent_text=parent_text,
                    children=children,
                    heading_path=path,
                    heading_level=b.level,
                    content_types=_section_content_types(children),
                )
            )

    return sections


def legacy_recursive_sections(
    full_text: str,
    parent_size: int = 1500,
    parent_overlap: int = 150,
    child_size: int = 800,
    child_overlap: int = 80,
) -> list[StructuralSection]:
    """Обёртка над recursive_chunk для non-MD документов."""
    from app.tools.doc_indexer import recursive_chunk

    sections: list[StructuralSection] = []
    for pi, parent_text in enumerate(recursive_chunk(full_text, parent_size, parent_overlap)):
        if not parent_text.strip():
            continue
        child_texts = recursive_chunk(parent_text, child_size, child_overlap)
        children = [
            ChildChunk(t, "prose", [])
            for t in child_texts
            if t.strip()
        ]
        if not children:
            continue
        sections.append(
            StructuralSection(
                parent_text=parent_text,
                children=children,
                heading_path=[],
                heading_level=0,
                content_types=["prose"],
                chunking="recursive",
            )
        )
    return sections
