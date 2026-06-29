"""Tests for markdown structural chunking."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.markdown_chunker import (  # noqa: E402
    has_markdown_headings,
    markdown_structural_chunk,
)


GUIDE = ROOT / "benchmark" / "assets" / "IEC-61131-3-ST-GUIDE.md"


def test_has_headings():
    assert has_markdown_headings("### Section\n\ntext")
    assert not has_markdown_headings("plain text without headers")


def test_code_block_atomic():
    md = """### Test

#### Example

```pascal
PROGRAM Foo
VAR
    x : INT;
END_VAR
END_PROGRAM
```
"""
    sections = markdown_structural_chunk(md)
    assert len(sections) == 1
    code_children = [c for c in sections[0].children if c.content_type == "code"]
    assert len(code_children) == 1
    assert "PROGRAM Foo" in code_children[0].content
    assert "END_PROGRAM" in code_children[0].content


def test_table_atomic():
    md = """### Types

#### 2.1 Table

| Тип | Описание |
|-----|----------|
| BOOL | TRUE / FALSE |
| INT | 16 bit |
"""
    sections = markdown_structural_chunk(md)
    tables = [c for s in sections for c in s.children if c.content_type == "table"]
    assert len(tables) == 1
    assert "BOOL" in tables[0].content
    assert "| INT |" in tables[0].content


def test_heading_path_nested():
    md = """### 3. Blocks

#### 3.2.1 Timers

TON description here.
"""
    sections = markdown_structural_chunk(md)
    assert sections[0].heading_path == ["3. Blocks"]
    child_paths = [c.heading_path for c in sections[0].children]
    assert any("3.2.1 Timers" in p for p in child_paths)


def test_fallback_no_headings():
    plain = "Just some plain documentation without any markdown headers at all."
    assert markdown_structural_chunk(plain) == []


def test_guide_smoke():
    text = GUIDE.read_text(encoding="utf-8")
    sections = markdown_structural_chunk(text)
    h3_count = sum(1 for line in text.splitlines() if line.startswith("### "))
    assert len(sections) >= h3_count - 1
    assert all(s.parent_text.strip() for s in sections)
    assert all(s.children for s in sections)
    types = {c.content_type for s in sections for c in s.children}
    assert "code" in types
    assert "table" in types


if __name__ == "__main__":
    test_has_headings()
    test_code_block_atomic()
    test_table_atomic()
    test_heading_path_nested()
    test_fallback_no_headings()
    test_guide_smoke()
    print("OK: all markdown_chunker tests passed")
