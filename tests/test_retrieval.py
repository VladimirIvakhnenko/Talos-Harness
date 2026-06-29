"""Unit tests for retrieval helpers (no DB)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.memory.store import reciprocal_rank_fusion  # noqa: E402


def test_rrf_merges_lists():
    dense = [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}]
    kw = [{"id": 2, "content": "b"}, {"id": 3, "content": "c"}]
    merged = reciprocal_rank_fusion([dense, kw], k=60)
    ids = [m["id"] for m in merged]
    assert ids[0] == 2
    assert set(ids) == {1, 2, 3}


if __name__ == "__main__":
    test_rrf_merges_lists()
    print("OK: retrieval tests passed")
