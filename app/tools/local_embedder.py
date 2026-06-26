"""
app/tools/local_embedder.py — локальная векторизация через llama.cpp server (GGUF).
"""
from __future__ import annotations

import httpx

from app.config import get_settings

settings = get_settings()


def _truncate(text: str) -> str:
    limit = settings.embedding_max_chars
    if len(text) <= limit:
        return text
    return text[:limit]


async def _encode_one(client: httpx.AsyncClient, text: str) -> list[float]:
    base = settings.llama_embedding_url.rstrip("/")
    payload = {
        "input": _truncate(text),
        "model": settings.embedding_model,
    }
    resp = await client.post(f"{base}/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = resp.json()
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    if not items:
        raise ValueError("llama.cpp returned empty embeddings response")
    emb = items[0]["embedding"]
    expected = settings.embedding_dimensions
    if len(emb) != expected:
        raise ValueError(
            f"GGUF model returned {len(emb)} dims, "
            f"but EMBEDDING_DIMENSIONS={expected}."
        )
    return emb


async def encode_texts(texts: list[str]) -> list[list[float]]:
    """По одному тексту на запрос — llama.cpp embedding mode лимитирует batch по токенам."""
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=120.0) as client:
        return [await _encode_one(client, text) for text in texts]
