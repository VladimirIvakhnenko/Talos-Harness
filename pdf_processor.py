"""
app/tools/pdf_processor.py — PDF → OCR → semantic chunking → pgvector.

Pipeline:
  PDF → pdf2image → pages as base64 → Qwen3.5-9B (vision via OpenRouter)
      → text → semantic/recursive chunking → parent-child → pgvector
"""
from __future__ import annotations
import base64
import io
import uuid
from pathlib import Path
from typing import Callable

from app.config import get_settings
from app.agents.llm_client import embed_texts

settings = get_settings()


async def process_pdf(
    pdf_path: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
) -> dict:
    """
    Полный пайплайн обработки PDF.
    Возвращает { pages, chunks, chunk_ids }.
    """
    from pdf2image import convert_from_path

    pages = convert_from_path(pdf_path, dpi=200)
    all_text = []

    for i, page in enumerate(pages):
        text = await _ocr_page(page, page_num=i + 1)
        all_text.append(text)

    full_text = "\n\n".join(all_text)
    chunks = _recursive_chunk(full_text, settings.chunk_size, settings.chunk_overlap)

    if db is None:
        return {"pages": len(pages), "chunks": len(chunks), "chunk_ids": []}

    # Parent-Child split: parent ~3000 chars, child ~800 chars
    parent_chunks = _recursive_chunk(full_text, 3000, 300)
    chunk_ids = []

    for pi, parent_text in enumerate(parent_chunks):
        child_texts = _recursive_chunk(parent_text, 800, 80)
        if not child_texts:
            continue

        parent_emb_list = await embed_texts([parent_text])
        parent_emb = parent_emb_list[0]

        child_embs = await embed_texts(child_texts)
        children = [{"content": t, "embedding": e, "index": ci}
                    for ci, (t, e) in enumerate(zip(child_texts, child_embs))]

        meta = {
            "type": "doc",
            "doc_type": doc_type,
            "source": source_name or Path(pdf_path).name,
            "parent_index": pi,
        }

        from app.memory.store import add_parent_child
        pid, cids = await add_parent_child(db, parent_text, parent_emb, children, meta)
        chunk_ids.extend(cids)

    return {"pages": len(pages), "chunks": len(chunk_ids), "chunk_ids": chunk_ids}


async def _ocr_page(page_image, page_num: int) -> str:
    """OCR страницы через OpenRouter (qwen/qwen3.5-9b с vision)."""
    try:
        import openai
        buf = io.BytesIO()
        page_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        resp = await client.chat.completions.create(
            model=settings.engineer_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text",
                     "text": "Extract all text from this PLC documentation page. "
                             "Preserve technical terms, code blocks, tables exactly as shown. "
                             "Return plain text only."},
                ],
            }],
            max_tokens=2000,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        # Fallback: pypdf текстовый экстракт
        return f"[OCR failed page {page_num}: {e}]"


def _recursive_chunk(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Простой рекурсивный чанкер."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        parts = text.split(sep) if sep else list(text)
        if len(parts) > 1:
            chunks, current = [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = current[-overlap:] + sep + part if overlap and current else part
            if current:
                chunks.append(current)
            result = [c.strip() for c in chunks if c.strip()]
            if result:
                return result

    return [text[:size]]