"""
app/tools/pdf_processor.py — PDF → OCR → semantic chunking → pgvector.

Pipeline:
  PDF → pdf2image → OCR (PaddleOCR или OpenRouter VLM) → text
      → semantic/recursive chunking → parent-child → pgvector
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, AsyncIterator

from app.config import get_settings
from app.tools.doc_indexer import stream_index_text
from app.tools.ocr import ocr_pil_image

settings = get_settings()


async def stream_process_pdf(
    pdf_path: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Потоковая обработка PDF с событиями прогресса.
    Фазы: convert → ocr → chunk → embed → done
    """
    from pdf2image import convert_from_path

    yield {"phase": "convert", "message": "Конвертация PDF…"}

    pages = convert_from_path(pdf_path, dpi=200)
    total_pages = len(pages)
    all_text: list[str] = []

    for i, page in enumerate(pages):
        current = i + 1
        pct = int(current / total_pages * 100) if total_pages else 100
        yield {
            "phase": "ocr",
            "current": current,
            "total": total_pages,
            "pct": pct,
            "message": f"OCR: страница {current}/{total_pages} ({pct}%)",
        }
        text = await ocr_pil_image(page, page_num=current)
        all_text.append(text)

    full_text = "\n\n".join(all_text)
    name = source_name or Path(pdf_path).name

    async for ev in stream_index_text(
        full_text, doc_type, name, db, pages=total_pages
    ):
        yield ev


async def process_pdf(
    pdf_path: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
) -> dict:
    """Полный пайплайн обработки PDF (блокирующая обёртка над stream_process_pdf)."""
    result: dict = {"pages": 0, "chunks": 0, "chunk_ids": []}
    async for ev in stream_process_pdf(pdf_path, doc_type, source_name, db):
        if ev.get("phase") == "done":
            result = ev["result"]
    return result
