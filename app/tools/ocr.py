"""
app/tools/ocr.py — единая точка входа для OCR страниц PDF.
"""
from __future__ import annotations

from typing import Literal

from PIL import Image

from app.config import get_settings

settings = get_settings()
OcrBackend = Literal["paddle", "openrouter"]


async def ocr_pil_image(image: Image.Image, page_num: int) -> str:
    backend = settings.ocr_backend.lower()
    if backend == "openrouter":
        from app.tools.llm_ocr import ocr_pil_image as _ocr
    elif backend == "paddle":
        from app.tools.paddle_ocr import ocr_pil_image as _ocr
    else:
        raise ValueError(
            f"Unknown OCR_BACKEND={settings.ocr_backend!r}. Use 'openrouter' or 'paddle'."
        )
    return await _ocr(image, page_num=page_num)


def ocr_backend_label() -> str:
    if settings.ocr_backend == "openrouter":
        return f"OpenRouter ({settings.ocr_model})"
    return f"PaddleOCR ({settings.ocr_lang})"
