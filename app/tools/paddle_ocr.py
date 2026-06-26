"""
app/tools/paddle_ocr.py — локальный OCR через PaddleOCR (PP-OCRv5, lang=ru).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from app.config import get_settings

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

settings = get_settings()
_ocr_engine: PaddleOCR | None = None


def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    from paddleocr import PaddleOCR

    kwargs: dict[str, Any] = {
        "lang": settings.ocr_lang,
        "ocr_version": settings.ocr_version,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if settings.ocr_use_server_models:
        kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
        kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"

    _ocr_engine = PaddleOCR(**kwargs)
    return _ocr_engine


def _sort_text_lines(data: dict[str, Any]) -> list[str]:
    texts: list[str] = data.get("rec_texts") or []
    scores: list[float] = list(data.get("rec_scores") or [])
    polys: list = list(data.get("rec_polys") or [])

    if not texts:
        return []

    min_score = settings.ocr_min_score
    indexed: list[tuple[float, float, str]] = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        if scores and i < len(scores) and scores[i] < min_score:
            continue
        y, x = 0.0, 0.0
        if i < len(polys) and polys[i] is not None and len(polys[i]):
            poly = polys[i]
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            y, x = min(ys), min(xs)
        indexed.append((y, x, text.strip()))

    indexed.sort(key=lambda item: (item[0], item[1]))
    return [line for _, _, line in indexed]


def ocr_pil_image_sync(image: Image.Image) -> str:
    ocr = get_ocr_engine()
    rgb = image.convert("RGB")
    result = ocr.predict(input=np.array(rgb))

    lines: list[str] = []
    for res in result:
        data = res.json if hasattr(res, "json") else {}
        lines.extend(_sort_text_lines(data))

    return "\n".join(lines)


async def ocr_pil_image(image: Image.Image, page_num: int) -> str:
    try:
        return await asyncio.to_thread(ocr_pil_image_sync, image)
    except ImportError as e:
        raise RuntimeError(
            "PaddleOCR не установлен. Используйте Docker (docker compose up ui) "
            "или установите: pip install paddlepaddle paddleocr opencv-python-headless"
        ) from e
    except Exception as e:
        return f"[OCR failed page {page_num}: {e}]"
