"""
app/tools/llm_ocr.py — OCR страниц PDF через vision-модель OpenRouter.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

from app.config import get_settings

settings = get_settings()

_OCR_PROMPT = (
    "Extract all text from this PLC documentation page. "
    "Preserve technical terms, code blocks, and tables exactly as shown. "
    "Return plain text only, without commentary."
)


async def ocr_pil_image(image: Image.Image, page_num: int) -> str:
    try:
        import openai

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        resp = await client.chat.completions.create(
            model=settings.ocr_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }],
            max_tokens=4096,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text and getattr(msg, "reasoning", None):
            text = str(msg.reasoning).strip()
        return text
    except Exception as e:
        return f"[OCR failed page {page_num}: {e}]"
