from pathlib import Path

from app.config import get_settings

settings = get_settings()
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_last_st: dict[str, str] = {}
