import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.deps import UPLOAD_DIR
from app.database import get_db
from app.tools.ocr import ocr_backend_label
from app.tools.pdf_processor import process_pdf
from app.tools.text_processor import process_text_file

router = APIRouter(tags=["Documents"])

_DOCUMENT_SUFFIXES = {".pdf", ".md", ".txt"}


@router.post("/upload_document", summary="Загрузка документации (PDF, MD, TXT)")
async def upload_document(
    file: UploadFile = File(..., description="PDF, Markdown или текстовый файл"),
    doc_type: str = Query("general", description="iec_standard | elbrus_manual | tz | general"),
):
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _DOCUMENT_SUFFIXES:
        raise HTTPException(400, "Only PDF, MD and TXT files are accepted")

    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())

    async with get_db() as db:
        if suffix == ".pdf":
            result = await process_pdf(
                str(path),
                doc_type=doc_type,
                source_name=file.filename,
                db=db,
            )
            pipeline = f"pdf2image → {ocr_backend_label()} → parent-child chunking → pgvector"
        else:
            result = await process_text_file(
                str(path),
                doc_type=doc_type,
                source_name=file.filename,
                db=db,
            )
            pipeline = "read → parent-child chunking → pgvector"

    return {
        "status": "processed",
        "filename": file.filename,
        "format": suffix.lstrip("."),
        "doc_type": doc_type,
        "path": str(path),
        "pages": result["pages"],
        "chunks": result["chunks"],
        "chunk_ids": result["chunk_ids"],
        "pipeline": pipeline,
    }


@router.post("/upload_pdf", summary="Загрузка PDF документации (IEC 61131-3, Эльбрус, ТЗ)")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF файл"),
    doc_type: str = Query("general", description="iec_standard | elbrus_manual | tz | general"),
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())

    async with get_db() as db:
        result = await process_pdf(
            str(path),
            doc_type=doc_type,
            source_name=file.filename,
            db=db,
        )

    return {
        "status": "processed",
        "filename": file.filename,
        "doc_type": doc_type,
        "path": str(path),
        "pages": result["pages"],
        "chunks": result["chunks"],
        "chunk_ids": result["chunk_ids"],
        "pipeline": f"pdf2image → {ocr_backend_label()} → parent-child chunking → pgvector",
    }


@router.post("/upload_signals", summary="Загрузка таблицы сигналов CSV/XLSX")
async def upload_signals(
    file: UploadFile = File(..., description="CSV или XLSX с таблицей сигналов"),
    controller: str = Query("elbrus", description="elbrus | baikal | codesys"),
):
    if not file.filename or not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only CSV/XLSX accepted")
    path = UPLOAD_DIR / f"signals_{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())
    return {
        "status": "saved",
        "filename": file.filename,
        "controller": controller,
        "path": str(path),
        "next": f"POST /generate_module with signals_path={path}",
    }
