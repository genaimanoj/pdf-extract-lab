"""FastAPI entrypoint for PDF Extractor Lab."""
from __future__ import annotations

import logging
import os
import pathlib
import re
import uuid

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import settings
from .extractors import DESCRIPTORS, get_extractor, list_descriptors


# Push provider keys from .env into the process environment so the descriptor
# availability check and any third-party SDK that reads env vars see the same
# source of truth.
for _env_name, _value in (
    ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
    ("GEMINI_API_KEY", settings.gemini_api_key),
):
    if _value and not os.environ.get(_env_name):
        os.environ[_env_name] = _value


logger = logging.getLogger("pdfextractorlab")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="PDF Extractor Lab", version="0.1.0")

_cors_origins = settings.cors_origins_list() or ["*"]
# Wildcard + credentials is invalid; strip credentials when origins are open.
_allow_credentials = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

UPLOAD_DIR = pathlib.Path(settings.upload_dir).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# file_id is always a lowercase uuid4 hex string generated server-side on upload.
_FILE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
# First four bytes of every valid PDF start with "%PDF".
_PDF_MAGIC = b"%PDF"


def _resolve_upload(file_id: str) -> pathlib.Path:
    """Resolve an upload path for file_id, rejecting any traversal attempts."""
    if not _FILE_ID_RE.match(file_id or ""):
        raise HTTPException(400, "Invalid file_id")
    path = (UPLOAD_DIR / f"{file_id}.pdf").resolve()
    try:
        path.relative_to(UPLOAD_DIR)
    except ValueError:
        # Path escaped UPLOAD_DIR somehow — refuse.
        raise HTTPException(400, "Invalid file_id")
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    return path


@app.get("/health")
def health():
    return {
        "ok": True,
        "engines": list(DESCRIPTORS.keys()),
        "default": settings.extraction_engine,
    }


@app.get("/engines")
def engines():
    """Return engine descriptors with runtime availability."""
    return {
        "engines": [d.to_public_dict() for d in list_descriptors()],
        "default": settings.extraction_engine,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf", "binary/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF accepted; got {file.content_type}")
    file_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}.pdf"
    total = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    first_chunk_checked = False
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            if not first_chunk_checked:
                if not chunk.startswith(_PDF_MAGIC):
                    await f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(400, "Uploaded file is not a valid PDF")
                first_chunk_checked = True
            total += len(chunk)
            if total > max_bytes:
                await f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            await f.write(chunk)
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")
    return {"file_id": file_id, "bytes": total, "filename": file.filename}


@app.get("/files/{file_id}")
def get_file(file_id: str):
    path = _resolve_upload(file_id)
    return FileResponse(path, media_type="application/pdf")


@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    # _resolve_upload raises 400 on bad id / 404 if already gone.
    path = _resolve_upload(file_id)
    try:
        path.unlink()
    except OSError as e:
        logger.exception("delete failed file_id=%s", file_id)
        raise HTTPException(500, "Could not delete file") from e
    return {"ok": True, "file_id": file_id}


@app.get("/pages/{file_id}/info")
def pages_info(file_id: str):
    import fitz
    path = _resolve_upload(file_id)
    pages = []
    with fitz.open(str(path)) as doc:
        for i, p in enumerate(doc):
            r = p.rect
            pages.append({"page": i + 1, "width": float(r.width), "height": float(r.height)})
    return {"file_id": file_id, "page_count": len(pages), "pages": pages}


@app.get("/pages/{file_id}/{page}.png")
def page_png(file_id: str, page: int, dpi: int = 120):
    import fitz
    path = _resolve_upload(file_id)
    # Clamp DPI to protect the process from memory exhaustion.
    dpi = max(settings.render_min_dpi, min(settings.render_max_dpi, dpi))
    with fitz.open(str(path)) as doc:
        if page < 1 or page > doc.page_count:
            raise HTTPException(404, "Page out of range")
        p = doc[page - 1]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        data = pix.tobytes("png")
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/extract")
async def extract(
    file_id: str = Form(...),
    engine: str = Form(None),
):
    engine_name = engine or settings.extraction_engine
    if engine_name not in DESCRIPTORS:
        raise HTTPException(400, f"Unknown engine '{engine_name}'. Known: {list(DESCRIPTORS)}")
    path = _resolve_upload(file_id)
    extractor = get_extractor(engine_name)
    try:
        result = await extractor.extract(str(path))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("extract failed engine=%s file_id=%s", engine_name, file_id)
        if settings.expose_error_details:
            detail = f"{engine_name} extraction failed: {e}"
        else:
            detail = f"{engine_name} extraction failed"
        raise HTTPException(500, detail) from e
    return JSONResponse(result.model_dump())


@app.get("/")
def root():
    return {
        "service": "pdf-extractor-lab",
        "version": "0.1.0",
        "endpoints": [
            "/health", "/engines", "/upload",
            "/files/{id}", "/files/{id} (DELETE)",
            "/pages/{id}/info", "/pages/{id}/{page}.png",
            "/extract",
        ],
    }
