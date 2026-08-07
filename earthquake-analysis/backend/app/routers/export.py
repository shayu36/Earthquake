"""Export router: CSV, PNG, HTML file downloads with path-traversal protection."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.config import CSV_OUTPUT_DIR, PNG_OUTPUT_DIR, HTML_OUTPUT_DIR

router = APIRouter(prefix="/api/v1/export", tags=["export"])


def safe_output_path(base_dir: Path, filename: str) -> Path:
    """Resolve a safe path within base_dir, rejecting path traversal attempts.

    Raises HTTPException(400) for suspicious filenames.
    Raises HTTPException(404) if the resolved file does not exist.
    """
    # Reject filenames that contain path separators or are not plain names
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = (base_dir / filename).resolve()
    base = base_dir.resolve()

    # Ensure the resolved path is actually inside base_dir
    if base not in path.parents and path != base:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return path


@router.get("/csv/{filename}")
def download_csv(filename: str):
    path = safe_output_path(CSV_OUTPUT_DIR, filename)
    return FileResponse(path, media_type="text/csv", filename=filename)


@router.get("/png/{filename}")
def download_png(filename: str):
    path = safe_output_path(PNG_OUTPUT_DIR, filename)
    return FileResponse(path, media_type="image/png", filename=filename)


@router.get("/html/{filename}")
def download_html(filename: str):
    path = safe_output_path(HTML_OUTPUT_DIR, filename)
    return FileResponse(path, media_type="text/html", filename=filename)
