"""Export router: CSV, PNG, HTML file downloads."""
from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.config import CSV_OUTPUT_DIR, PNG_OUTPUT_DIR, HTML_OUTPUT_DIR

router = APIRouter(prefix="/api/v1/export", tags=["export"])


@router.get("/csv/{filename}")
def download_csv(filename: str):
    path = CSV_OUTPUT_DIR / filename
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    return FileResponse(path, media_type="text/csv", filename=filename)


@router.get("/png/{filename}")
def download_png(filename: str):
    path = PNG_OUTPUT_DIR / filename
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    return FileResponse(path, media_type="image/png", filename=filename)


@router.get("/html/{filename}")
def download_html(filename: str):
    path = HTML_OUTPUT_DIR / filename
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    return FileResponse(path, media_type="text/html", filename=filename)
