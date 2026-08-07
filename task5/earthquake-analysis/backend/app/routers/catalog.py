"""Catalog router: upload, verification, filtered events."""
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.app.config import PROCESSED_CSV_PATH, RAW_CSV_PATH
from backend.app.services.catalog_service import CatalogService

router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])

catalog_service = CatalogService(PROCESSED_CSV_PATH)


@router.post("/upload")
async def upload_catalog(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50 MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    # Step 1: Validate the uploaded CSV FIRST (don't overwrite raw yet)
    try:
        verification = catalog_service.load_uploaded_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Step 2: Only write to disk if verification passed
    if not verification.get("verification_passed", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV verification failed — data not saved.",
                "verification": verification,
            },
        )

    RAW_CSV_PATH.write_bytes(content)
    catalog_service.save_processed()

    return {
        "message": "CSV uploaded and processed successfully.",
        "filename": filename,
        "verification": verification,
    }


@router.get("/verification")
def get_verification():
    try:
        return catalog_service.get_verification()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/events")
def get_events(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    min_mag: float = Query(default=4.5),
    max_mag: float | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    mag_type: str | None = None,
    place_keyword: str | None = None,
    lat_min: float = Query(default=-90, ge=-90, le=90),
    lat_max: float = Query(default=90, ge=-90, le=90),
    lon_min: float = Query(default=-180, ge=-180, le=180),
    lon_max: float = Query(default=180, ge=-180, le=180),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=20000),
):
    try:
        df = catalog_service.require_data()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = df.copy()

    tz = result["time"].dt.tz
    if start_time is not None:
        s = start_time if start_time.tzinfo else start_time.replace(tzinfo=tz)
        result = result[result["time"] >= s]
    if end_time is not None:
        e = end_time if end_time.tzinfo else end_time.replace(tzinfo=tz)
        result = result[result["time"] <= e]

    result = result[result["mag"] >= min_mag]
    if max_mag is not None:
        result = result[result["mag"] <= max_mag]
    if min_depth is not None:
        result = result[result["depth"] >= min_depth]
    if max_depth is not None:
        result = result[result["depth"] <= max_depth]
    if mag_type:
        result = result[result["magType"] == mag_type]
    if place_keyword:
        result = result[result["place"].fillna("").str.contains(place_keyword, case=False, regex=False)]

    # Left-closed, right-open for lat/lon bounds to avoid double counting
    result = result[
        (result["latitude"] >= lat_min) & (result["latitude"] < lat_max)
        & (result["longitude"] >= lon_min) & (result["longitude"] < lon_max)
    ]

    total = len(result)
    offset = (page - 1) * page_size
    page_df = result.iloc[offset: offset + page_size]

    return {
        "total": int(total),
        "page": page,
        "page_size": page_size,
        "items": catalog_service.to_records(page_df),
    }
