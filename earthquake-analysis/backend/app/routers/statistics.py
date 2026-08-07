"""Statistics router: overview, monthly, annual, grid, window, top events, energy."""
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import CoordinateWindowRequest
from backend.app.routers.catalog import catalog_service
from backend.app.services.statistics_service import StatisticsService

router = APIRouter(prefix="/api/v1/statistics", tags=["statistics"])

stats_service = StatisticsService(catalog_service)


def _require():
    try:
        catalog_service.require_data()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/overview")
def get_overview():
    _require()
    return stats_service.get_overview()


@router.get("/monthly")
def get_monthly():
    _require()
    return stats_service.get_monthly()


@router.get("/annual")
def get_annual():
    _require()
    return stats_service.get_annual()


@router.get("/grid")
def get_grid(grid_size: float = Query(default=10.0, ge=1.0, le=90.0)):
    _require()
    return stats_service.get_grid(grid_size)


@router.post("/window")
def coordinate_window_statistics(request: CoordinateWindowRequest):
    _require()
    try:
        return stats_service.get_coordinate_window(
            name=request.name,
            lat_min=request.lat_min, lat_max=request.lat_max,
            lon_min=request.lon_min, lon_max=request.lon_max,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/top-events")
def get_top_events(n: int = Query(default=10, ge=1, le=100)):
    _require()
    return stats_service.get_top_events(n)


@router.get("/monthly-energy")
def get_monthly_energy():
    _require()
    return stats_service.get_monthly_energy()
