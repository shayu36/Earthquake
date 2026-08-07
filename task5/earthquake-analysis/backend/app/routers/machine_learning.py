"""Machine Learning router: K-Means evaluate, train, results."""
from fastapi import APIRouter, HTTPException, Query

from backend.app.routers.catalog import catalog_service
from backend.app.schemas import KMeansTrainRequest
from backend.app.services.ml_service import EarthquakeMLService

router = APIRouter(prefix="/api/v1/ml", tags=["machine-learning"])

ml_service = EarthquakeMLService()


def _require():
    try:
        catalog_service.require_data()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/kmeans/evaluate")
def evaluate_kmeans(
    k_min: int = Query(default=2, ge=2, le=19),
    k_max: int = Query(default=10, ge=3, le=20),
    include_depth: bool = False,
    include_magnitude: bool = False,
):
    _require()
    try:
        results = ml_service.evaluate_k_values(
            catalog_service.require_data(),
            k_min=k_min, k_max=k_max,
            include_depth=include_depth, include_magnitude=include_magnitude,
        )
        best = max(results, key=lambda item: item["silhouette_score"])
        return {
            "results": results,
            "recommended_k": best["k"],
            "best_silhouette_score": best["silhouette_score"],
        }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/kmeans/train")
def train_kmeans(request: KMeansTrainRequest):
    _require()
    try:
        return ml_service.train(
            catalog_service.require_data(),
            n_clusters=request.n_clusters,
            include_depth=request.include_depth,
            include_magnitude=request.include_magnitude,
            random_state=request.random_state,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/kmeans/results")
def get_kmeans_results():
    try:
        records = ml_service.get_clustered_records()
        return {"total": len(records), "items": records}
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
