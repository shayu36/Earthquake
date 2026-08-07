"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.catalog import router as catalog_router
from backend.app.routers.statistics import router as statistics_router
from backend.app.routers.machine_learning import router as ml_router
from backend.app.routers.export import router as export_router

app = FastAPI(
    title="Global Earthquake Analysis API",
    description="2024-2025 Global M4.5+ Earthquake Spatio-Temporal Analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(catalog_router)
app.include_router(statistics_router)
app.include_router(ml_router)
app.include_router(export_router)


@app.get("/")
def root():
    return {"service": "Global Earthquake Analysis API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
