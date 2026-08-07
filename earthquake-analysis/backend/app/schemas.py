"""Pydantic request/response models."""
from pydantic import BaseModel, Field


class CoordinateWindowRequest(BaseModel):
    name: str = Field(default="Window")
    lat_min: float = Field(ge=-90, le=90)
    lat_max: float = Field(ge=-90, le=90)
    lon_min: float = Field(ge=-180, le=180)
    lon_max: float = Field(ge=-180, le=180)


class KMeansTrainRequest(BaseModel):
    n_clusters: int = Field(default=6, ge=2, le=20)
    include_depth: bool = False
    include_magnitude: bool = False
    random_state: int = 42
