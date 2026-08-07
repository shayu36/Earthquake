"""ML service: spherical-coordinate K-Means clustering for earthquake data.

Key design decisions:
- Converts (lon, lat) to 3D unit-sphere (x, y, z) so K-Means
  respects the spherical geometry of the Earth.
- Does NOT use raw lat/lon — avoids date-line discontinuity
  and unequal degree-lengths at different latitudes.
- Depth and magnitude are optional extra features (off by default).
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.app.config import KMEANS_MODEL_PATH, SCALER_PATH


class EarthquakeMLService:
    def __init__(self):
        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.clustered_df: pd.DataFrame | None = None
        self.cluster_summary: pd.DataFrame | None = None

    @staticmethod
    def add_spherical_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """Convert (latitude, longitude) → 3D unit-sphere (x, y, z)."""
        result = df.copy()
        lat_rad = np.radians(result["latitude"])
        lon_rad = np.radians(result["longitude"])
        result["sphere_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        result["sphere_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        result["sphere_z"] = np.sin(lat_rad)
        return result

    def build_features(
        self, df: pd.DataFrame, *,
        include_depth: bool = False,
        include_magnitude: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:
        feature_df = self.add_spherical_coordinates(df)
        feature_names = ["sphere_x", "sphere_y", "sphere_z"]
        if include_depth:
            feature_names.append("depth")
        if include_magnitude:
            feature_names.append("mag")
        clean = feature_df[feature_names].replace([np.inf, -np.inf], np.nan).dropna()
        return clean, feature_names

    def evaluate_k_values(
        self, df: pd.DataFrame, *,
        k_min: int = 2, k_max: int = 10,
        include_depth: bool = False, include_magnitude: bool = False,
        sample_size: int = 5000, random_state: int = 42,
    ) -> list[dict[str, Any]]:
        if k_min < 2:
            raise ValueError("k_min must be at least 2")
        if k_max <= k_min:
            raise ValueError("k_max must be greater than k_min")

        features, _ = self.build_features(
            df, include_depth=include_depth, include_magnitude=include_magnitude,
        )
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)

        results = []
        for k in range(k_min, k_max + 1):
            model = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
            labels = model.fit_predict(x_scaled)
            eff_size = min(sample_size, len(x_scaled))
            score = silhouette_score(x_scaled, labels, sample_size=eff_size, random_state=random_state)
            results.append({
                "k": k,
                "silhouette_score": float(score),
                "inertia": float(model.inertia_),
            })
        return results

    def train(
        self, df: pd.DataFrame, *,
        n_clusters: int,
        include_depth: bool = False, include_magnitude: bool = False,
        random_state: int = 42,
    ) -> dict[str, Any]:
        working_df = self.add_spherical_coordinates(df)
        feature_names = ["sphere_x", "sphere_y", "sphere_z"]
        if include_depth:
            feature_names.append("depth")
        if include_magnitude:
            feature_names.append("mag")

        valid_mask = (
            working_df[feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .notna().all(axis=1)
        )
        training_df = working_df.loc[valid_mask].copy()
        x = training_df[feature_names]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=random_state)
        labels = model.fit_predict(x_scaled)
        training_df["cluster"] = labels.astype(int)

        silhouette = silhouette_score(
            x_scaled, labels,
            sample_size=min(5000, len(training_df)),
            random_state=random_state,
        )

        cluster_summary = (
            training_df.groupby("cluster", as_index=False)
            .agg(
                event_count=("id", "count"),
                mean_latitude=("latitude", "mean"),
                mean_longitude=("longitude", "mean"),
                mean_depth=("depth", "mean"),
                median_depth=("depth", "median"),
                mean_magnitude=("mag", "mean"),
                max_magnitude=("mag", "max"),
                min_time=("time", "min"),
                max_time=("time", "max"),
            )
        )

        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.clustered_df = training_df
        self.cluster_summary = cluster_summary

        joblib.dump({
            "model": model, "feature_names": feature_names,
            "include_depth": include_depth, "include_magnitude": include_magnitude,
        }, KMEANS_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        return {
            "n_clusters": n_clusters,
            "training_record_count": int(len(training_df)),
            "excluded_record_count": int(len(working_df) - len(training_df)),
            "feature_names": feature_names,
            "silhouette_score": float(silhouette),
            "cluster_summary": json.loads(
                cluster_summary.to_json(orient="records", date_format="iso")
            ),
        }

    def get_clustered_records(self) -> list[dict[str, Any]]:
        if self.clustered_df is None:
            raise RuntimeError("Model not trained yet.")
        return json.loads(
            self.clustered_df.to_json(orient="records", date_format="iso")
        )
