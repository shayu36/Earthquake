"""ML service: spherical-coordinate K-Means clustering for earthquake data.

Key design decisions:
- Converts (lon, lat) to 3D unit-sphere (x, y, z) so K-Means
  respects the spherical geometry of the Earth.
- Does NOT use raw lat/lon — avoids date-line discontinuity
  and unequal degree-lengths at different latitudes.
- For spatial-only clustering: NO scaling on sphere coords
  (they are already on a unit sphere; scaling would distort distances).
- Depth and magnitude (when included) ARE scaled, with explicit
  alpha weight communicated in the output metadata.
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
        self.train_metadata: dict[str, Any] = {}

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
        depth_weight: float = 0.5,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Build feature matrix.

        - Sphere coords are NOT scaled (they are unit sphere; scaling distorts).
        - Depth is scaled and weighted by depth_weight (default 0.5).
        - Magnitude is scaled and given unit weight.

        Returns (X, feature_names, feature_meta).
        """
        sphere_df = self.add_spherical_coordinates(df)
        clean = sphere_df[["sphere_x", "sphere_y", "sphere_z"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        feature_names = ["sphere_x", "sphere_y", "sphere_z"]
        feature_meta: dict[str, Any] = {
            "sphere_coords": "unit sphere (x, y, z) — NOT scaled",
        }

        # Start with sphere coords as-is
        X = clean[["sphere_x", "sphere_y", "sphere_z"]].to_numpy()

        if include_depth:
            depth_clean = sphere_df.loc[clean.index, "depth"].replace(
                [np.inf, -np.inf], np.nan
            )
            valid = depth_clean.notna()
            X = X[valid]
            clean = clean.loc[clean.index[valid]]
            depth_values = depth_clean[valid].values.reshape(-1, 1)
            depth_scaled = StandardScaler().fit_transform(depth_values)
            X = np.column_stack([X, depth_weight * depth_scaled])
            feature_names.append("depth")
            feature_meta["depth"] = f"StandardScaler + weight={depth_weight}"

        if include_magnitude:
            mag_clean = sphere_df.loc[clean.index, "mag"].replace(
                [np.inf, -np.inf], np.nan
            )
            valid = mag_clean.notna()
            X = X[valid]
            clean = clean.loc[clean.index[valid]]
            mag_values = mag_clean[valid].values.reshape(-1, 1)
            mag_scaled = StandardScaler().fit_transform(mag_values)
            X = np.column_stack([X, mag_scaled])
            feature_names.append("mag")
            feature_meta["mag"] = "StandardScaler (unit weight)"

        return X, feature_names, feature_meta

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

        X, feature_names, feature_meta = self.build_features(
            df, include_depth=include_depth, include_magnitude=include_magnitude,
        )

        results = []
        for k in range(k_min, k_max + 1):
            model = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
            labels = model.fit_predict(X)
            eff_size = min(sample_size, len(X))
            score = silhouette_score(X, labels, sample_size=eff_size, random_state=random_state)
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

        X, feature_names, feature_meta = self.build_features(
            working_df, include_depth=include_depth, include_magnitude=include_magnitude,
        )

        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=random_state)
        labels = model.fit_predict(X)

        # Map labels back to the original dataframe using the clean index
        clean_df = working_df.loc[working_df.index[:len(X)]].copy()
        # We need to align. Let's re-derive the clean rows directly.
        # Simpler: re-do feature building inline to get the right rows.
        feature_names_check = ["sphere_x", "sphere_y", "sphere_z"]
        if include_depth:
            feature_names_check.append("depth")
        if include_magnitude:
            feature_names_check.append("mag")

        valid_mask = (
            working_df[feature_names_check]
            .replace([np.inf, -np.inf], np.nan)
            .notna().all(axis=1)
        )
        training_df = working_df.loc[valid_mask].copy()
        training_df["cluster"] = labels.astype(int)

        silhouette = silhouette_score(
            X, labels,
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
        self.scaler = None  # We don't use a single scaler anymore
        self.feature_names = feature_names
        self.clustered_df = training_df
        self.cluster_summary = cluster_summary

        # Store training metadata
        self.train_metadata = {
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "features": feature_names,
            "include_depth": include_depth,
            "include_magnitude": include_magnitude,
            "random_state": random_state,
            "silhouette_score": float(silhouette),
            "feature_meta": feature_meta,
            "note": "sphere_x/y/z are NOT scaled (unit sphere). Depth/mag are scaled if included.",
        }

        joblib.dump({
            "model": model, "feature_names": feature_names,
            "include_depth": include_depth, "include_magnitude": include_magnitude,
            "metadata": self.train_metadata,
        }, KMEANS_MODEL_PATH)
        # Save an empty scaler for backward compatibility
        joblib.dump({"note": "No global scaler used; see train_metadata"}, SCALER_PATH)

        return {
            "n_clusters": n_clusters,
            "training_record_count": int(len(training_df)),
            "excluded_record_count": int(len(working_df) - len(training_df)),
            "feature_names": feature_names,
            "silhouette_score": float(silhouette),
            "feature_meta": feature_meta,
            "cluster_summary": json.loads(
                cluster_summary.to_json(orient="records", date_format="iso")
            ),
            "metadata": self.train_metadata,
        }

    def get_clustered_records(self) -> list[dict[str, Any]]:
        if self.clustered_df is None:
            raise RuntimeError("Model not trained yet.")
        return json.loads(
            self.clustered_df.to_json(orient="records", date_format="iso")
        )
