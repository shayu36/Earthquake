"""Statistics service: monthly, annual, grid, coordinate window, top events."""
from __future__ import annotations

from typing import Any

import pandas as pd


class StatisticsService:
    def __init__(self, catalog_service):
        self.catalog = catalog_service

    def _df(self) -> pd.DataFrame:
        return self.catalog.require_data()

    def get_overview(self) -> dict[str, Any]:
        df = self._df()
        return {
            "total_events": int(len(df)),
            "time_min": df["time"].min().isoformat(),
            "time_max": df["time"].max().isoformat(),
            "mag_min": float(df["mag"].min()),
            "mag_max": float(df["mag"].max()),
            "mag_mean": float(df["mag"].mean()),
            "depth_min": float(df["depth"].min()),
            "depth_max": float(df["depth"].max()),
            "depth_mean": float(df["depth"].mean()),
            "magtype_distribution": df["magType"].value_counts().to_dict(),
        }

    def get_monthly(self) -> list[dict[str, Any]]:
        df = self._df()
        df = df.copy()
        df["year_month"] = df["time"].dt.to_period("M").astype(str)
        monthly = df.groupby("year_month").agg(
            count=("id", "count"),
            mean_mag=("mag", "mean"),
            max_mag=("mag", "max"),
            mean_depth=("depth", "mean"),
        ).reset_index()
        return monthly.to_dict(orient="records")

    def get_annual(self) -> list[dict[str, Any]]:
        df = self._df()
        df = df.copy()
        df["year"] = df["time"].dt.year
        annual = df.groupby("year").agg(
            count=("id", "count"),
            mean_mag=("mag", "mean"),
            max_mag=("mag", "max"),
            mean_depth=("depth", "mean"),
        ).reset_index()
        return annual.to_dict(orient="records")

    def get_grid(self, grid_size: float = 10.0) -> list[dict[str, Any]]:
        df = self._df()
        rows = []
        for lat_start in range(-90, 90, int(grid_size)):
            for lon_start in range(-180, 180, int(grid_size)):
                lat_end = lat_start + grid_size
                lon_end = lon_start + grid_size
                mask = (
                    df["latitude"].between(lat_start, lat_end)
                    & df["longitude"].between(lon_start, lon_end)
                )
                subset = df[mask]
                cnt = len(subset)
                if cnt > 0:
                    rows.append({
                        "lat_min": lat_start, "lat_max": lat_end,
                        "lon_min": lon_start, "lon_max": lon_end,
                        "count": cnt,
                        "mean_mag": round(subset["mag"].mean(), 3),
                        "max_mag": round(subset["mag"].max(), 2),
                        "mean_depth": round(subset["depth"].mean(), 1),
                    })
        return rows

    def get_coordinate_window(
        self, *, name: str,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
    ) -> dict[str, Any]:
        if lat_min >= lat_max:
            raise ValueError("lat_min must be less than lat_max")
        if lon_min >= lon_max:
            raise ValueError("lon_min must be less than lon_max")

        df = self._df()
        subset = df[
            df["latitude"].between(lat_min, lat_max, inclusive="both")
            & df["longitude"].between(lon_min, lon_max, inclusive="both")
        ].copy()

        if subset.empty:
            return {
                "name": name,
                "bounds": {"lat_min": lat_min, "lat_max": lat_max,
                           "lon_min": lon_min, "lon_max": lon_max},
                "event_count": 0,
                "mean_magnitude": None, "max_magnitude": None,
                "mean_depth": None, "max_depth": None,
                "shallow_count": 0, "intermediate_count": 0, "deep_count": 0,
            }

        return {
            "name": name,
            "bounds": {"lat_min": lat_min, "lat_max": lat_max,
                       "lon_min": lon_min, "lon_max": lon_max},
            "event_count": int(len(subset)),
            "mean_magnitude": float(subset["mag"].mean()),
            "max_magnitude": float(subset["mag"].max()),
            "median_magnitude": float(subset["mag"].median()),
            "mean_depth": float(subset["depth"].mean()),
            "median_depth": float(subset["depth"].median()),
            "max_depth": float(subset["depth"].max()),
            "shallow_count": int((subset["depth"] < 70).sum()),
            "intermediate_count": int(subset["depth"].between(70, 300, inclusive="left").sum()),
            "deep_count": int((subset["depth"] >= 300).sum()),
        }

    def get_top_events(self, n: int = 10) -> list[dict[str, Any]]:
        df = self._df()
        top = df.nlargest(n, "mag")[
            ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"]
        ].copy()
        top["time"] = top["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        return top.to_dict(orient="records")

    def get_monthly_energy(self) -> list[dict[str, Any]]:
        df = self._df()
        df = df.copy()
        df["year_month"] = df["time"].dt.to_period("M").astype(str)
        df["energy"] = 10 ** (1.5 * df["mag"])
        monthly = df.groupby("year_month").agg(
            count=("id", "count"),
            total_energy=("energy", "sum"),
            mean_mag=("mag", "mean"),
            max_mag=("mag", "max"),
        ).reset_index()
        total = monthly["total_energy"].sum()
        monthly["energy_pct"] = (monthly["total_energy"] / total * 100).round(2)
        return monthly.to_dict(orient="records")
