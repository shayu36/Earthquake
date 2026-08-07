"""Earthquake catalog service: read, validate, filter CSV data."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "time", "latitude", "longitude", "depth", "mag", "magType",
    "nst", "gap", "dmin", "rms", "net", "id", "updated", "place",
    "type", "horizontalError", "depthError", "magError", "magNst",
    "status", "locationSource", "magSource",
]

QUALITY_COLUMNS = [
    "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError", "magNst",
]


class CatalogService:
    def __init__(self, processed_path: Path):
        self.processed_path = processed_path
        self.df: pd.DataFrame | None = None
        self.verification: dict[str, Any] = {}

        if processed_path.exists():
            self.df = self._read_dataframe(processed_path)
            self.verification = self._verify_dataframe(self.df)

    @staticmethod
    def _read_dataframe(path_or_buffer: Any) -> pd.DataFrame:
        df = pd.read_csv(path_or_buffer)

        missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {', '.join(missing_columns)}")

        # time = earthquake origin time; updated = catalog update time
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["updated"] = pd.to_datetime(df["updated"], utc=True, errors="coerce")

        numeric_columns = [
            "latitude", "longitude", "depth", "mag",
            "nst", "gap", "dmin", "rms",
            "horizontalError", "depthError", "magError", "magNst",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    def load_uploaded_bytes(self, content: bytes) -> dict[str, Any]:
        """Validate uploaded CSV bytes BEFORE overwriting raw data.

        Returns verification dict. Caller should only write to disk
        after this returns successfully.
        """
        df = self._read_dataframe(io.BytesIO(content))
        df = df.sort_values("time", ascending=True).reset_index(drop=True)
        verification = self._verify_dataframe(df)
        # Only update internal state after successful validation
        self.verification = verification
        self.df = df
        return verification

    def save_processed(self) -> None:
        """Persist current dataframe to processed CSV path."""
        if self.df is not None:
            self.df.to_csv(self.processed_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _verify_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        expected_start = pd.Timestamp("2024-01-01T00:00:00Z")
        expected_end = pd.Timestamp("2025-12-31T23:59:59.999Z")

        missing_quality = {
            column: int(df[column].isna().sum())
            for column in QUALITY_COLUMNS
        }

        result = {
            "record_count": int(len(df)),
            "expected_record_count": 14953,
            "record_count_matches": bool(len(df) == 14953),

            "time_missing_count": int(df["time"].isna().sum()),
            "updated_missing_count": int(df["updated"].isna().sum()),

            "time_min": df["time"].min().isoformat() if df["time"].notna().any() else None,
            "time_max": df["time"].max().isoformat() if df["time"].notna().any() else None,

            "time_range_valid": bool(
                df["time"].dropna().between(expected_start, expected_end, inclusive="both").all()
            ),

            "id_missing_count": int(df["id"].isna().sum()),
            "id_duplicate_count": int(df["id"].duplicated().sum()),
            "full_row_duplicate_count": int(df.duplicated().sum()),

            "latitude_invalid_count": int((~df["latitude"].between(-90, 90)).sum()),
            "longitude_invalid_count": int((~df["longitude"].between(-180, 180)).sum()),
            "magnitude_below_4_5_count": int((df["mag"] < 4.5).sum()),
            "non_earthquake_type_count": int((df["type"] != "earthquake").sum()),
            "negative_depth_count": int((df["depth"] < 0).sum()),

            "time_ascending": bool(df["time"].is_monotonic_increasing),

            "missing_quality_fields": missing_quality,
        }

        # Full verification: ALL checks must pass
        result["verification_passed"] = all([
            result["record_count_matches"],
            result["time_missing_count"] == 0,
            result["updated_missing_count"] == 0,
            result["time_range_valid"],
            result["id_missing_count"] == 0,
            result["id_duplicate_count"] == 0,
            result["full_row_duplicate_count"] == 0,
            result["latitude_invalid_count"] == 0,
            result["longitude_invalid_count"] == 0,
            result["magnitude_below_4_5_count"] == 0,
            result["non_earthquake_type_count"] == 0,
            result["negative_depth_count"] == 0,
            result["time_ascending"],
        ])

        return result

    def require_data(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("No earthquake CSV data loaded. Upload first.")
        return self.df

    def get_verification(self) -> dict[str, Any]:
        self.require_data()
        return self.verification

    @staticmethod
    def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(df.to_json(orient="records", date_format="iso"))
