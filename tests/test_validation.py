"""Tests for data validation."""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task1_data_validation import (
    read_and_prepare,
    calculate_sha256,
    validate_ranges,
    validate_uniqueness,
)

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_CSV = BASE_DIR / "USGS_2024_2025_M4.5plus_earthquakes.csv"


@pytest.fixture
def df():
    """Load test data (skip if CSV not found)."""
    if not TEST_CSV.exists():
        pytest.skip("Test CSV not found")
    return read_and_prepare(TEST_CSV)


def test_record_count(df):
    """Total records should be 14,953."""
    assert len(df) == 14953


def test_time_range(df):
    """Time should be within 2024-01-01 to 2025-12-31."""
    assert df["time"].min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert df["time"].max() <= pd.Timestamp("2025-12-31 23:59:59.999", tz="UTC")


def test_no_duplicate_ids(df):
    """All IDs should be unique."""
    assert df["id"].duplicated().sum() == 0


def test_no_duplicate_rows(df):
    """No fully duplicate rows."""
    assert df.duplicated().sum() == 0


def test_latitude_range(df):
    """Latitude should be in [-90, 90]."""
    assert df["latitude"].between(-90, 90).all()


def test_longitude_range(df):
    """Longitude should be in [-180, 180]."""
    assert df["longitude"].between(-180, 180).all()


def test_magnitude_threshold(df):
    """All magnitudes should be >= 4.5."""
    assert (df["mag"] >= 4.5).all()


def test_depth_non_negative(df):
    """Depth should be >= 0."""
    assert (df["depth"] >= 0).all()


def test_type_is_earthquake(df):
    """All events should be type 'earthquake'."""
    assert (df["type"] == "earthquake").all()


def test_time_ascending(df):
    """Data should be sorted by time ascending."""
    assert df["time"].is_monotonic_increasing


def test_sha256_dynamic():
    """SHA-256 should be computed dynamically from file."""
    if not TEST_CSV.exists():
        pytest.skip("Test CSV not found")
    result = calculate_sha256(TEST_CSV)
    assert len(result) == 64
    assert result == result.upper()


def test_validation_passes(df):
    """Core validation should find no issues for the standard dataset."""
    import io
    f = io.StringIO()
    issues = validate_ranges(df, f)
    issues += validate_uniqueness(df, f)
    assert len(issues) == 0
