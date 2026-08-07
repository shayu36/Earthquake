"""Tests for statistical computations."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task1.task1_data_validation import read_and_prepare

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_CSV = BASE_DIR / "USGS_2024_2025_M4.5plus_earthquakes.csv"


@pytest.fixture
def df():
    if not TEST_CSV.exists():
        pytest.skip("Test CSV not found")
    return read_and_prepare(TEST_CSV)


def test_monthly_count_sum(df):
    """Sum of monthly counts should equal total records."""
    df = df.copy()
    df["year_month"] = df["time"].dt.to_period("M")
    monthly = df.groupby("year_month").size()
    assert len(monthly) == 24
    assert monthly.sum() == 14953


def test_grid_count_sum(df):
    """Sum of 10x10 grid counts should equal total records."""
    import numpy as np
    lat_bins = np.arange(-90, 91, 10)
    lon_bins = np.arange(-180, 181, 10)
    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            lat_min, lat_max = lat_bins[i], lat_bins[i+1]
            lon_min, lon_max = lon_bins[j], lon_bins[j+1]
            mask = (df["latitude"] >= lat_min) & (df["latitude"] < lat_max) & \
                   (df["longitude"] >= lon_min) & (df["longitude"] < lon_max)
            # just checking structure — no need to accumulate here
    # Use numpy histogram2d which handles left-closed, right-open
    hist2d, _, _ = np.histogram2d(
        df["latitude"], df["longitude"], bins=[lat_bins, lon_bins]
    )
    assert hist2d.sum() == 14953


def test_mag_bins_sum(df):
    """Sum of magnitude bins should equal total records."""
    mag_bins = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0]
    mag_labels = ["4.5-5.0", "5.0-5.5", "5.5-6.0", "6.0-6.5",
                  "6.5-7.0", "7.0-7.5", "7.5-8.0", "8.0-9.0"]
    df = df.copy()
    df["mag_bin"] = pd.cut(df["mag"], bins=mag_bins, labels=mag_labels, right=False)
    assert df["mag_bin"].value_counts().sum() == 14953


def test_time_range_ok(df):
    """Time range verification should pass."""
    assert df["time"].min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert df["time"].max() <= pd.Timestamp("2025-12-31 23:59:59.999", tz="UTC")
    assert df["time"].is_monotonic_increasing
