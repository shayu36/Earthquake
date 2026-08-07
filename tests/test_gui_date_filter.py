"""Tests for GUI date filter logic (extracted for testability)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task5_gui_browser import filter_by_date


@pytest.fixture
def sample_df():
    """Create a small sample dataframe for date filter testing."""
    data = {
        "time": pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-06-15T12:00:00Z",
            "2025-12-31T00:00:00Z",
            "2025-12-31T12:00:00Z",
            "2025-12-31T23:59:59Z",
        ]),  # "Z" suffix already → UTC; do NOT call .tz_localize() again
        "mag": [5.0, 5.5, 6.0, 6.5, 7.0],
        "id": ["a", "b", "c", "d", "e"],
    }
    return pd.DataFrame(data)


def test_end_date_includes_whole_day(sample_df):
    """End date should include all events on that day."""
    result = filter_by_date(sample_df, "2025-12-31", "2025-12-31")
    # Should include all 3 events on 2025-12-31
    assert len(result) == 3
    assert set(result["id"]) == {"c", "d", "e"}


def test_start_date_includes_whole_day(sample_df):
    """Start date should include events from 00:00:00 onwards."""
    result = filter_by_date(sample_df, "2025-12-31", "2025-12-31")
    # The event at 00:00:00 should be included
    assert "c" in set(result["id"])


def test_single_day_range(sample_df):
    """Filtering a single day should work."""
    result = filter_by_date(sample_df, "2024-06-15", "2024-06-15")
    assert len(result) == 1
    assert result.iloc[0]["id"] == "b"


def test_multi_day_range(sample_df):
    """Filtering across multiple days should include boundary days fully."""
    result = filter_by_date(sample_df, "2024-01-01", "2024-06-15")
    assert len(result) == 2
    assert set(result["id"]) == {"a", "b"}


def test_filter_before_data(sample_df):
    """Filtering before all data should return empty."""
    result = filter_by_date(sample_df, "2020-01-01", "2020-12-31")
    assert len(result) == 0


def test_filter_after_data(sample_df):
    """Filtering after all data should return empty."""
    result = filter_by_date(sample_df, "2026-01-01", "2026-12-31")
    assert len(result) == 0
