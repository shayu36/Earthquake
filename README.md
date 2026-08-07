# Global M4.5+ Earthquake Spatio-Temporal Analysis

USGS ANSS Comprehensive Earthquake Catalog (ComCat) — 2024-2025, M ≥ 4.5.

## Project Structure

```
.
├── USGS_2024_2025_M4.5plus_earthquakes.csv   # Raw data (14,953 records)
├── 数据说明.txt                                # Data description
│
├── task1/task1_data_validation.py              # Task 1: Data reading & validation
├── task2/task2_temporal_analysis.py            # Task 2: Temporal statistics
├── task3/task3_spatial_analysis.py             # Task 3: Spatial distribution
├── task4/task4_energy_analysis.py              # Task 4: Top events & energy
├── task5/task5_gui_browser.py                  # Task 5: Tkinter GUI browser
├── task6/task6_ml_analysis.py                  # Task 6: DBSCAN + PCA
│
├── task5/earthquake-analysis/                 # Web application (FastAPI + Streamlit)
│   ├── backend/                               # FastAPI backend
│   │   └── app/
│   │       ├── main.py                        # API entry point
│   │       ├── config.py                      # Path configuration
│   │       ├── schemas.py                     # Pydantic models
│   │       ├── routers/                       # API route handlers
│   │       │   ├── catalog.py                 # Upload & query
│   │       │   ├── statistics.py              # Statistical endpoints
│   │       │   ├── machine_learning.py        # K-Means clustering
│   │       │   └── export.py                  # File downloads
│   │       └── services/                      # Business logic
│   │           ├── catalog_service.py         # CSV validation
│   │           ├── statistics_service.py      # Stats computation
│   │           └── ml_service.py              # ML pipeline
│   └── frontend/
│       └── app.py                             # Streamlit dashboard
│
├── tests/                                     # Unit tests
│   ├── test_validation.py
│   ├── test_statistics.py
│   └── test_gui_date_filter.py
│
└── requirements.txt                           # Root Python dependencies
```
```

## Quick Start

### 1. Install Dependencies

Web application dependencies（root scripts only need pandas/numpy/matplotlib/scikit-learn）:

```bash
pip install -r earthquake-analysis/requirements.txt
```

### 2. Run Individual Tasks

All scripts accept `--input` and `--output-dir` arguments (defaults are relative paths):

```bash
# Task 1: Data validation
python task1/task1_data_validation.py

# Task 2: Temporal analysis
python task2/task2_temporal_analysis.py

# Task 3: Spatial analysis
python task3/task3_spatial_analysis.py

# Task 4: Energy analysis
python task4/task4_energy_analysis.py

# Task 5: GUI browser (interactive)
python task5/task5_gui_browser.py

# Task 6: ML analysis (DBSCAN + PCA)
python task6/task6_ml_analysis.py
```

With custom paths:

```bash
python task1/task1_data_validation.py --input my_data.csv --output-dir ./task1
```

### 3. Run Web Application

#### One-click launcher (recommended)

```bash
cd task5/earthquake-analysis
python start_gui.py
```

Or double-click `task5/earthquake-analysis/启动地震分析系统.bat` on Windows.

This automatically starts FastAPI backend + Streamlit frontend and opens the browser.
Press `Ctrl+C` to stop both services.

#### Manual start (two terminals)

```bash
# Terminal 1: Start backend
cd task5/earthquake-analysis
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start frontend
cd task5/earthquake-analysis
streamlit run frontend/app.py --server.port 8501
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Data Source

- **Provider**: United States Geological Survey (USGS)
- **Catalog**: ANSS Comprehensive Earthquake Catalog (ComCat)
- **Time range**: 2024-01-01 to 2025-12-31
- **Magnitude threshold**: M ≥ 4.5
- **Records**: 14,953

## Key Results

| Metric | Value |
|--------|-------|
| Total records | 14,953 |
| Time range | 2024-01-01 ~ 2025-12-31 |
| Magnitude range | 4.5 ~ 8.8 |
| Most active month | 2025-07 (1,321 events) |
| Largest event | M8.8 — Kamchatka Peninsula, Russia (2025-07-29) |
| Most common magType | mb (79.5%) |

## Method Notes

- **DBSCAN**: Uses Haversine (great-circle) distance for correct global spatial clustering
- **K-Means**: Converts (lat, lon) to 3D unit-sphere coordinates to handle spherical geometry
- **Grid statistics**: Left-closed, right-open intervals to avoid double-counting on boundaries
- **Energy index**: E_rel = 10^(1.5×(M−4.5)), M=4.5 → 1.0 (proxy index, not joules)

## License

Educational project.
