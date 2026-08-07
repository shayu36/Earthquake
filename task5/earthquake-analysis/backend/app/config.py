"""Project-wide path configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
PNG_OUTPUT_DIR = OUTPUT_DIR / "png"
HTML_OUTPUT_DIR = OUTPUT_DIR / "html"

RAW_CSV_PATH = RAW_DIR / "earthquakes_raw.csv"
PROCESSED_CSV_PATH = PROCESSED_DIR / "earthquakes_processed.csv"

KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

for directory in [
    RAW_DIR, PROCESSED_DIR, MODEL_DIR,
    CSV_OUTPUT_DIR, PNG_OUTPUT_DIR, HTML_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
