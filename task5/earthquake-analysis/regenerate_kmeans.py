"""Regenerate K-Means results using updated ml_service (sphere coords NOT scaled)."""
import sys
from pathlib import Path

# Add the earthquake-analysis directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from backend.app.services.ml_service import EarthquakeMLService
from backend.app.config import (
    PROCESSED_CSV_PATH, CSV_OUTPUT_DIR, KMEANS_MODEL_PATH,
)

# Load data
df = pd.read_csv(PROCESSED_CSV_PATH)
df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")

print(f"Loaded {len(df)} records from {PROCESSED_CSV_PATH}")

ml = EarthquakeMLService()

# Train with depth (matching what the Streamlit app does by default)
# K=6, include_depth=True (as was the original setting)
result = ml.train(
    df,
    n_clusters=6,
    include_depth=True,
    include_magnitude=False,
    random_state=42,
)

print(f"Trained K-Means (K=6, include_depth=True)")
print(f"  Silhouette: {result['silhouette_score']:.4f}")
print(f"  Training records: {result['training_record_count']}")
print(f"  Excluded: {result['excluded_record_count']}")
print(f"  Features: {result['feature_names']}")

# Export cluster summary
summary_df = ml.cluster_summary
summary_df.to_csv(CSV_OUTPUT_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig")
print(f"  -> Saved: {CSV_OUTPUT_DIR / 'cluster_summary.csv'}")

# Export clustered records
clustered = ml.clustered_df.copy()
clustered.to_csv(CSV_OUTPUT_DIR / "clustered_events.csv", index=False, encoding="utf-8-sig")
print(f"  -> Saved: {CSV_OUTPUT_DIR / 'clustered_events.csv'}")

# Export metadata
import json
meta_path = CSV_OUTPUT_DIR / "kmeans_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(result["metadata"], f, ensure_ascii=False, indent=2)
print(f"  -> Saved: {meta_path}")

print("\nDone. HTML maps will be regenerated when Streamlit app runs.")
