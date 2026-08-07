#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3: Global Spatial Distribution Analysis
=============================================
1. Global scatter map of earthquake epicenters
2. 10°×10° grid heatmap
3. Reusable coordinate window statistics function
4. Compare three fixed 10°×10° regions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, json
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from collections import OrderedDict

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
DATA_DIR = Path(r"D:\Users\lenovo\Desktop\题目2_USGS地震数据")
INPUT_CSV = DATA_DIR / "task1_processed_data.csv"
OUTPUT_DIR = DATA_DIR

# ── Read data ──
df = pd.read_csv(INPUT_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")

print(f"Loaded {len(df)} records")

# ================================================================
# (1) Global scatter map
# ================================================================
print("\n=== (1) Global Scatter Map ===")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- (a) Global epicenter scatter ---
ax = axes[0, 0]
# Draw simplified continental outlines using a manual approach:
# shade background as light blue (ocean), draw grid

# Plot all epicenters
sc = ax.scatter(df["longitude"], df["latitude"],
                c=df["mag"], cmap="plasma", s=1.5, alpha=0.5,
                norm=Normalize(vmin=4.5, vmax=8.0))
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_xlabel("Longitude (°)", fontsize=11)
ax.set_ylabel("Latitude (°)", fontsize=11)
ax.set_title("Global Earthquake Epicenters (2024-2025, M>=4.5)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Magnitude", fontsize=10)

# --- (b) Longitude histogram ---
ax = axes[0, 1]
ax.hist(df["longitude"], bins=72, color="#4A90D9", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Longitude (°)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Longitude Distribution", fontsize=12, fontweight="bold")
ax.set_xlim(-180, 180)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# --- (c) Latitude histogram ---
ax = axes[1, 0]
ax.hist(df["latitude"], bins=36, color="#50B86A", alpha=0.8, edgecolor="white", linewidth=0.3,
        orientation="horizontal")
ax.set_ylabel("Latitude (°)", fontsize=11)
ax.set_xlabel("Count", fontsize=11)
ax.set_title("Latitude Distribution", fontsize=12, fontweight="bold")
ax.set_ylim(-90, 90)
ax.grid(axis="x", alpha=0.3, linestyle="--")

# --- (d) Depth-colored scatter (sub-sampled for clarity) ---
ax = axes[1, 1]
sample = df.sample(min(8000, len(df)), random_state=42)
sc2 = ax.scatter(sample["longitude"], sample["latitude"],
                 c=sample["depth"], cmap="viridis_r", s=2, alpha=0.6,
                 norm=LogNorm(vmin=1, vmax=700))
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_xlabel("Longitude (°)", fontsize=11)
ax.set_ylabel("Latitude (°)", fontsize=11)
ax.set_title("Earthquakes Colored by Depth", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.8)
cbar2.set_label("Depth (km)", fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "task3_global_spatial_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved: task3_global_spatial_map.png")


# ================================================================
# (2) 10°×10° grid heatmap
# ================================================================
print("\n=== (2) Spatial Grid Heatmap (10°×10°) ===")

lat_bins = np.arange(-90, 91, 10)
lon_bins = np.arange(-180, 181, 10)

# 2D histogram: shape (n_lat_bins, n_lon_bins)
hist2d, lat_edges, lon_edges = np.histogram2d(
    df["latitude"], df["longitude"],
    bins=[lat_bins, lon_bins]
)
# For pcolormesh with 'flat' shading: C shape = (n_lat, n_lon), X edges = n_lon+1, Y edges = n_lat+1
# hist2d already has shape (18, 36) which matches lat_edges(19) x lon_edges(37)

# Create grid statistics DataFrame
grid_data = []
for i in range(len(lat_bins) - 1):
    for j in range(len(lon_bins) - 1):
        lat_min, lat_max = lat_bins[i], lat_bins[i+1]
        lon_min, lon_max = lon_bins[j], lon_bins[j+1]
        mask = (df["latitude"]  >= lat_min) & (df["latitude"]  < lat_max) & \
               (df["longitude"] >= lon_min) & (df["longitude"] < lon_max)
        count = mask.sum()
        if count > 0:
            sub = df[mask]
            grid_data.append({
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max,
                "count": count,
                "mean_mag": round(sub["mag"].mean(), 3),
                "max_mag":   round(sub["mag"].max(), 2),
                "mean_depth": round(sub["depth"].mean(), 1),
                "max_depth":  round(sub["depth"].max(), 1),
            })

grid_df = pd.DataFrame(grid_data)
grid_df.to_csv(OUTPUT_DIR / "task3_grid_10x10.csv", index=False, encoding="utf-8-sig")
print(f"  Non-empty grid cells: {len(grid_df)} / {18 * 36}")

# ── Heatmap plot ──
fig, ax = plt.subplots(figsize=(16, 8))

# Use pcolormesh for the heatmap
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

mesh = ax.pcolormesh(lon_edges, lat_edges, hist2d,
                     cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=max(hist2d.max(), 10)),
                     edgecolors="lightgray", linewidth=0.3)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_xlabel("Longitude (°)", fontsize=13)
ax.set_ylabel("Latitude (°)", fontsize=13)
ax.set_title("10°×10° Grid Earthquake Density Heatmap", fontsize=15, fontweight="bold")
ax.grid(True, alpha=0.15, linestyle="--")

cbar = plt.colorbar(mesh, ax=ax, shrink=0.75)
cbar.set_label("Earthquake Count (log scale)", fontsize=11)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "task3_grid_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved: task3_grid_heatmap.png")

# Top 10 grid cells
print("\n  Top 10 most active 10°×10° grid cells:")
top10 = grid_df.nlargest(10, "count")
for _, row in top10.iterrows():
    print(f"    Lon [{row['lon_min']:6.0f}, {row['lon_max']:6.0f}]  "
          f"Lat [{row['lat_min']:6.0f}, {row['lat_max']:6.0f}]  "
          f"Count: {int(row['count']):5d}  "
          f"MaxMag: {row['max_mag']:.1f}")


# ================================================================
# (3) Reusable coordinate window statistics function
# ================================================================
print("\n=== (3) Reusable Coordinate Window Function ===")


def window_stats(df, lon_range, lat_range):
    """
    Compute statistics for earthquakes within a coordinate window.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog with columns: latitude, longitude, depth, mag, time, place
    lon_range : tuple (lon_min, lon_max)
        Longitude bounds in degrees.
    lat_range : tuple (lat_min, lat_max)
        Latitude bounds in degrees.

    Returns
    -------
    dict
        Statistics including count, mean/max mag, mean/max depth,
        time range, magnitude distribution, depth distribution.
    """
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range

    mask = (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) & \
           (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max)
    sub = df[mask]

    if len(sub) == 0:
        return {
            "lon_range": lon_range,
            "lat_range": lat_range,
            "count": 0,
            "message": "No earthquakes in this window."
        }

    # Magnitude distribution
    mag_bins = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0]
    mag_labels = ["4.5-5.0", "5.0-5.5", "5.5-6.0", "6.0-6.5", "6.5-7.0", "7.0-7.5", "7.5-8.0", "8.0-9.0"]
    mag_dist = pd.cut(sub["mag"], bins=mag_bins, labels=mag_labels, right=False).value_counts().to_dict()

    # Depth distribution
    depth_bins = [0, 50, 100, 300, 1000]
    depth_labels = ["0-50km", "50-100km", "100-300km", ">300km"]
    depth_dist = pd.cut(sub["depth"], bins=depth_bins, labels=depth_labels, right=False).value_counts().to_dict()

    return {
        "lon_range": [lon_min, lon_max],
        "lat_range": [lat_min, lat_max],
        "count": int(len(sub)),
        "mean_mag": round(sub["mag"].mean(), 3),
        "max_mag": round(sub["mag"].max(), 2),
        "mean_depth_km": round(sub["depth"].mean(), 1),
        "max_depth_km": round(sub["depth"].max(), 1),
        "time_start": str(sub["time"].min()),
        "time_end": str(sub["time"].max()),
        "mag_distribution": {str(k): int(v) for k, v in mag_dist.items()},
        "depth_distribution": {str(k): int(v) for k, v in depth_dist.items()},
    }


# Quick test
test_result = window_stats(df, (120, 150), (30, 45))  # Japan region
print(f"\n  Test: Japan region (120-150°E, 30-45°N)")
print(f"    Count: {test_result['count']},  Mean mag: {test_result['mean_mag']},  Max mag: {test_result['max_mag']}")
print(f"    Mean depth: {test_result['mean_depth_km']} km")

print("\n  Function 'window_stats(df, lon_range, lat_range)' defined and tested.")


# ================================================================
# (4) Compare three fixed 10°×10° regions
# ================================================================
print("\n=== (4) Compare Three 10°×10° Regions ===")

# Choose three seismically distinct regions
regions = OrderedDict([
    ("Region A: Japan Trench",     ((130, 140), (30, 40))),
    ("Region B: Pacific Ring - S. America", ((-80, -70), (-35, -25))),
    ("Region C: Southeast Asia - Indonesia", ((95, 105), (-10, 0))),
])

region_results = {}
for name, (lon_r, lat_r) in regions.items():
    res = window_stats(df, lon_r, lat_r)
    region_results[name] = res
    print(f"\n  {name}")
    print(f"    Window: Lon {lon_r}, Lat {lat_r}")
    print(f"    Count: {res['count']}")
    print(f"    Mean mag: {res['mean_mag']},  Max mag: {res['max_mag']}")
    print(f"    Mean depth: {res['mean_depth_km']} km,  Max depth: {res['max_depth_km']} km")
    print(f"    Time range: {res['time_start'][:10]} ~ {res['time_end'][:10]}")

# ── Plot: comparison charts ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
region_names = list(regions.keys())

for idx, name in enumerate(region_names):
    res = region_results[name]
    lon_r = res["lon_range"]
    lat_r = res["lat_range"]

    # Scatter of this region on world map context
    ax_map = axes[0, idx]
    # Background: all events (faint)
    ax_map.scatter(df["longitude"], df["latitude"], c="#cccccc", s=0.3, alpha=0.4)
    # Region rectangle
    rect = Rectangle((lon_r[0], lat_r[0]), lon_r[1] - lon_r[0], lat_r[1] - lat_r[0],
                      linewidth=2, edgecolor="red", facecolor="none", linestyle="-")
    ax_map.add_patch(rect)
    # Events in region
    mask = (df["longitude"] >= lon_r[0]) & (df["longitude"] <= lon_r[1]) & \
           (df["latitude"]  >= lat_r[0]) & (df["latitude"]  <= lat_r[1])
    ax_map.scatter(df.loc[mask, "longitude"], df.loc[mask, "latitude"],
                   c="#E0554A", s=3, alpha=0.7)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    ax_map.set_title(name, fontsize=10, fontweight="bold")
    ax_map.grid(True, alpha=0.3, linestyle="--")

    # Depth distribution pie
    ax_pie = axes[1, idx]
    depth_labels = ["0-50km", "50-100km", "100-300km", ">300km"]
    depths = [res["depth_distribution"].get(l, 0) for l in depth_labels]
    colors = ["#4A90D9", "#50B86A", "#F5A623", "#E0554A"]
    if sum(depths) > 0:
        wedges, texts, autotexts = ax_pie.pie(
            depths, labels=depth_labels, autopct="%1.1f%%",
            colors=colors, startangle=140, pctdistance=0.6
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax_pie.set_title(f"Depth Distribution (n={res['count']})", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "task3_three_regions_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n  -> Saved: task3_three_regions_comparison.png")

# Save region results as JSON
with open(OUTPUT_DIR / "task3_region_stats.json", "w", encoding="utf-8") as f:
    json.dump(region_results, f, ensure_ascii=False, indent=2)
print("  -> Saved: task3_region_stats.json")

# Save to CSV as well
region_rows = []
for name, res in region_results.items():
    region_rows.append({
        "region": name,
        "lon_range": f"{res['lon_range'][0]}~{res['lon_range'][1]}",
        "lat_range": f"{res['lat_range'][0]}~{res['lat_range'][1]}",
        "count": res["count"],
        "mean_mag": res["mean_mag"],
        "max_mag": res["max_mag"],
        "mean_depth_km": res["mean_depth_km"],
        "max_depth_km": res["max_depth_km"],
    })
pd.DataFrame(region_rows).to_csv(OUTPUT_DIR / "task3_region_stats.csv", index=False, encoding="utf-8-sig")


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("Task 3 Complete!")
print("=" * 60)
print("Generated files:")
print("  task3_global_spatial_map.png   - 4-panel: scatter, lon/lat hist, depth map")
print("  task3_grid_heatmap.png         - 10°×10° grid density heatmap")
print("  task3_three_regions_comparison.png - 3-region map + depth pie")
print("  task3_grid_10x10.csv           - Grid cell statistics")
print("  task3_region_stats.csv         - Region comparison table")
print("  task3_region_stats.json        - Region comparison (JSON)")
