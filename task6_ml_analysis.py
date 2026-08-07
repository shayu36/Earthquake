#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 6: ML Analysis — DBSCAN Spatial Clustering + PCA Feature Analysis
======================================================================
- DBSCAN: density-based clustering on (longitude, latitude), auto-detects
  seismic belts and labels isolated events as noise.
- PCA: dimensionality reduction on numerical features to reveal structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
DATA_DIR = Path(r"D:\Users\lenovo\Desktop\题目2_USGS地震数据")
INPUT_CSV = DATA_DIR / "task1_processed_data.csv"

# ── Read ──
df = pd.read_csv(INPUT_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
print(f"Loaded {len(df)} records")

# ── Coordinate encoding for DBSCAN ──
# Use (lon, lat) directly after scaling so euclidean distance approximates
# degrees. At the equator 1° ≈ 111 km; near poles it distorts but the
# clusters we want (seismic belts) are still separable in degree-space.

coords = df[["longitude", "latitude"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)


# ================================================================
# 1. DBSCAN — tune eps via k-distance graph
# ================================================================
print("\n=== (1) DBSCAN Spatial Clustering ===")

# --- Step 1: k-distance graph to guide eps choice ---
k = 5  # min_samples candidate
nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords_scaled)
distances, _ = nbrs.kneighbors(coords_scaled)
k_dist = np.sort(distances[:, -1])

fig, axes = plt.subplots(2, 2, figsize=(16, 13))

ax = axes[0, 0]
ax.plot(k_dist)
ax.set_xlabel("Sorted Points", fontsize=11)
ax.set_ylabel(f"{k}-th Nearest Neighbor Distance", fontsize=11)
ax.set_title(f"k-Distance Graph (k={k}) — Elbow guides eps choice", fontsize=12, fontweight="bold")
# Mark typical elbows
for eps_guess, color in [(0.15, "red"), (0.25, "orange"), (0.40, "green")]:
    n_pts = (k_dist <= eps_guess).sum()
    ax.axhline(y=eps_guess, color=color, linestyle="--", alpha=0.7, linewidth=1)
    ax.text(len(k_dist) * 0.98, eps_guess, f" eps={eps_guess} ({n_pts} pts)",
            ha="right", va="bottom", fontsize=8, color=color)
ax.grid(alpha=0.3)

# --- Step 2: Run DBSCAN with two parameter sets ---
# (a) Fine-grained: catches tight clusters
# (b) Coarse: merges nearby sub-zones into main seismic belts
configs = [
    ("Fine Clusters\n(eps=0.15, min=10)", 0.15, 10),
    ("Main Seismic Belts\n(eps=0.30, min=15)", 0.30, 15),
]

results = {}
for idx, (label, eps, min_samp) in enumerate(configs):
    db = DBSCAN(eps=eps, min_samples=min_samp)
    clusters = db.fit_predict(coords_scaled)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise    = (clusters == -1).sum()
    noise_pct  = n_noise / len(df) * 100

    print(f"\n  {label}")
    print(f"    eps={eps}, min_samples={min_samp}")
    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points:   {n_noise} ({noise_pct:.1f}%)")

    # Silhouette score (exclude noise for scoring)
    if n_clusters >= 2:
        mask = clusters != -1
        sil = silhouette_score(coords_scaled[mask], clusters[mask])
        print(f"    Silhouette score: {sil:.3f}")

    results[label] = {"clusters": clusters, "n_clusters": n_clusters, "n_noise": n_noise}
    df[f"dbscan_{idx}"] = clusters

    # --- Plot cluster map ---
    ax = axes[0, 1] if idx == 0 else axes[1, 0]

    unique_labels = sorted(set(clusters))
    # Build colormap: noise=gray, clusters=spectrum
    n_colors = max(1, len(unique_labels) - (1 if -1 in unique_labels else 0))
    cmap = ListedColormap(["#888888"] + plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20))).tolist())

    # Plot noise first
    noise_mask = clusters == -1
    ax.scatter(df.loc[noise_mask, "longitude"], df.loc[noise_mask, "latitude"],
               c="#cccccc", s=0.5, alpha=0.4, label=f"Noise ({n_noise})")

    # Plot clusters (sample large ones for speed)
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = clusters == lbl
        cluster_pts = df.loc[mask]
        if len(cluster_pts) > 3000:
            cluster_pts = cluster_pts.sample(3000, random_state=42)
        ax.scatter(cluster_pts["longitude"], cluster_pts["latitude"],
                   s=2, alpha=0.6)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title(f"{label}\n{n_clusters} clusters, {noise_pct:.1f}% noise", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")

    # Cluster stats
    if n_clusters >= 1:
        print(f"\n    Top clusters (by size):")
        cluster_sizes = pd.Series(clusters[clusters != -1]).value_counts().head(8)
        for clbl, csize in cluster_sizes.items():
            sub = df[clusters == clbl]
            print(f"      Cluster {clbl}: {csize:5d} events  "
                  f"mean_mag={sub['mag'].mean():.2f}  "
                  f"center=({sub['longitude'].mean():.0f}°, {sub['latitude'].mean():.0f}°)  "
                  f"depth={sub['depth'].mean():.0f}km")

# Use the finer clusters for per-cluster stats CSV
best_clusters = results[list(results.keys())[0]]["clusters"]
cluster_rows = []
for lbl in sorted(set(best_clusters)):
    if lbl == -1:
        continue
    sub = df[best_clusters == lbl]
    cluster_rows.append({
        "cluster_id": int(lbl),
        "count": len(sub),
        "mean_mag": round(sub["mag"].mean(), 2),
        "max_mag": round(sub["mag"].max(), 1),
        "mean_depth": round(sub["depth"].mean(), 1),
        "center_lon": round(sub["longitude"].mean(), 1),
        "center_lat": round(sub["latitude"].mean(), 1),
        "time_start": str(sub["time"].min().date()),
        "time_end": str(sub["time"].max().date()),
    })
cluster_df = pd.DataFrame(cluster_rows).sort_values("count", ascending=False)
cluster_df.to_csv(DATA_DIR / "task6_dbscan_clusters.csv", index=False, encoding="utf-8-sig")
print(f"\n  -> Saved: task6_dbscan_clusters.csv")

# ── Per-cluster summary bar chart ──
ax = axes[1, 1]
top_clusters = cluster_df.head(12)
colors_bar = plt.cm.viridis(np.linspace(0.15, 0.85, len(top_clusters)))
bars = ax.barh(
    [f"C{c['cluster_id']:d} ({c['center_lon']:.0f}°,{c['center_lat']:.0f}°)" for _, c in top_clusters.iterrows()],
    top_clusters["count"],
    color=colors_bar, edgecolor="white"
)
ax.set_xlabel("Event Count", fontsize=10)
ax.set_title("Top DBSCAN Clusters by Size", fontsize=12, fontweight="bold")
ax.invert_yaxis()
for bar, (_, c) in zip(bars, top_clusters.iterrows()):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"n={c['count']}  Mmax={c['max_mag']}",
            va="center", fontsize=7)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig.savefig(DATA_DIR / "task6_dbscan_clustering.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved: task6_dbscan_clustering.png")


# ================================================================
# 2. PCA Feature Analysis
# ================================================================
print("\n=== (2) PCA Feature Analysis ===")

feature_cols = ["depth", "mag", "horizontalError", "depthError",
                "magError", "rms", "gap", "dmin", "nst", "magNst"]

# Drop rows with any NaN in these features
pca_df = df[feature_cols].dropna()
print(f"  Samples for PCA: {len(pca_df)} / {len(df)} "
      f"({len(df) - len(pca_df)} dropped due to NaN)")

X = StandardScaler().fit_transform(pca_df)

pca = PCA()
X_pca = pca.fit_transform(X)

# --- Explained variance ---
evr = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)

print(f"\n  Explained variance ratio:")
for i, (v, c) in enumerate(zip(evr[:6], cumsum[:6])):
    print(f"    PC{i+1}: {v*100:5.1f}%  (cumulative: {c*100:5.1f}%)")

# --- Biplot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) Scree plot
ax = axes[0]
colors_bar = ["#4A90D9"] * 5 + ["#cccccc"] * (len(evr) - 5)
ax.bar(range(1, len(evr)+1), evr * 100, color=colors_bar, edgecolor="white")
ax.plot(range(1, len(evr)+1), cumsum * 100, "o-", color="#E0554A", linewidth=2, markersize=6)
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Explained Variance (%)", fontsize=11)
ax.set_title("PCA Scree Plot", fontsize=13, fontweight="bold")
ax.set_xticks(range(1, len(evr)+1))
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=evr[0]*100, color="#4A90D9", linestyle="--", alpha=0.4)
ax.annotate(f"PC1: {evr[0]*100:.1f}%", (1, evr[0]*100), textcoords="offset points",
            xytext=(20, 5), fontsize=10, color="#4A90D9")

# (b) Biplot: PC1 vs PC2
ax = axes[1]
# Plot points (sample for speed)
n_sample = min(5000, len(X_pca))
idx_sample = np.random.RandomState(42).choice(len(X_pca), n_sample, replace=False)
sc = ax.scatter(X_pca[idx_sample, 0], X_pca[idx_sample, 1],
                c=pca_df["mag"].values[idx_sample], cmap="plasma",
                s=3, alpha=0.5, norm=plt.Normalize(4.5, 9))

# Loadings as arrows
loadings = pca.components_.T
scale_factor = X_pca[:, :2].std(axis=0) * 2.5
for i, col in enumerate(feature_cols):
    ax.arrow(0, 0,
             loadings[i, 0] * scale_factor[0],
             loadings[i, 1] * scale_factor[1],
             head_width=0.25, head_length=0.3, fc="#333333", ec="#333333", alpha=0.7, linewidth=1)
    ax.text(loadings[i, 0] * scale_factor[0] * 1.15,
            loadings[i, 1] * scale_factor[1] * 1.15,
            col, fontsize=8, color="#333333", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11)
ax.set_title("PCA Biplot: PC1 vs PC2", fontsize=13, fontweight="bold")
ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.3)
ax.grid(alpha=0.2)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Magnitude", fontsize=9)

# --- Top contributing features ---
print(f"\n  Top feature contributions:")
for pc_idx in range(3):
    pc_loadings = pd.Series(abs(loadings[:, pc_idx]), index=feature_cols).sort_values(ascending=False)
    print(f"    PC{pc_idx+1}: " + " | ".join([f"{f}={v:.3f}" for f, v in pc_loadings.head(4).items()]))

plt.tight_layout()
fig.savefig(DATA_DIR / "task6_pca_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n  -> Saved: task6_pca_analysis.png")

# Save PCA loadings
loadings_df = pd.DataFrame(
    loadings[:, :3],
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(3)]
)
loadings_df.to_csv(DATA_DIR / "task6_pca_loadings.csv", encoding="utf-8-sig")
print("  -> Saved: task6_pca_loadings.csv")


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("Task 6 Complete!")
print("=" * 60)
print("Generated files:")
print("  task6_dbscan_clustering.png  - k-distance + 2 cluster maps + bar chart")
print("  task6_dbscan_clusters.csv    - Per-cluster statistics")
print("  task6_pca_analysis.png       - Scree plot + biplot")
print("  task6_pca_loadings.csv       - Feature loadings per PC")
