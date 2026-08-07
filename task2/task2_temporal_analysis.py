#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2: Temporal Statistics and Magnitude/Depth Analysis
=========================================================
1. Monthly M>=4.5 earthquake count (2024.01 ~ 2025.12) — line + bar chart
2. Magnitude distribution: all / 2024 / 2025 comparison
3. Depth distribution across ranges: all / 2024 / 2025 comparison
4. magType distribution: overall AND 2024 vs 2025 annual comparison

Usage:
    python task2_temporal_analysis.py
    python task2_temporal_analysis.py --input task1_processed_data.csv --output-dir ./outputs
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys, io
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Font setup for Chinese ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2: Temporal Statistics and Magnitude/Depth Analysis"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "task1_processed_data.csv",
        help="Path to processed CSV (from task 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory for output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MONTHLY_CSV = OUTPUT_DIR / "task2_monthly_counts.csv"
    MAG_DIST_CSV = OUTPUT_DIR / "task2_mag_distribution.csv"
    DEPTH_DIST_CSV = OUTPUT_DIR / "task2_depth_distribution.csv"
    MAGTYPE_CSV = OUTPUT_DIR / "task2_magtype_distribution.csv"
    MAGTYPE_ANNUAL_CSV = OUTPUT_DIR / "task2_magtype_annual.csv"

    # ── Read data ──
    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["year_month"] = df["time"].dt.to_period("M")

    print(f"Loaded {len(df)} records")
    print(f"Time range: {df['time'].min()} ~ {df['time'].max()}")

    # ================================================================
    # (1) Monthly earthquake count — 24 months
    # ================================================================
    print("\n=== (1) Monthly Earthquake Count ===")
    monthly = df.groupby("year_month").size()
    monthly.index = monthly.index.astype(str)
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["year_month", "count"]
    monthly_df.to_csv(MONTHLY_CSV, index=False, encoding="utf-8-sig")

    print(monthly_df.to_string(index=False))
    print(f"\n  Max month: {monthly.idxmax()} ({monthly.max()} events)")
    print(f"  Min month: {monthly.idxmin()} ({monthly.min()} events)")
    print(f"  Monthly mean: {monthly.mean():.1f}, std: {monthly.std():.1f}")

    # ── Plot: bar + line ──
    fig, ax1 = plt.subplots(figsize=(16, 7))

    x = np.arange(len(monthly))
    bars = ax1.bar(x, monthly.values, width=0.6, color="#4A90D9", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Month", fontsize=13)
    ax1.set_ylabel("Earthquake Count", fontsize=13, color="#4A90D9")
    ax1.tick_params(axis="y", labelcolor="#4A90D9")

    ax2 = ax1.twinx()
    ax2.plot(x, monthly.values, color="#E0554A", marker="o", linewidth=2, markersize=6)
    ax2.set_ylabel("Earthquake Count (Line)", fontsize=13, color="#E0554A")
    ax2.tick_params(axis="y", labelcolor="#E0554A")

    labels = [str(m) for m in monthly.index]
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(labels[::2], rotation=45, ha="right", fontsize=9)
    ax1.set_xlim(-0.6, len(monthly) - 0.4)

    ax1.set_title("Monthly M>=4.5 Earthquake Count (2024.01 - 2025.12)", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate max/min
    ymax, ymin = monthly.max(), monthly.min()
    xmax, xmin = monthly.values.argmax(), monthly.values.argmin()
    ax1.annotate(f"Max: {ymax}\n{labels[xmax]}", (xmax, ymax), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9, color="#E0554A", fontweight="bold")
    ax1.annotate(f"Min: {ymin}\n{labels[xmin]}", (xmin, ymin), textcoords="offset points",
                 xytext=(0, -18), ha="center", fontsize=9, color="#333333", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_monthly_counts.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task2_monthly_counts.png")

    # ================================================================
    # (2) Magnitude distribution — all / 2024 / 2025
    # ================================================================
    print("\n=== (2) Magnitude Distribution ===")

    mag_bins = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0]
    mag_labels = ["4.5-5.0", "5.0-5.5", "5.5-6.0", "6.0-6.5", "6.5-7.0", "7.0-7.5", "7.5-8.0", "8.0-9.0"]

    df["mag_bin"] = pd.cut(df["mag"], bins=mag_bins, labels=mag_labels, right=False)

    def mag_dist(subset, label):
        dist = subset["mag_bin"].value_counts().reindex(mag_labels, fill_value=0)
        dist_df = dist.reset_index()
        dist_df.columns = ["mag_range", label]
        return dist_df

    all_dist   = mag_dist(df, "all")
    d2024_dist = mag_dist(df[df["year"] == 2024], "2024")
    d2025_dist = mag_dist(df[df["year"] == 2025], "2025")

    mag_combined = all_dist.merge(d2024_dist, on="mag_range").merge(d2025_dist, on="mag_range")
    mag_combined.to_csv(MAG_DIST_CSV, index=False, encoding="utf-8-sig")
    print(mag_combined.to_string(index=False))

    # ── Plot: grouped bar ──
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(mag_labels))
    w = 0.25

    ax.bar(x - w, mag_combined["all"],  w, color="#4A90D9", alpha=0.85, label="All (2024-2025)")
    ax.bar(x,      mag_combined["2024"], w, color="#50B86A", alpha=0.85, label="2024")
    ax.bar(x + w,  mag_combined["2025"], w, color="#E0554A", alpha=0.85, label="2025")

    ax.set_xticks(x)
    ax.set_xticklabels(mag_labels, rotation=30, ha="right", fontsize=11)
    ax.set_xlabel("Magnitude Range", fontsize=13)
    ax.set_ylabel("Earthquake Count", fontsize=13)
    ax.set_title("Magnitude Distribution Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_yscale("log")

    for bar_group in ax.containers:
        ax.bar_label(bar_group, fmt="%d", fontsize=7, padding=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_mag_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task2_mag_distribution.png")


    # ================================================================
    # (3) Depth distribution
    # ================================================================
    print("\n=== (3) Depth Distribution ===")

    depth_bins = [0, 50, 100, 300, 1000]
    depth_labels = ["0-50 km", "50-100 km", "100-300 km", ">300 km"]

    df["depth_bin"] = pd.cut(df["depth"], bins=depth_bins, labels=depth_labels, right=False)

    def depth_dist(subset, label):
        dist = subset["depth_bin"].value_counts().reindex(depth_labels, fill_value=0)
        dist_df = dist.reset_index()
        dist_df.columns = ["depth_range", label]
        return dist_df

    depth_all   = depth_dist(df, "all")
    depth_2024  = depth_dist(df[df["year"] == 2024], "2024")
    depth_2025  = depth_dist(df[df["year"] == 2025], "2025")

    depth_combined = depth_all.merge(depth_2024, on="depth_range").merge(depth_2025, on="depth_range")
    depth_combined.to_csv(DEPTH_DIST_CSV, index=False, encoding="utf-8-sig")
    print(depth_combined.to_string(index=False))

    # ── Plot: pie (all) + grouped bar ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart
    colors = ["#4A90D9", "#50B86A", "#F5A623", "#E0554A"]
    explode = (0.02, 0.02, 0.02, 0.08)
    wedges, texts, autotexts = ax1.pie(
        depth_combined["all"], labels=depth_combined["depth_range"],
        autopct="%1.1f%%", colors=colors, explode=explode,
        startangle=140, pctdistance=0.6
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax1.set_title("Depth Distribution (All Data)", fontsize=13, fontweight="bold")

    # Grouped bar
    x = np.arange(len(depth_labels))
    w = 0.25
    ax2.bar(x - w, depth_combined["all"],  w, color="#4A90D9", alpha=0.85, label="All")
    ax2.bar(x,      depth_combined["2024"], w, color="#50B86A", alpha=0.85, label="2024")
    ax2.bar(x + w,  depth_combined["2025"], w, color="#E0554A", alpha=0.85, label="2025")
    ax2.set_xticks(x)
    ax2.set_xticklabels(depth_labels, fontsize=11)
    ax2.set_xlabel("Depth Range", fontsize=13)
    ax2.set_ylabel("Earthquake Count", fontsize=13)
    ax2.set_title("Depth Distribution Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_depth_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task2_depth_distribution.png")


    # ================================================================
    # (4) magType distribution — overall + 2024 vs 2025 comparison
    # ================================================================
    print("\n=== (4) magType Distribution ===")

    # --- Overall ---
    magtype_counts = df["magType"].value_counts()
    magtype_pct   = (magtype_counts / len(df) * 100).round(2)
    magtype_df = pd.DataFrame({"count": magtype_counts, "percentage": magtype_pct})
    magtype_df.index.name = "magType"
    magtype_df.to_csv(MAGTYPE_CSV, encoding="utf-8-sig")
    print(magtype_df.to_string())

    # --- Annual comparison (2024 vs 2025) ---
    print("\n  --- magType Annual Comparison (2024 vs 2025) ---")
    magtype_annual = pd.crosstab(df["magType"], df["year"])
    magtype_annual["all"] = magtype_annual.sum(axis=1)
    # Ensure columns exist even if year missing
    for y in [2024, 2025]:
        if y not in magtype_annual.columns:
            magtype_annual[y] = 0
    magtype_annual = magtype_annual[["all", 2024, 2025]]
    magtype_annual.to_csv(MAGTYPE_ANNUAL_CSV, encoding="utf-8-sig")
    print(magtype_annual.to_string())

    # ── Plot: horizontal bar (overall) + grouped bar (annual) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Overall horizontal bar
    mtypes = magtype_df.index.tolist()
    counts = magtype_df["count"].tolist()
    pcts   = magtype_df["percentage"].tolist()

    bars = ax1.barh(mtypes, counts, color="#4A90D9", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Count", fontsize=13)
    ax1.set_ylabel("Magnitude Type", fontsize=13)
    ax1.set_title("MagType Distribution (Overall)", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()

    for bar, pct in zip(bars, pcts):
        ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width())}  ({pct:.1f}%)",
                va="center", fontsize=10, color="#333333")
    ax1.set_xlim(0, max(counts) * 1.25)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # Annual grouped bar
    annual_types = magtype_annual.index.tolist()
    x = np.arange(len(annual_types))
    w = 0.3

    ax2.bar(x - w/2, magtype_annual[2024], w, color="#50B86A", alpha=0.85, label="2024")
    ax2.bar(x + w/2, magtype_annual[2025], w, color="#E0554A", alpha=0.85, label="2025")

    ax2.set_xticks(x)
    ax2.set_xticklabels(annual_types, fontsize=11)
    ax2.set_xlabel("Magnitude Type", fontsize=13)
    ax2.set_ylabel("Earthquake Count", fontsize=13)
    ax2.set_title("MagType Annual Comparison (2024 vs 2025)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar_group in ax2.containers:
        ax2.bar_label(bar_group, fmt="%d", fontsize=8, padding=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_magtype_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task2_magtype_distribution.png")


    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("=" * 60)
    print(f"  {MONTHLY_CSV.name}")
    print(f"  {MAG_DIST_CSV.name}")
    print(f"  {DEPTH_DIST_CSV.name}")
    print(f"  {MAGTYPE_CSV.name}")
    print(f"  {MAGTYPE_ANNUAL_CSV.name}")
    print(f"  task2_monthly_counts.png")
    print(f"  task2_mag_distribution.png")
    print(f"  task2_depth_distribution.png")
    print(f"  task2_magtype_distribution.png")


if __name__ == "__main__":
    main()
