#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Representative Earthquakes and Energy Analysis
=======================================================
1. Top 10 events by magnitude
2. Monthly relative energy index — compare count vs energy peak

Usage:
    python task4_energy_analysis.py
    python task4_energy_analysis.py --input task1_processed_data.csv --output-dir ./outputs
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 4: Representative Earthquakes and Energy Analysis"
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

    # ── Read data ──
    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year_month"] = df["time"].dt.to_period("M")

    print(f"Loaded {len(df)} records\n")

    # ================================================================
    # (1) Top 10 earthquakes by magnitude
    # ================================================================
    print("=" * 70)
    print("(1) Top 10 Earthquake Events by Magnitude")
    print("=" * 70)

    top10 = df.nlargest(10, "mag")[
        ["time", "latitude", "longitude", "depth", "mag", "magType", "place"]
    ].copy()
    top10["time_str"] = top10["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    top10 = top10.reset_index(drop=True)
    top10.index = top10.index + 1
    top10.index.name = "Rank"

    # Print formatted table
    print(f"\n{'Rank':<5} {'Time':<28} {'Lat':>8} {'Lon':>9} {'Depth(km)':>10} {'Mag':>6} {'MagType':>8}  Place")
    print("-" * 110)
    for i, row in top10.iterrows():
        print(f"{i:<5} {row['time_str']:<28} {row['latitude']:>8.3f} {row['longitude']:>9.3f} "
              f"{row['depth']:>10.1f} {row['mag']:>6.1f} {row['magType']:>8}  {row['place']}")

    top10_out = top10.drop(columns=["time"])
    top10_out.to_csv(OUTPUT_DIR / "task4_top10_events.csv", encoding="utf-8-sig")
    print(f"\n  -> Saved: task4_top10_events.csv")

    # Highlight key info
    print(f"\n  Largest event: M{top10['mag'].max():.1f} — {top10['place'].iloc[0]}")
    print(f"  Date: {top10['time_str'].iloc[0]}")

    # ================================================================
    # (2) Monthly Relative Energy Index
    # ================================================================
    print("\n" + "=" * 70)
    print("(2) Monthly Relative Energy Index")
    print("=" * 70)

    # Energy ∝ 10^(1.5 * M)
    df["energy"] = 10 ** (1.5 * df["mag"])

    monthly_count  = df.groupby("year_month").size()
    monthly_energy = df.groupby("year_month")["energy"].sum()

    # Normalize for readability
    monthly_energy_normalized = monthly_energy / monthly_energy.sum() * 100

    # Compile results
    monthly_df = pd.DataFrame({
        "count": monthly_count,
        "total_energy": monthly_energy,
        "energy_pct": monthly_energy_normalized,
        "mean_mag": df.groupby("year_month")["mag"].mean().round(3),
        "max_mag":  df.groupby("year_month")["mag"].max(),
    })
    monthly_df.index.name = "year_month"

    print(f"\n{'Month':<8} {'Count':>6} {'Energy':>14} {'Energy%':>8} {'MeanMag':>8} {'MaxMag':>6}")
    print("-" * 55)
    for idx, row in monthly_df.iterrows():
        print(f"{str(idx):<8} {int(row['count']):>6} {row['total_energy']:>14.4e} "
              f"{row['energy_pct']:>7.2f}% {row['mean_mag']:>8.3f} {row['max_mag']:>6.1f}")

    monthly_df.to_csv(OUTPUT_DIR / "task4_monthly_energy.csv", encoding="utf-8-sig")

    # ── Key comparison ──
    month_max_count  = monthly_count.idxmax()
    month_max_energy = monthly_energy.idxmax()

    print(f"\n  --- Key Comparison ---")
    print(f"  Month with MOST earthquakes:     {month_max_count}  ({monthly_count.max()} events)")
    print(f"  Month with HIGHEST energy index: {month_max_energy}  ({monthly_energy_normalized.max():.2f}%)")

    if month_max_count == month_max_energy:
        print("  => RESULT: The same month! Count and energy peaks coincide.")
    else:
        print(f"  => RESULT: DIFFERENT months! "
              f"Energy peak at {month_max_energy} is driven by larger-magnitude events, "
              f"not just frequency.")

    # ── Detailed comparison ──
    cnt_energy = monthly_energy.loc[month_max_count]
    max_energy = monthly_energy.loc[month_max_energy]
    cnt_count  = monthly_count.loc[month_max_count]
    max_count  = monthly_count.loc[month_max_energy]

    print(f"\n  {month_max_count}: {cnt_count} events, energy = {cnt_energy:.4e}")
    print(f"  {month_max_energy}: {max_count} events, energy = {max_energy:.4e}")
    print(f"  Energy ratio (peak / count-peak): {max_energy / cnt_energy:.2f}x")

    # ── Plot: dual-axis monthly count + energy ──
    fig, ax1 = plt.subplots(figsize=(16, 7))

    months = [str(m) for m in monthly_df.index]
    x = np.arange(len(months))

    bars = ax1.bar(x, monthly_df["count"], width=0.55, color="#4A90D9", alpha=0.85,
                   edgecolor="white", linewidth=0.5, label="Earthquake Count")
    ax1.set_xlabel("Month", fontsize=13)
    ax1.set_ylabel("Earthquake Count", fontsize=13, color="#4A90D9")
    ax1.tick_params(axis="y", labelcolor="#4A90D9")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, monthly_df["energy_pct"], color="#E0554A", marker="s", linewidth=2.5,
             markersize=7, label="Energy Index (%)")
    ax2.set_ylabel("Relative Energy Index (%)", fontsize=13, color="#E0554A")
    ax2.tick_params(axis="y", labelcolor="#E0554A")

    ymax_count = monthly_df["count"].max()
    xmax_count = monthly_df["count"].values.argmax()
    ax1.annotate(f"Most Events: {months[xmax_count]}\n({int(ymax_count)} events)",
                 (xmax_count, ymax_count), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=10, color="#4A90D9", fontweight="bold")

    ymax_energy = monthly_df["energy_pct"].max()
    xmax_energy = monthly_df["energy_pct"].values.argmax()
    ax1.annotate(f"Highest Energy: {months[xmax_energy]}\n({ymax_energy:.1f}%)",
                 (xmax_energy, ymax_energy), textcoords="offset points",
                 xytext=(0, -22), ha="center", fontsize=10, color="#E0554A", fontweight="bold")

    ax1.set_title("Monthly Earthquake Count vs Relative Energy Index", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    bars_leg = plt.Rectangle((0, 0), 1, 1, color="#4A90D9", alpha=0.85)
    line_leg = plt.Line2D([0], [0], color="#E0554A", marker="s", linewidth=2.5, markersize=7)
    ax1.legend([bars_leg, line_leg], ["Earthquake Count", "Energy Index (%)"],
               loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task4_monthly_energy.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved: task4_monthly_energy.png")

    # ── Scatter: count vs mean energy per event ──
    # Using mean_energy_per_event instead of total_energy avoids the
    # M8.8 outlier dominating the y-axis and spreads all 24 months out.
    monthly_df["mean_energy_per_event"] = monthly_df["total_energy"] / monthly_df["count"]
    monthly_df["plot_year"] = [m[:4] for m in months]

    fig, ax = plt.subplots(figsize=(12, 7))

    year_colors = {"2024": "#50B86A", "2025": "#E0554A"}

    for yr, color in year_colors.items():
        subset = monthly_df[monthly_df["plot_year"] == yr]
        ax.scatter(
            subset["count"], subset["mean_energy_per_event"],
            c=color, s=subset["count"] / 3, alpha=0.75,
            edgecolors="#333333", linewidth=0.5,
            label=f"{yr}  ({len(subset)} months)",
            zorder=3,
        )

    # Smart labels: only label notable months to avoid clutter
    outlier_threshold = monthly_df["mean_energy_per_event"].quantile(0.85)
    for i, m in enumerate(months):
        row = monthly_df.iloc[i]
        label = m  # full "2024-01" format
        is_notable = (
            row["mean_energy_per_event"] >= outlier_threshold
            or row["count"] >= monthly_df["count"].quantile(0.80)
        )
        if is_notable:
            ax.annotate(label,
                        (row["count"], row["mean_energy_per_event"]),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        else:
            ax.annotate(label.split("-")[1],
                        (row["count"], row["mean_energy_per_event"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, alpha=0.55)

    # Highlight 2025-07 (the peak on both axes)
    peak = monthly_df.loc["2025-07"]
    ax.annotate("★ 2025-07: M8.8\n  peak count + peak energy",
                (peak["count"], peak["mean_energy_per_event"]),
                textcoords="offset points", xytext=(15, -25),
                fontsize=9, color="#C0392B", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))

    ax.set_xlabel("Earthquake Count", fontsize=13)
    ax.set_ylabel("Mean Energy per Event  (total_energy / count)", fontsize=13)
    ax.set_title("Count vs Mean Energy per Event — 2024 vs 2025", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task4_count_vs_energy.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task4_count_vs_energy.png")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("Task 4 Complete!")
    print("=" * 70)
    print(f"  task4_top10_events.csv       - Top 10 earthquake events")
    print(f"  task4_monthly_energy.csv     - Monthly count & energy index")
    print(f"  task4_monthly_energy.png     - Dual-axis: count + energy")
    print(f"  task4_count_vs_energy.png    - Scatter: count vs energy")


if __name__ == "__main__":
    main()
