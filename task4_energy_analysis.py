#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Representative Earthquakes and Energy Analysis
=======================================================
1. Top 10 events by magnitude
2. Monthly relative energy index — compare count vs energy peak
"""

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
DATA_DIR = Path(r"D:\Users\lenovo\Desktop\题目2_USGS地震数据")
INPUT_CSV = DATA_DIR / "task1_processed_data.csv"
OUTPUT_DIR = DATA_DIR

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

# Bar: count
bars = ax1.bar(x, monthly_df["count"], width=0.55, color="#4A90D9", alpha=0.85,
               edgecolor="white", linewidth=0.5, label="Earthquake Count")
ax1.set_xlabel("Month", fontsize=13)
ax1.set_ylabel("Earthquake Count", fontsize=13, color="#4A90D9")
ax1.tick_params(axis="y", labelcolor="#4A90D9")
ax1.set_xticks(x[::2])
ax1.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=9)

# Line: energy
ax2 = ax1.twinx()
ax2.plot(x, monthly_df["energy_pct"], color="#E0554A", marker="s", linewidth=2.5,
         markersize=7, label="Energy Index (%)")
ax2.set_ylabel("Relative Energy Index (%)", fontsize=13, color="#E0554A")
ax2.tick_params(axis="y", labelcolor="#E0554A")

# Annotations
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

# Legend
bars_leg = plt.Rectangle((0, 0), 1, 1, color="#4A90D9", alpha=0.85)
line_leg = plt.Line2D([0], [0], color="#E0554A", marker="s", linewidth=2.5, markersize=7)
ax1.legend([bars_leg, line_leg], ["Earthquake Count", "Energy Index (%)"],
           loc="upper left", fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "task4_monthly_energy.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved: task4_monthly_energy.png")

# ── Scatter: count vs energy ──
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(monthly_df["count"], monthly_df["energy_pct"],
                c=monthly_df["max_mag"], cmap="plasma", s=120, alpha=0.85,
                edgecolors="#333333", linewidth=0.5)

# Label each point
for i, m in enumerate(months):
    ax.annotate(m, (monthly_df["count"].iloc[i], monthly_df["energy_pct"].iloc[i]),
                textcoords="offset points", xytext=(6, 3), fontsize=7, alpha=0.8)

ax.set_xlabel("Earthquake Count", fontsize=13)
ax.set_ylabel("Relative Energy Index (%)", fontsize=13)
ax.set_title("Count vs Energy by Month", fontsize=14, fontweight="bold")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Max Magnitude", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

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
