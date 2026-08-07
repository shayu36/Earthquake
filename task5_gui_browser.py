#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5: Interactive GUI — Earthquake Data Browser
==================================================
Features:
  - Filter by time range, magnitude, depth
  - Global map display with embedded matplotlib
  - Click on map points to view details
  - Data table with scrollable results
  - Stats summary panel
  - Export filtered data

Built with: Tkinter + matplotlib (no extra installs needed)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
DATA_DIR = Path(r"D:\Users\lenovo\Desktop\题目2_USGS地震数据")
INPUT_CSV = DATA_DIR / "task1_processed_data.csv"

# ── Load data ──
@pd.api.extensions.register_dataframe_accessor("quake")
class QuakeAccessor:
    """Cache-enabled accessor for filtered earthquake data."""
    def __init__(self, pandas_obj):
        self._obj = pandas_obj


def load_data():
    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df


# ================================================================
# Main Application
# ================================================================
class EarthquakeBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Earthquake Data Browser — USGS 2024-2025 M>=4.5")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)

        # Load data
        self.df_full = load_data()
        self.df = self.df_full.copy()
        self.selected_index = None

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Build UI
        self._build_control_panel()
        self._build_map()
        self._build_detail_panel()
        self._build_statusbar()

        # Initial render
        self._update_display()

    # ── Control Panel (left sidebar) ──
    def _build_control_panel(self):
        panel = ttk.Frame(self.root, width=320)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        title = ttk.Label(panel, text="Filter Controls", font=("", 13, "bold"))
        title.pack(pady=(5, 10))

        # --- Time filter ---
        frm_time = ttk.LabelFrame(panel, text="Time Range", padding=8)
        frm_time.pack(fill=tk.X, padx=5, pady=3)

        ttk.Label(frm_time, text="From (YYYY-MM-DD):").pack(anchor=tk.W)
        self.var_time_from = tk.StringVar(value="2024-01-01")
        ttk.Entry(frm_time, textvariable=self.var_time_from, width=22).pack(fill=tk.X, pady=2)

        ttk.Label(frm_time, text="To (YYYY-MM-DD):").pack(anchor=tk.W)
        self.var_time_to = tk.StringVar(value="2025-12-31")
        ttk.Entry(frm_time, textvariable=self.var_time_to, width=22).pack(fill=tk.X, pady=2)

        # --- Magnitude filter ---
        frm_mag = ttk.LabelFrame(panel, text="Magnitude Range", padding=8)
        frm_mag.pack(fill=tk.X, padx=5, pady=3)

        ttk.Label(frm_mag, text="Min Magnitude:").pack(anchor=tk.W)
        self.var_mag_min = tk.DoubleVar(value=4.5)
        ttk.Scale(frm_mag, from_=4.5, to=9.0, variable=self.var_mag_min,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_mag_label()).pack(fill=tk.X)
        self.lbl_mag_min = ttk.Label(frm_mag, text="4.5")
        self.lbl_mag_min.pack(anchor=tk.E)

        ttk.Label(frm_mag, text="Max Magnitude:").pack(anchor=tk.W)
        self.var_mag_max = tk.DoubleVar(value=9.0)
        ttk.Scale(frm_mag, from_=4.5, to=9.0, variable=self.var_mag_max,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_mag_label()).pack(fill=tk.X)
        self.lbl_mag_max = ttk.Label(frm_mag, text="9.0")
        self.lbl_mag_max.pack(anchor=tk.E)

        # --- Depth filter ---
        frm_depth = ttk.LabelFrame(panel, text="Depth Range (km)", padding=8)
        frm_depth.pack(fill=tk.X, padx=5, pady=3)

        ttk.Label(frm_depth, text="Min Depth:").pack(anchor=tk.W)
        self.var_depth_min = tk.DoubleVar(value=0)
        ttk.Scale(frm_depth, from_=0, to=700, variable=self.var_depth_min,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_depth_label()).pack(fill=tk.X)
        self.lbl_depth_min = ttk.Label(frm_depth, text="0")
        self.lbl_depth_min.pack(anchor=tk.E)

        ttk.Label(frm_depth, text="Max Depth:").pack(anchor=tk.W)
        self.var_depth_max = tk.DoubleVar(value=700)
        ttk.Scale(frm_depth, from_=0, to=700, variable=self.var_depth_max,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_depth_label()).pack(fill=tk.X)
        self.lbl_depth_max = ttk.Label(frm_depth, text="700")
        self.lbl_depth_max.pack(anchor=tk.E)

        # --- Buttons ---
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(btn_frame, text="Apply Filters", command=self._apply_filters).pack(
            side=tk.LEFT, padx=3, ipadx=8)
        ttk.Button(btn_frame, text="Reset", command=self._reset_filters).pack(
            side=tk.LEFT, padx=3, ipadx=8)
        ttk.Button(btn_frame, text="Export CSV", command=self._export_csv).pack(
            side=tk.LEFT, padx=3, ipadx=8)

        # --- Summary stats ---
        self.frm_stats = ttk.LabelFrame(panel, text="Filtered Summary", padding=8)
        self.frm_stats.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_stats = ttk.Label(self.frm_stats, text="", font=("Consolas", 10))
        self.lbl_stats.pack(anchor=tk.NW)

    def _update_mag_label(self):
        self.lbl_mag_min.config(text=f"{self.var_mag_min.get():.1f}")
        self.lbl_mag_max.config(text=f"{self.var_mag_max.get():.1f}")

    def _update_depth_label(self):
        self.lbl_depth_min.config(text=f"{self.var_depth_min.get():.0f}")
        self.lbl_depth_max.config(text=f"{self.var_depth_max.get():.0f}")

    # ── Map (center) ──
    def _build_map(self):
        map_frame = ttk.Frame(self.root)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, map_frame)
        toolbar.update()

        # Click event
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

    # ── Detail Panel (right) ──
    def _build_detail_panel(self):
        panel = ttk.Frame(self.root, width=350)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        ttk.Label(panel, text="Event Details", font=("", 13, "bold")).pack(pady=(5, 8))
        ttk.Label(panel, text="Click a point on the map to view details",
                  font=("", 9), foreground="gray").pack()

        # Text widget for details
        self.txt_detail = tk.Text(panel, width=42, height=18, font=("Consolas", 10),
                                   wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.txt_detail.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar
        scroll = ttk.Scrollbar(self.txt_detail, command=self.txt_detail.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_detail.config(yscrollcommand=scroll.set)

        # Data table below
        ttk.Label(panel, text="Filtered Data (first 20 rows)", font=("", 10, "bold")).pack(pady=(8, 2))

        tbl_frame = ttk.Frame(panel)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        columns = ("time", "mag", "depth", "place")
        self.tree = ttk.Treeview(tbl_frame, columns=columns, show="headings", height=10)
        self.tree.heading("time", text="Time")
        self.tree.heading("mag", text="Mag")
        self.tree.heading("depth", text="Depth")
        self.tree.heading("place", text="Place")
        self.tree.column("time", width=140)
        self.tree.column("mag", width=50, anchor=tk.CENTER)
        self.tree.column("depth", width=60, anchor=tk.CENTER)
        self.tree.column("place", width=200)

        tree_scroll = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ── Status Bar ──
    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor=tk.W, padding=3)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Apply / Reset Filters ──
    def _apply_filters(self):
        self.df = self.df_full.copy()

        # Time
        try:
            t_from = pd.Timestamp(self.var_time_from.get()).tz_localize("UTC")
            t_to   = pd.Timestamp(self.var_time_to.get()).tz_localize("UTC")
            self.df = self.df[(self.df["time"] >= t_from) & (self.df["time"] <= t_to)]
        except Exception as e:
            messagebox.showerror("Invalid Date", f"Date format error: {e}")
            return

        # Magnitude
        self.df = self.df[(self.df["mag"] >= self.var_mag_min.get()) &
                          (self.df["mag"] <= self.var_mag_max.get())]

        # Depth
        self.df = self.df[(self.df["depth"] >= self.var_depth_min.get()) &
                          (self.df["depth"] <= self.var_depth_max.get())]

        self._update_display()

    def _reset_filters(self):
        self.var_time_from.set("2024-01-01")
        self.var_time_to.set("2025-12-31")
        self.var_mag_min.set(4.5)
        self.var_mag_max.set(9.0)
        self.var_depth_min.set(0)
        self.var_depth_max.set(700)
        self._update_mag_label()
        self._update_depth_label()
        self.df = self.df_full.copy()
        self._update_display()

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(DATA_DIR),
            initialfile="filtered_data.csv"
        )
        if path:
            self.df.to_csv(path, index=False, encoding="utf-8-sig")
            self.status_var.set(f"Exported {len(self.df)} rows to {path}")

    # ── Update Display ──
    def _update_display(self):
        n = len(self.df)
        self.status_var.set(f"Showing {n} / {len(self.df_full)} earthquakes")

        # Stats
        if n > 0:
            stats_text = (
                f"  Records:        {n}\n"
                f"  Time range:\n"
                f"    {self.df['time'].min().strftime('%Y-%m-%d')}\n"
                f"    ~ {self.df['time'].max().strftime('%Y-%m-%d')}\n"
                f"  Magnitude:\n"
                f"    Mean: {self.df['mag'].mean():.2f}\n"
                f"    Min:  {self.df['mag'].min():.1f}\n"
                f"    Max:  {self.df['mag'].max():.1f}\n"
                f"  Depth:\n"
                f"    Mean: {self.df['depth'].mean():.0f} km\n"
                f"    Range: {self.df['depth'].min():.0f} ~ {self.df['depth'].max():.0f} km\n"
                f"  magType: {', '.join(self.df['magType'].value_counts().head(3).index.tolist())}"
            )
        else:
            stats_text = "  No data matching filters."

        self.lbl_stats.config(text=stats_text)

        # Redraw map
        self._draw_map()
        self.canvas.draw()

        # Update table
        self._update_table()

    def _draw_map(self):
        self.ax.clear()
        n = len(self.df)

        if n == 0:
            self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes,
                         ha="center", va="center", fontsize=16, color="gray")
            self.ax.set_xlim(-180, 180)
            self.ax.set_ylim(-90, 90)
            return

        # Sample for performance if > 5000
        if n > 5000:
            plot_df = self.df.sample(5000, random_state=42)
        else:
            plot_df = self.df

        self.scatter = self.ax.scatter(
            plot_df["longitude"], plot_df["latitude"],
            c=plot_df["mag"], cmap="plasma", s=6, alpha=0.6,
            norm=Normalize(vmin=4.5, vmax=9.0),
            edgecolors="none", picker=True, pickradius=3
        )

        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_xticks(np.arange(-180, 181, 60))
        self.ax.set_yticks(np.arange(-90, 91, 30))
        self.ax.set_xlabel("Longitude (°)", fontsize=10)
        self.ax.set_ylabel("Latitude (°)", fontsize=10)
        self.ax.set_title(f"Earthquake Epicenters (n={n})", fontsize=12, fontweight="bold")
        self.ax.grid(True, alpha=0.3, linestyle="--")

        # Colorbar (remove old, add new)
        if hasattr(self, "_cbar") and self._cbar is not None:
            self._cbar.remove()
        self._cbar = self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.75, pad=0.02)
        self._cbar.set_label("Magnitude", fontsize=9)

        # Store sample for click lookup
        self._plot_df = plot_df

    def _update_table(self):
        self.tree.delete(*self.tree.get_children())
        for _, row in self.df.head(20).iterrows():
            self.tree.insert("", tk.END, values=(
                row["time"].strftime("%Y-%m-%d %H:%M"),
                f"{row['mag']:.1f}",
                f"{row['depth']:.0f}",
                row["place"]
            ))

    # ── Map Click Handler ──
    def _on_map_click(self, event):
        if event.inaxes != self.ax or not hasattr(self, "_plot_df"):
            return
        if event.xdata is None or event.ydata is None:
            return

        # Find nearest point
        lon_click, lat_click = event.xdata, event.ydata
        plot_df = self._plot_df
        distances = np.sqrt(
            (plot_df["longitude"] - lon_click) ** 2 +
            (plot_df["latitude"]  - lat_click) ** 2
        )
        nearest_idx = distances.idxmin()
        row = plot_df.loc[nearest_idx]

        # Now find it in the full filtered df for complete details
        full_row = self.df.loc[nearest_idx] if nearest_idx in self.df.index else row

        self._show_detail(full_row)

    def _on_tree_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        idx = selection[0]
        # The tree shows first 20 — match by time string
        values = self.tree.item(idx)["values"]
        time_str = values[0]
        mag_str = values[1]

        # Find in filtered df
        matches = self.df[
            (self.df["time"].dt.strftime("%Y-%m-%d %H:%M") == time_str) &
            (self.df["mag"].round(1) == float(mag_str))
        ]
        if len(matches) > 0:
            self._show_detail(matches.iloc[0])

    def _show_detail(self, row):
        self.txt_detail.delete("1.0", tk.END)

        detail = (
            f"╔══════════════════════════════════╗\n"
            f"║  EARTHQUAKE EVENT DETAILS       ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"  Time (UTC):\n"
            f"    {row['time'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"  Location:\n"
            f"    Latitude:  {row['latitude']:.4f}°\n"
            f"    Longitude: {row['longitude']:.4f}°\n\n"
            f"  Magnitude:\n"
            f"    Mag:       {row['mag']:.1f} ({row['magType']})\n"
            f"    Mag Error: {row.get('magError', 'N/A')}\n\n"
            f"  Depth:\n"
            f"    {row['depth']:.1f} km (Error: {row.get('depthError', 'N/A')} km)\n\n"
            f"  Quality:\n"
            f"    RMS:       {row['rms']:.3f} s\n"
            f"    Gap:       {row.get('gap', 'N/A')}°\n"
            f"    NST:       {row.get('nst', 'N/A')}\n"
            f"    Horiz Err: {row.get('horizontalError', 'N/A')} km\n\n"
            f"  Place:\n"
            f"    {row['place']}\n\n"
            f"  ID: {row['id']}\n"
            f"  Network: {row['net']}\n"
            f"  Updated: {pd.Timestamp(row['updated']).strftime('%Y-%m-%d %H:%M')}\n"
        )
        self.txt_detail.insert("1.0", detail)

        # Highlight on map
        if hasattr(self, "_plot_df"):
            self._draw_map()
            self.ax.scatter(row["longitude"], row["latitude"],
                            c="red", s=120, marker="*", edgecolors="white", linewidth=1.5,
                            zorder=10, label="Selected")
            self.ax.legend(loc="lower right", fontsize=9)
            self.canvas.draw()


# ================================================================
# Main
# ================================================================
def main():
    root = tk.Tk()
    app = EarthquakeBrowser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
