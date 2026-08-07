#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5: Interactive GUI — Earthquake Data Browser
==================================================
Features:
  - File selection (File → Open) with filename/records/time-range/status display
  - Filters: UTC date range, min_mag, depth range, magType
  - Monthly event count chart + lat/lon scatter (linked to filtered data)
  - Sortable table with full result count; click row → event detail
  - Export: filtered CSV and save current chart as PNG
  - Reset button

Usage:
    python task5_gui_browser.py
    python task5_gui_browser.py --input ../task1/earthquakes_prepared.csv
"""

import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Task 5: Interactive GUI Earthquake Browser")
    parser.add_argument(
        "--input", type=Path,
        default=BASE_DIR.parent / "task1" / "earthquakes_prepared.csv",
        help="Path to earthquakes_prepared.csv",
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year"] = df["time"].dt.year
    df["year_month"] = df["time"].dt.to_period("M").astype(str)
    return df


def filter_by_date(df, start_date_str, end_date_str):
    """End date INCLUSIVE (includes all records on end_date)."""
    start = pd.Timestamp(start_date_str, tz="UTC")
    end_exclusive = pd.Timestamp(end_date_str, tz="UTC") + pd.Timedelta(days=1)
    return df[(df["time"] >= start) & (df["time"] < end_exclusive)]


class EarthquakeBrowser:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("Earthquake Data Browser — USGS 2024-2025 M>=4.5")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 750)
        self.data_path = data_path
        self.current_fig_type = "scatter"  # "scatter" or "monthly"

        # Menu bar
        self._build_menu()

        # Load data
        self._load_file(data_path)

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Build UI
        self._build_control_panel()
        self._build_chart_area()
        self._build_detail_panel()
        self._build_statusbar()

        self._update_display()

    # ── Menu ──
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open CSV...", command=self._file_open, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Chart as PNG...", command=self._save_chart_png)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _file_open(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(BASE_DIR.parent),
        )
        if path:
            self.data_path = Path(path)
            self._load_file(self.data_path)
            self._reset_filters()
            self._update_display()
            self.status_var.set(f"Loaded: {self.data_path.name}")

    def _load_file(self, path):
        self.df_full = load_data(Path(path))
        self.df = self.df_full.copy()
        self._file_info = {
            "name": Path(path).name,
            "records": len(self.df_full),
            "time_min": str(self.df_full["time"].min()),
            "time_max": str(self.df_full["time"].max()),
        }

    # ── Control Panel (left) ──
    def _build_control_panel(self):
        panel = ttk.Frame(self.root, width=340)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        # File info
        frm_info = ttk.LabelFrame(panel, text="File Info", padding=6)
        frm_info.pack(fill=tk.X, padx=5, pady=3)
        self.lbl_file_info = ttk.Label(frm_info, text="", font=("Consolas", 9))
        self.lbl_file_info.pack(anchor=tk.W)

        # Time filter
        frm_time = ttk.LabelFrame(panel, text="Time Range (UTC, inclusive)", padding=6)
        frm_time.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(frm_time, text="From (YYYY-MM-DD):").pack(anchor=tk.W)
        self.var_time_from = tk.StringVar(value="2024-01-01")
        ttk.Entry(frm_time, textvariable=self.var_time_from, width=24).pack(fill=tk.X, pady=1)
        ttk.Label(frm_time, text="To (YYYY-MM-DD):").pack(anchor=tk.W)
        self.var_time_to = tk.StringVar(value="2025-12-31")
        ttk.Entry(frm_time, textvariable=self.var_time_to, width=24).pack(fill=tk.X, pady=1)

        # Magnitude
        frm_mag = ttk.LabelFrame(panel, text="Min Magnitude", padding=6)
        frm_mag.pack(fill=tk.X, padx=5, pady=3)
        self.var_mag_min = tk.DoubleVar(value=4.5)
        ttk.Scale(frm_mag, from_=4.5, to=9.0, variable=self.var_mag_min,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_labels()).pack(fill=tk.X)
        self.lbl_mag_min = ttk.Label(frm_mag, text="4.5")
        self.lbl_mag_min.pack(anchor=tk.E)

        # Depth
        frm_depth = ttk.LabelFrame(panel, text="Depth Range (km)", padding=6)
        frm_depth.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(frm_depth, text="Min:").pack(anchor=tk.W)
        self.var_depth_min = tk.DoubleVar(value=0)
        ttk.Scale(frm_depth, from_=0, to=700, variable=self.var_depth_min,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_labels()).pack(fill=tk.X)
        self.lbl_depth_min = ttk.Label(frm_depth, text="0")
        self.lbl_depth_min.pack(anchor=tk.E)
        ttk.Label(frm_depth, text="Max:").pack(anchor=tk.W)
        self.var_depth_max = tk.DoubleVar(value=700)
        ttk.Scale(frm_depth, from_=0, to=700, variable=self.var_depth_max,
                  orient=tk.HORIZONTAL, command=lambda v: self._update_labels()).pack(fill=tk.X)
        self.lbl_depth_max = ttk.Label(frm_depth, text="700")
        self.lbl_depth_max.pack(anchor=tk.E)

        # magType
        frm_mt = ttk.LabelFrame(panel, text="magType", padding=6)
        frm_mt.pack(fill=tk.X, padx=5, pady=3)
        self.var_magtype = tk.StringVar(value="All")
        all_types = sorted(self.df_full["magType"].dropna().unique().tolist())
        ttk.Combobox(frm_mt, textvariable=self.var_magtype,
                     values=["All"] + all_types, state="readonly", width=22).pack(fill=tk.X)

        # Chart toggle
        frm_chart = ttk.LabelFrame(panel, text="Chart", padding=6)
        frm_chart.pack(fill=tk.X, padx=5, pady=3)
        self.var_chart = tk.StringVar(value="scatter")
        ttk.Radiobutton(frm_chart, text="Lat/Lon Scatter", variable=self.var_chart,
                        value="scatter", command=self._update_display).pack(anchor=tk.W)
        ttk.Radiobutton(frm_chart, text="Monthly Count", variable=self.var_chart,
                        value="monthly", command=self._update_display).pack(anchor=tk.W)

        # Buttons
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, padx=5, pady=8)
        ttk.Button(btn_frame, text="Apply Filters", command=self._apply_filters).pack(
            side=tk.LEFT, padx=2, ipadx=6)
        ttk.Button(btn_frame, text="Reset", command=self._reset_filters).pack(
            side=tk.LEFT, padx=2, ipadx=6)
        ttk.Button(btn_frame, text="Export CSV", command=self._export_csv).pack(
            side=tk.LEFT, padx=2, ipadx=6)

        # Summary
        self.frm_stats = ttk.LabelFrame(panel, text="Filtered Summary", padding=6)
        self.frm_stats.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        self.lbl_stats = ttk.Label(self.frm_stats, text="", font=("Consolas", 9))
        self.lbl_stats.pack(anchor=tk.NW)

    def _update_labels(self):
        self.lbl_mag_min.config(text=f"{self.var_mag_min.get():.1f}")
        self.lbl_depth_min.config(text=f"{self.var_depth_min.get():.0f}")
        self.lbl_depth_max.config(text=f"{self.var_depth_max.get():.0f}")

    # ── Chart Area (center) ──
    def _build_chart_area(self):
        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        # Two subplots toggled by visibility
        self.ax_scatter = self.fig.add_subplot(111)
        self.ax_monthly = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_scatter_click)

    # ── Detail Panel (right) ──
    def _build_detail_panel(self):
        panel = ttk.Frame(self.root, width=360)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        ttk.Label(panel, text="Event Details", font=("", 13, "bold")).pack(pady=(5, 5))
        ttk.Label(panel, text="Click scatter point or select table row",
                  font=("", 9), foreground="gray").pack()

        self.txt_detail = tk.Text(panel, width=44, height=16, font=("Consolas", 10),
                                   wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.txt_detail.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll = ttk.Scrollbar(self.txt_detail, command=self.txt_detail.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_detail.config(yscrollcommand=scroll.set)

        # Table
        ttk.Label(panel, text="Filtered Results", font=("", 10, "bold")).pack(pady=(8, 2))
        tbl_frame = ttk.Frame(panel)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        columns = ("time", "mag", "depth", "magType", "place")
        self.tree = ttk.Treeview(tbl_frame, columns=columns, show="headings", height=12)
        self.tree.heading("time", text="Time (UTC)")
        self.tree.heading("mag", text="Mag")
        self.tree.heading("depth", text="Depth")
        self.tree.heading("magType", text="Type")
        self.tree.heading("place", text="Place")
        self.tree.column("time", width=130)
        self.tree.column("mag", width=45, anchor=tk.CENTER)
        self.tree.column("depth", width=55, anchor=tk.CENTER)
        self.tree.column("magType", width=50, anchor=tk.CENTER)
        self.tree.column("place", width=160)

        tree_scroll_y = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_x = ttk.Scrollbar(tbl_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        tbl_frame.grid_rowconfigure(0, weight=1)
        tbl_frame.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ── Status Bar ──
    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor=tk.W, padding=3)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Filters ──
    def _apply_filters(self):
        try:
            self.df = filter_by_date(self.df_full, self.var_time_from.get(), self.var_time_to.get())
        except Exception as e:
            messagebox.showerror("Invalid Date", f"Date format error: {e}")
            return
        self.df = self.df[self.df["mag"] >= self.var_mag_min.get()]
        self.df = self.df[(self.df["depth"] >= self.var_depth_min.get()) &
                          (self.df["depth"] <= self.var_depth_max.get())]
        if self.var_magtype.get() != "All":
            self.df = self.df[self.df["magType"] == self.var_magtype.get()]
        self._update_display()

    def _reset_filters(self):
        self.var_time_from.set("2024-01-01")
        self.var_time_to.set("2025-12-31")
        self.var_mag_min.set(4.5)
        self.var_depth_min.set(0)
        self.var_depth_max.set(700)
        self.var_magtype.set("All")
        self._update_labels()
        self.df = self.df_full.copy()
        self._update_display()

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
            initialdir=str(BASE_DIR), initialfile="filtered_data.csv",
        )
        if path:
            self.df.to_csv(path, index=False, encoding="utf-8-sig")
            self.status_var.set(f"Exported {len(self.df)} rows → {path}")

    def _save_chart_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")],
            initialdir=str(BASE_DIR), initialfile="chart.png",
        )
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"Chart saved → {path}")

    # ── Update Display ──
    def _update_display(self):
        n = len(self.df)

        # File info
        self.lbl_file_info.config(text=(
            f"File: {self._file_info['name']}\n"
            f"Records: {self._file_info['records']}\n"
            f"Time: {self._file_info['time_min'][:10]} ~ {self._file_info['time_max'][:10]}"
        ))

        # Stats
        if n > 0:
            stats = (
                f"Filtered: {n} / {len(self.df_full)}\n"
                f"Time: {self.df['time'].min().strftime('%Y-%m-%d')} ~ "
                f"{self.df['time'].max().strftime('%Y-%m-%d')}\n"
                f"Mag: {self.df['mag'].min():.1f} ~ {self.df['mag'].max():.1f} "
                f"(mean {self.df['mag'].mean():.2f})\n"
                f"Depth: {self.df['depth'].min():.0f} ~ {self.df['depth'].max():.0f} km"
            )
        else:
            stats = "No data matching filters."
        self.lbl_stats.config(text=stats)

        self._draw_chart()
        self._update_table()
        self.status_var.set(f"Showing {n} events")

    # ── Chart ──
    def _draw_chart(self):
        self.fig.clear()
        chart_type = self.var_chart.get()
        n = len(self.df)

        if chart_type == "scatter":
            self.ax = self.fig.add_subplot(111)
            if n == 0:
                self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes, ha="center", va="center")
                self.ax.set_xlim(-180, 180); self.ax.set_ylim(-90, 90)
            else:
                plot_df = self.df if n <= 5000 else self.df.sample(5000, random_state=42)
                sizes = np.clip((plot_df["mag"] - 4.0) * 6, 1, 60)
                self._scatter = self.ax.scatter(
                    plot_df["longitude"], plot_df["latitude"],
                    c=plot_df["depth"], cmap="viridis_r", s=sizes, alpha=0.55,
                    norm=Normalize(vmin=0, vmax=700), edgecolors="none", picker=True, pickradius=5,
                )
                self.ax.set_xlim(-180, 180); self.ax.set_ylim(-90, 90)
                self.ax.set_xlabel("Longitude (°)", fontsize=9)
                self.ax.set_ylabel("Latitude (°)", fontsize=9)
                cbar = self.fig.colorbar(self._scatter, ax=self.ax, shrink=0.75, pad=0.02)
                cbar.set_label("Depth (km)", fontsize=8)
                self._plot_df = plot_df
            self.ax.set_title(f"Epicenters (n={n})", fontsize=11, fontweight="bold")
            self.ax.grid(True, alpha=0.3, linestyle="--")

        elif chart_type == "monthly":
            self.ax = self.fig.add_subplot(111)
            if n > 0:
                monthly = self.df.groupby("year_month").size()
                full_months = pd.period_range("2024-01", "2025-12", freq="M")
                monthly = monthly.reindex(full_months, fill_value=0)
                monthly.index = monthly.index.astype(str)
                x = np.arange(len(monthly))
                self.ax.bar(x, monthly.values, width=0.6, color="#4A90D9", alpha=0.85)
                self.ax.set_xticks(x[::2])
                self.ax.set_xticklabels(monthly.index[::2], rotation=45, ha="right", fontsize=8)
                self.ax.set_ylabel("Count", fontsize=9)
            else:
                self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes, ha="center", va="center")
            self.ax.set_title(f"Monthly Event Count (n={n})", fontsize=11, fontweight="bold")
            self.ax.grid(axis="y", alpha=0.3, linestyle="--")

        self.fig.tight_layout()
        self.canvas.draw()

    # ── Table ──
    def _update_table(self):
        self.tree.delete(*self.tree.get_children())
        for _, row in self.df.head(50).iterrows():
            event_id = str(row["id"])
            self.tree.insert("", tk.END, iid=event_id, values=(
                row["time"].strftime("%Y-%m-%d %H:%M"),
                f"{row['mag']:.1f}",
                f"{row['depth']:.0f}",
                row.get("magType", ""),
                row["place"],
            ))

    # ── Click / Select ──
    def _on_scatter_click(self, event):
        if event.inaxes != self.ax or not hasattr(self, "_plot_df"):
            return
        if event.xdata is None or event.ydata is None:
            return
        plot_df = self._plot_df
        distances = np.sqrt((plot_df["longitude"] - event.xdata)**2 + (plot_df["latitude"] - event.ydata)**2)
        nearest_idx = distances.idxmin()
        event_id = str(plot_df.loc[nearest_idx, "id"])
        matches = self.df[self.df["id"].astype(str) == event_id]
        if len(matches) > 0:
            self._show_detail(matches.iloc[0])

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        event_id = sel[0]
        matches = self.df[self.df["id"].astype(str) == event_id]
        if len(matches) > 0:
            self._show_detail(matches.iloc[0])

    def _show_detail(self, row):
        self.txt_detail.delete("1.0", tk.END)
        detail = (
            f"ID:       {row['id']}\n"
            f"Time:     {row['time'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Lat:      {row['latitude']:.4f}°\n"
            f"Lon:      {row['longitude']:.4f}°\n"
            f"Mag:      {row['mag']:.1f} ({row.get('magType','')})\n"
            f"Depth:    {row['depth']:.1f} km\n"
            f"Place:    {row['place']}\n"
            f"Network:  {row.get('net','')}\n"
            f"RMS:      {row.get('rms','N/A')}\n"
            f"Mag Err:  {row.get('magError','N/A')}\n"
            f"Dep Err:  {row.get('depthError','N/A')}\n"
            f"NST:      {row.get('nst','N/A')}\n"
            f"Gap:      {row.get('gap','N/A')}°\n"
            f"Horz Err: {row.get('horizontalError','N/A')} km\n"
            f"Updated:  {pd.Timestamp(row['updated']).strftime('%Y-%m-%d %H:%M')}"
        )
        self.txt_detail.insert("1.0", detail)

        # Highlight on scatter
        if self.var_chart.get() == "scatter" and hasattr(self, "_plot_df"):
            self._draw_chart()
            self.ax.scatter(row["longitude"], row["latitude"],
                            c="red", s=120, marker="*", edgecolors="white", linewidth=1.5, zorder=10)
            self.canvas.draw()


def main():
    args = parse_args()
    root = tk.Tk()
    EarthquakeBrowser(root, args.input)
    root.mainloop()


if __name__ == "__main__":
    main()
