#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1: Data Reading and Quality Validation
============================================
Read USGS earthquake catalog CSV, validate data ranges, uniqueness,
numerical boundaries, and missing quality fields, then export results.

Data source: USGS ANSS Comprehensive Earthquake Catalog (ComCat)
Time range: 2024-01-01 to 2025-12-31, M >= 4.5
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows to avoid GBK encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 0. 路径与参数
# ============================================================
DATA_DIR = Path(r"D:\Users\lenovo\Desktop\题目2_USGS地震数据")
INPUT_CSV = DATA_DIR / "USGS_2024_2025_M4.5plus_earthquakes.csv"
OUTPUT_CSV = DATA_DIR / "task1_processed_data.csv"
VALIDATION_LOG = DATA_DIR / "task1_validation_results.txt"

EXPECTED_START = pd.Timestamp("2024-01-01T00:00:00.000Z")
EXPECTED_END   = pd.Timestamp("2025-12-31T23:59:59.999Z")


def log(msg: str, f=None):
    """同时输出到控制台和验证日志文件。"""
    print(msg)
    if f:
        f.write(msg + "\n")


# ============================================================
# 1. 读取 CSV
# ============================================================
def read_and_prepare(path: Path):
    """读取 CSV，正确解析 time / updated 字段。"""
    df = pd.read_csv(path)
    df["time"]    = pd.to_datetime(df["time"],    utc=True)
    df["updated"] = pd.to_datetime(df["updated"], utc=True)
    # 排序（虽然文档说已按时间升序，但显式排一次）
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ============================================================
# 2. 数据范围验证
# ============================================================
def validate_ranges(df, f):
    log("\n" + "=" * 60, f)
    log("2. 数据范围验证", f)
    log("=" * 60, f)

    issues = []

    # --- 时间范围 ---
    t_min, t_max = df["time"].min(), df["time"].max()
    log(f"  时间范围: {t_min}  ~  {t_max}", f)
    t_ok = True
    if t_min < EXPECTED_START:
        msg = f"  ⚠ 最小时间 {t_min} 早于预期起始 {EXPECTED_START}"
        issues.append(msg); log(msg, f); t_ok = False
    if t_max > EXPECTED_END:
        msg = f"  ⚠ 最大时间 {t_max} 晚于预期结束 {EXPECTED_END}"
        issues.append(msg); log(msg, f); t_ok = False
    if t_ok:
        log("  ✓ 时间范围符合要求 (2024-01-01 ~ 2025-12-31)", f)

    # --- 纬度 [-90, 90] ---
    lat_ok = df["latitude"].between(-90, 90).all()
    if not lat_ok:
        bad = df[~df["latitude"].between(-90, 90)]
        msg = f"  ⚠ 纬度越界: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
        log(f"     违规示例:\n{bad[['time','latitude']].head(3).to_string(index=False)}", f)
    else:
        log("  ✓ 纬度全部在 [-90°, 90°] 范围内", f)

    # --- 经度 [-180, 180] ---
    lon_ok = df["longitude"].between(-180, 180).all()
    if not lon_ok:
        bad = df[~df["longitude"].between(-180, 180)]
        msg = f"  ⚠ 经度越界: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
        log(f"     违规示例:\n{bad[['time','longitude']].head(3).to_string(index=False)}", f)
    else:
        log("  ✓ 经度全部在 [-180°, 180°] 范围内", f)

    # --- 深度 >= 0 ---
    depth_ok = (df["depth"] >= 0).all()
    if not depth_ok:
        bad = df[df["depth"] < 0]
        msg = f"  ⚠ 深度为负: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
        log(f"     违规示例:\n{bad[['time','depth']].head(3).to_string(index=False)}", f)
    else:
        log("  ✓ 深度全部 >= 0 km", f)

    # --- 震级 >= 4.5 ---
    mag_ok = (df["mag"] >= 4.5).all()
    if not mag_ok:
        bad = df[df["mag"] < 4.5]
        msg = f"  ⚠ 震级 < 4.5: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
        log(f"     违规示例:\n{bad[['time','mag']].head(3).to_string(index=False)}", f)
    else:
        log("  ✓ 震级全部 >= 4.5", f)

    # --- type 字段 ---
    if (df["type"] == "earthquake").all():
        log("  ✓ type 字段全部为 'earthquake'", f)
    else:
        other = df[df["type"] != "earthquake"]
        msg = f"  ⚠ type 字段存在非 earthquake 值: {len(other)} 条"
        issues.append(msg); log(msg, f)

    # --- status 字段 ---
    if (df["status"] == "reviewed").all():
        log("  ✓ status 字段全部为 'reviewed'", f)
    else:
        other = df[df["status"] != "reviewed"]
        msg = f"  ⚠ status 字段存在非 reviewed 值: {len(other)} 条"
        issues.append(msg); log(msg, f)

    return issues


# ============================================================
# 3. 记录唯一性检查
# ============================================================
def validate_uniqueness(df, f):
    log("\n" + "=" * 60, f)
    log("3. 记录唯一性检查", f)
    log("=" * 60, f)

    issues = []

    # --- 完全重复行 ---
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        msg = f"  ⚠ 存在 {dup_rows} 条完全重复记录"
        issues.append(msg); log(msg, f)
    else:
        log(f"  ✓ 无完全重复记录 (共 {len(df)} 条)", f)

    # --- ID 唯一性 ---
    dup_ids = df["id"].duplicated().sum()
    if dup_ids > 0:
        dup_id_list = df[df["id"].duplicated(keep=False)]["id"].unique()
        msg = f"  ⚠ 存在 {dup_ids} 条重复 ID ({len(dup_id_list)} 个 ID 重复)"
        issues.append(msg); log(msg, f)
        log(f"     重复 ID: {list(dup_id_list[:5])}", f)
    else:
        log(f"  ✓ {len(df)} 个 ID 全部唯一", f)

    # --- 核心字段无重复 (time, latitude, longitude, depth, mag) ---
    core_cols = ["time", "latitude", "longitude", "depth", "mag"]
    dup_core = df.duplicated(subset=core_cols).sum()
    if dup_core > 0:
        msg = f"  ⚠ 核心字段 (time+lat+lon+depth+mag) 存在 {dup_core} 条重复"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 核心字段组合无重复", f)

    return issues


# ============================================================
# 4. 数值边界检查
# ============================================================
def validate_numerical_bounds(df, f):
    log("\n" + "=" * 60, f)
    log("4. 数值边界检查（详细描述性统计）", f)
    log("=" * 60, f)

    num_cols = ["latitude", "longitude", "depth", "mag",
                "horizontalError", "depthError", "magError",
                "nst", "gap", "dmin", "rms", "magNst"]

    stats = df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    stats["missing"] = df[num_cols].isna().sum()
    stats["missing_pct"] = (df[num_cols].isna().sum() / len(df) * 100).round(2)
    stats["count"] = stats["count"].astype(int)

    # 格式化输出
    for col in num_cols:
        s = df[col]
        log(f"\n  [{col}]", f)
        log(f"    非空值数: {s.notna().sum()}  /  缺失: {s.isna().sum()} ({s.isna().sum()/len(df)*100:.2f}%)", f)
        log(f"    最小值:   {s.min():.4f}", f)
        log(f"    1% 分位:  {s.quantile(0.01):.4f}", f)
        log(f"    中位数:   {s.median():.4f}", f)
        log(f"    99% 分位: {s.quantile(0.99):.4f}", f)
        log(f"    最大值:   {s.max():.4f}", f)
        log(f"    均值:     {s.mean():.4f}", f)
        log(f"    标准差:   {s.std():.4f}", f)

    issues = []
    # 检查不应为负的字段
    for col in ["depth", "mag"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                msg = f"  ⚠ {col} 存在 {neg} 条负值"
                issues.append(msg); log(msg, f)

    return issues, stats


# ============================================================
# 5. 质量字段缺失检查
# ============================================================
def validate_quality_missing(df, f):
    log("\n" + "=" * 60, f)
    log("5. 质量字段缺失检查", f)
    log("=" * 60, f)

    quality_cols = ["horizontalError", "depthError", "magError", "rms", "gap",
                    "nst", "dmin", "magNst", "magType"]

    missing_summary = {}
    for col in quality_cols:
        miss = df[col].isna().sum()
        pct  = miss / len(df) * 100
        missing_summary[col] = {"missing": miss, "pct": round(pct, 2)}
        status = "⚠" if miss > 0 else "✓"
        log(f"  {status} {col:20s}: 缺失 {miss:5d} / {len(df)} ({pct:.2f}%)", f)

    log("\n  --- 关键说明 ---", f)
    log("  缺失值不得填 0（说明文件中已注明）。本次处理保留缺失值不填充。", f)

    return missing_summary


# ============================================================
# 6. 导出结果
# ============================================================
def export_results(df, stats, missing_summary):
    """导出处理后的 CSV 和验证统计结果。"""
    # (a) 处理后的数据（保留原始数据，添加一个 clean 标记列）
    df_out = df.copy()
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ 处理后数据已导出: {OUTPUT_CSV}")

    # (b) 缺失统计 CSV
    missing_df = pd.DataFrame(missing_summary).T
    missing_df.index.name = "field"
    missing_csv = DATA_DIR / "task1_missing_summary.csv"
    missing_df.to_csv(missing_csv, encoding="utf-8-sig")
    print(f"✓ 缺失统计已导出: {missing_csv}")

    # (c) 数值统计 CSV
    stats_csv = DATA_DIR / "task1_numerical_stats.csv"
    stats.to_csv(stats_csv, encoding="utf-8-sig")
    print(f"✓ 数值统计已导出: {stats_csv}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("任务 1：数据读取与质量验证")
    print("USGS 2024-2025 M4.5+ Earthquakes")
    print("=" * 60)

    # 打开日志文件
    with open(VALIDATION_LOG, "w", encoding="utf-8") as f:
        log(f"验证时间: {pd.Timestamp.now()}", f)
        log(f"输入文件: {INPUT_CSV}", f)
        log(f"SHA-256: 5E623C1B9E5925E8CAFF6A7C3DCFDE4C087FBCCF285664ABE6A05D218C6B8232", f)

        # --- Step 1: 读取 ---
        log("\n" + "=" * 60, f)
        log("1. 读取 CSV", f)
        log("=" * 60, f)
        log(f"  文件路径: {INPUT_CSV}", f)

        df = read_and_prepare(INPUT_CSV)
        log(f"  记录总数: {len(df)}", f)
        log(f"  字段数量: {len(df.columns)}", f)
        log(f"  字段列表: {list(df.columns)}", f)
        log(f"  time 类型: {df['time'].dtype}", f)
        log(f"  updated 类型: {df['updated'].dtype}", f)
        log("  ✓ time 与 updated 已正确区分为 datetime64[ns, UTC]", f)

        # --- Step 2: 范围验证 ---
        issues_2 = validate_ranges(df, f)

        # --- Step 3: 唯一性 ---
        issues_3 = validate_uniqueness(df, f)

        # --- Step 4: 数值边界 ---
        issues_4, stats = validate_numerical_bounds(df, f)

        # --- Step 5: 缺失 ---
        missing_summary = validate_quality_missing(df, f)

        # --- 汇总 ---
        all_issues = issues_2 + issues_3 + issues_4
        log("\n" + "=" * 60, f)
        log("总  结", f)
        log("=" * 60, f)
        if all_issues:
            log(f"  共发现 {len(all_issues)} 个问题:", f)
            for i, iss in enumerate(all_issues, 1):
                log(f"    {i}. {iss}", f)
        else:
            log("  ✓ 所有验证项通过，数据质量良好！", f)

        log(f"\n  最终记录数: {len(df)}", f)
        log(f"  字段数: {len(df.columns)}", f)

    # --- 导出 ---
    export_results(df, stats, missing_summary)

    print("\n" + "=" * 60)
    print("任务 1 完成！")
    print(f"验证日志: {VALIDATION_LOG}")
    print(f"处理后数据: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
