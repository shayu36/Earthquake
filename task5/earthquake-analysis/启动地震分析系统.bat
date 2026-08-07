@echo off
chcp 65001 >nul
title 全球M4.5以上地震时空分析系统

cd /d "%~dp0"

echo ============================================================
echo 正在启动全球 M4.5+ 地震时空分析系统
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" start_gui.py
) else (
    python start_gui.py
)

echo.
echo 程序已经结束。
pause
