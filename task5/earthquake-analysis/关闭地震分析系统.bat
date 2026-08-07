@echo off
chcp 65001 >nul

echo 正在关闭本地地震分析服务……

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000"') do (
    taskkill /PID %%a /T /F >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
    taskkill /PID %%a /T /F >nul 2>&1
)

echo 服务已经关闭。
timeout /t 2 >nul
