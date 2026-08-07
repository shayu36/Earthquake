from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 8501


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> bool:
    """等待指定端口启动。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def stop_process(process: subprocess.Popen | None) -> None:
    """停止进程及其子进程。"""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    backend_process: subprocess.Popen | None = None
    frontend_process: subprocess.Popen | None = None

    print("=" * 60)
    print("全球 M4.5+ 地震时空分析系统")
    print("=" * 60)
    print(f"项目目录：{PROJECT_ROOT}")
    print()

    try:
        # 1. 启动 FastAPI 后端
        print("[1/3] 正在启动 FastAPI 后端……")

        backend_command = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ]

        backend_process = subprocess.Popen(
            backend_command,
            cwd=PROJECT_ROOT,
        )

        if not wait_for_port(BACKEND_HOST, BACKEND_PORT, timeout=30):
            raise RuntimeError("FastAPI 后端启动失败，端口 8000 未响应。")

        print(f"FastAPI 已启动：http://{BACKEND_HOST}:{BACKEND_PORT}")
        print(f"API 文档：http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
        print()

        # 2. 启动 Streamlit 前端
        print("[2/3] 正在启动 Streamlit 前端……")

        frontend_command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "frontend" / "app.py"),
            "--server.address",
            FRONTEND_HOST,
            "--server.port",
            str(FRONTEND_PORT),
            "--browser.gatherUsageStats",
            "false",
        ]

        frontend_process = subprocess.Popen(
            frontend_command,
            cwd=PROJECT_ROOT,
        )

        if not wait_for_port(FRONTEND_HOST, FRONTEND_PORT, timeout=45):
            raise RuntimeError("Streamlit 前端启动失败，端口 8501 未响应。")

        frontend_url = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"
        print(f"Streamlit 已启动：{frontend_url}")
        print()

        # 3. 打开浏览器
        print("[3/3] 正在打开浏览器……")
        webbrowser.open(frontend_url)

        print()
        print("=" * 60)
        print("系统已经启动。")
        print(f"GUI 页面：{frontend_url}")
        print(f"API 文档：http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
        print()
        print("按 Ctrl+C 可关闭前端和后端。")
        print("=" * 60)

        # 持续检查子进程是否异常退出
        while True:
            if backend_process.poll() is not None:
                raise RuntimeError("FastAPI 后端进程意外退出。")
            if frontend_process.poll() is not None:
                raise RuntimeError("Streamlit 前端进程意外退出。")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在关闭地震分析系统……")

    except Exception as exc:
        print()
        print(f"启动失败：{exc}")
        print()
        print("请检查：")
        print("1. 是否已经安装 requirements.txt 中的依赖")
        print("2. 8000 或 8501 端口是否被其他程序占用")
        print("3. 是否在正确的 Python 环境中运行")

    finally:
        stop_process(frontend_process)
        stop_process(backend_process)
        print("前端和后端均已关闭。")


if __name__ == "__main__":
    main()
