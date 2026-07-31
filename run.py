"""
run.py
------
Convenience launcher that starts the FastAPI backend (Uvicorn) and the
Streamlit frontend together as subprocesses, so the whole project can be
started with a single command:

    python run.py

Press Ctrl+C to stop both processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from backend.config import get_settings

settings = get_settings()


def main() -> None:
    env = os.environ.copy()
    env["BACKEND_URL"] = f"http://localhost:{settings.api_port}"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api:app",
        "--host",
        settings.api_host,
        "--port",
        str(settings.api_port),
    ]
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.join("frontend", "streamlit_app.py"),
        "--server.port",
        str(settings.frontend_port),
    ]

    print(f"Starting FastAPI backend on port {settings.api_port} ...")
    backend_proc = subprocess.Popen(backend_cmd, env=env)

    # Give the backend a moment to boot (loads embedding model, builds index).
    time.sleep(3)

    print(f"Starting Streamlit frontend on port {settings.frontend_port} ...")
    frontend_proc = subprocess.Popen(frontend_cmd, env=env)

    def shutdown(*_args) -> None:
        print("\nShutting down...")
        for proc in (frontend_proc, backend_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            if backend_proc.poll() is not None or frontend_proc.poll() is not None:
                print("A process exited unexpectedly. Shutting down.")
                shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
