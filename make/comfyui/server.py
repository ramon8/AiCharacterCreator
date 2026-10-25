"""Start/stop the local ComfyUI server headless.

All ComfyUI inputs, outputs and temp files are redirected to
ACC_WORK_DIR/comfyui so nothing lands in the repo or the install dir.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from make import settings
from make.comfyui import http
from make.comfyui.install import venv_python


def _state_dir() -> Path:
    path = settings.work_dir() / "comfyui"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pid_file() -> Path:
    return _state_dir() / "server.pid"


def is_up(url: str) -> bool:
    try:
        http.get_json(f"{url}/system_stats", timeout=2)
        return True
    except (http.ServerUnreachable, http.HttpError):
        return False


def start(wait: float = 300) -> str:
    url = settings.comfyui_url()
    if is_up(url):
        print(f"ComfyUI already running at {url}")
        return url
    root = settings.comfyui_dir()
    python = venv_python(root)
    main = root / "ComfyUI" / "main.py"
    if not main.exists() or not python.exists():
        raise SystemExit(f"ComfyUI not installed at {root}. Run `uv run make comfyui install`.")
    parsed = urlparse(url)
    state = _state_dir()
    cmd = [
        str(python), str(main),
        "--listen", parsed.hostname or "127.0.0.1", "--port", str(parsed.port or 8188),
        "--disable-auto-launch",
        "--deterministic",
        "--input-directory", str(state / "input"),
        "--output-directory", str(state / "output"),
        "--temp-directory", str(state),
    ]
    cmd += os.environ.get("COMFYUI_ARGS", "").split()
    log = open(state / "server.log", "ab")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(cmd, cwd=root / "ComfyUI", stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, creationflags=flags)
    _pid_file().write_text(str(process.pid))
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"ComfyUI exited with code {process.returncode}; see {state / 'server.log'}")
        if is_up(url):
            print(f"ComfyUI up at {url} (pid {process.pid}, log {state / 'server.log'})")
            return url
        time.sleep(1)
    raise SystemExit(f"ComfyUI did not come up within {wait}s; see {state / 'server.log'}")


def stop() -> None:
    pid_file = _pid_file()
    if not pid_file.exists():
        print("ComfyUI not started by us (no pid file)")
        return
    pid = int(pid_file.read_text())
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    pid_file.unlink()
    print(f"ComfyUI (pid {pid}) stopped")
