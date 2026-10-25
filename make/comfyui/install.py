"""Reproducible ComfyUI install outside the repo.

Idempotent: re-running skips what is already in place. Layout under
COMFYUI_DIR (default C:\\Develop\\ComfyUI):

    ComfyUI/            git clone of ComfyUI core
    ComfyUI/custom_nodes/<node>/
    ComfyUI/models/<folder>/<file>
    .venv/              its own Python 3.12 venv with CUDA torch
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from make import settings
from make.comfyui import manifest


def _run(*cmd: str | Path, cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def _git_checkout(repo: str, ref: str, dest: Path) -> None:
    if not (dest / ".git").exists():
        _run("git", "clone", repo, dest)
    _run("git", "-C", dest, "fetch", "--quiet", "origin")
    _run("git", "-C", dest, "checkout", "--quiet", ref)
    if ref in ("master", "main"):
        _run("git", "-C", dest, "pull", "--quiet", "--ff-only")


def venv_python(root: Path) -> Path:
    return root / ".venv" / "Scripts" / "python.exe" if os.name == "nt" else root / ".venv" / "bin" / "python"


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not path.exists():
        path = path.with_name("token.txt")
    return path.read_text().strip() if path.exists() else None


def _complete_size(error: urllib.error.HTTPError) -> int | None:
    """Total size from a 416 answer's `Content-Range: bytes */<size>`, if given."""
    _, _, total = (error.headers.get("Content-Range") or "").partition("/")
    return int(total) if total.isdigit() else None


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"ok   {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    if token := _hf_token():
        request.add_header("Authorization", f"Bearer {token}")
    print(f"get  {url} -> {dest}", flush=True)
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if not (offset and error.code == 416):
            raise
        # Range past the end: the .part is the whole file if its size is the file's size.
        if _complete_size(error) == offset:
            part.rename(dest)
            return
        part.unlink()  # longer than the file, or unverifiable: start over
        return download(url, dest)
    with response, open(part, "ab" if offset else "wb") as out:
        if offset and response.status != 206:  # server ignored Range: restart
            out.truncate(0)
        shutil.copyfileobj(response, out, length=16 * 1024 * 1024)
    part.rename(dest)


def install(root: Path | None = None) -> None:
    root = root or settings.comfyui_dir()
    core = root / "ComfyUI"
    root.mkdir(parents=True, exist_ok=True)

    _git_checkout(manifest.COMFYUI_REPO, os.environ.get("COMFYUI_REF", manifest.COMFYUI_REF), core)

    python = venv_python(root)
    if not python.exists():
        _run("uv", "venv", "--python", "3.12", root / ".venv")
    _run("uv", "pip", "install", "--python", python, "torch", "torchvision", "torchaudio",
         "--index-url", manifest.TORCH_INDEX_URL)
    _run("uv", "pip", "install", "--python", python, "-r", core / "requirements.txt")

    for node in manifest.CUSTOM_NODES:
        dest = core / "custom_nodes" / node.name
        _git_checkout(node.repo, node.ref, dest)
        if (dest / "requirements.txt").exists():
            _run("uv", "pip", "install", "--python", python, "-r", dest / "requirements.txt")

    for model in manifest.MODELS:
        download(model.url, core / "models" / model.folder / model.filename)

    _run(python, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    print("ComfyUI installed at", core)


if __name__ == "__main__":
    install(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
