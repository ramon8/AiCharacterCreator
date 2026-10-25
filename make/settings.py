"""Machine-local locations, overridable through environment variables.

Nothing listed here lives inside the repo: ComfyUI, its models and every
intermediate or output image stay outside so the repo remains clean.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COMFYUI_DIR = Path(r"C:\Develop\ComfyUI")
DEFAULT_WORK_DIR = Path(r"C:\Develop\AiCharacterCreator-work")
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")


def blender_exe() -> Path:
    return Path(os.environ.get("BLENDER_EXE", DEFAULT_BLENDER_EXE))


def comfyui_dir() -> Path:
    return Path(os.environ.get("COMFYUI_DIR", DEFAULT_COMFYUI_DIR))


def work_dir() -> Path:
    return Path(os.environ.get("ACC_WORK_DIR", DEFAULT_WORK_DIR))


def comfyui_url() -> str:
    return os.environ.get("COMFYUI_URL", DEFAULT_COMFYUI_URL).rstrip("/")
