"""Run Blender headless with our own scripts.

`run(op, params, blend=None)` launches
`blender --background --factory-startup [blend] --python scripts/ops.py -- op params.json`
and returns the JSON the op wrote. Ops live in `make/blender/scripts/ops.py`;
they only depend on bpy/bmesh/numpy (Blender's own Python), never on `make`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from make import settings

OPS_SCRIPT = Path(__file__).parent / "scripts" / "ops.py"


class BlenderError(Exception):
    pass


def available() -> bool:
    return settings.blender_exe().is_file()


def run(op: str, params: dict | None = None, blend: Path | None = None, timeout: float = 900) -> dict:
    exe = settings.blender_exe()
    if not exe.is_file():
        raise BlenderError(f"Blender not found at {exe} (set BLENDER_EXE)")
    with tempfile.TemporaryDirectory(prefix="acc_blender_") as tmp:
        params_file, result_file = Path(tmp) / "params.json", Path(tmp) / "result.json"
        params_file.write_text(json.dumps({**(params or {}), "result": str(result_file)}), encoding="utf-8")
        cmd = [str(exe), "--background", "--factory-startup"]
        if blend is not None:
            cmd.append(str(blend))
        cmd += ["--python-exit-code", "1", "--python", str(OPS_SCRIPT), "--", op, str(params_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        if proc.returncode != 0 or not result_file.exists():
            log = (proc.stdout + proc.stderr)[-4000:]
            raise BlenderError(f"Blender op {op!r} failed (exit {proc.returncode}):\n{log}")
        return json.loads(result_file.read_text(encoding="utf-8"))


def inspect_textures(blend: Path) -> dict:
    """What a .blend carries as textures: per image its size, packing, color count,
    sampling interpolation and the meshes whose materials use it."""
    return run("inspect_textures", blend=blend)
