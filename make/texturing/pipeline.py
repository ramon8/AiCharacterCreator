"""Texturing pipeline, single view (#18). Each step is its own function so the
multiview ticket (#19) can loop `render_view` / `generate_view` / `project_view`
over several `View`s and reuse `finish_texture` and `pack` as they are.

    mesh.blend --export_mesh--> MeshData
               --render_view(View)--> depth/normal/mask renders
    renders + reference --generate_view(client)--> generated image
    MeshData + View + image --project_view--> ViewBake (color + weight on the UV)
    ViewBakes --finish_texture--> merge, pad, PSX reduce (128 px, <=256 colors)
    texture --pack--> textured .blend (packed, nearest) + preview renders

All intermediates go to `work_dir` (outside the repo).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from make import blender
from make.generation import Conditions, GenerationClient
from make.texturing.mesh import MeshData
from make.texturing.projection import BAKE_SIZE, ViewBake, merge_views, project_view, to_texture
from make.texturing.psx import reduce_to_psx
from make.texturing.views import View

PREVIEW_SIZE = 512


@dataclass(frozen=True)
class ViewRenders:
    depth: Image.Image
    normal: Image.Image
    mask: Image.Image


@dataclass
class TexturingResult:
    blend: Path
    texture: Path
    preview: Path
    bakes: dict[str, ViewBake]
    timings: dict[str, float] = field(default_factory=dict)


@contextmanager
def _timed(timings: dict[str, float], step: str):
    started = time.perf_counter()
    yield
    timings[step] = round(time.perf_counter() - started, 2)


def export_mesh(mesh_blend: Path, work_dir: Path) -> MeshData:
    path = Path(work_dir) / "mesh.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    blender.run("export_mesh", {"out": str(path)}, blend=mesh_blend)
    return MeshData.load(path)


def _blender_view(view: View) -> dict:
    return {**view.to_dict(), "matrix_world": view.matrix_world()}


def render_view(mesh_blend: Path, mesh: MeshData, view: View, work_dir: Path) -> ViewRenders:
    out_dir = Path(work_dir) / "views" / view.name
    near, far = view.depth_range(mesh)
    paths = blender.run("render_view", {"view": _blender_view(view), "near": near, "far": far,
                                        "out_dir": str(out_dir)}, blend=mesh_blend)
    (out_dir / "view.json").write_text(json.dumps(view.to_dict(), indent=1), encoding="utf-8")
    return ViewRenders(*(Image.open(paths[k]).convert(m) for k, m in
                         (("depth", "RGB"), ("normal", "RGB"), ("mask", "L"))))


def generate_view(client: GenerationClient, renders: ViewRenders, reference: Image.Image, prompt: str,
                  seed: int, out: Path | None = None) -> Image.Image:
    image = client.generate(Conditions(depth=renders.depth, normal=renders.normal, mask=renders.mask,
                                       references=(reference,), prompt=prompt, seed=seed))
    if out is not None:
        image.save(out)
    return image


def finish_texture(bakes: Mapping[str, ViewBake]) -> Image.Image:
    """Blend the view bakes, bleed them past the UV island borders and reduce to PSX."""
    color, covered = merge_views(list(bakes.values()))
    return reduce_to_psx(to_texture(color, covered))


def pack(mesh_blend: Path, texture: Path, out_blend: Path, previews: Mapping[str, View] = {}) -> list[Path]:
    """Save `mesh_blend` as `out_blend` with `texture` packed in; render the previews
    (shadeless, nearest) next to the texture and return their paths."""
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    jobs = [{"view": _blender_view(v), "path": str(texture.parent / f"preview_{name}.png")}
            for name, v in previews.items()]
    result = blender.run("pack_texture", {"texture": str(texture), "out": str(out_blend), "previews": jobs},
                         blend=mesh_blend)
    return [Path(p) for p in result["previews"]]


def _sheet(images: list[Image.Image], size: int) -> Image.Image:
    sheet = Image.new("RGB", (size * len(images), size), (255, 255, 255))
    for i, image in enumerate(images):
        sheet.paste(image.convert("RGB").resize((size, size), Image.Resampling.NEAREST), (i * size, 0))
    return sheet


def texture_single_view(mesh_blend: Path, reference: Path | Image.Image, prompt: str, seed: int,
                        out_blend: Path, client: GenerationClient, work_dir: Path,
                        resolution: int = 1024, bake_size: int = BAKE_SIZE) -> TexturingResult:
    """Texture `mesh_blend` from a front camera and save it as `out_blend`."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reference = reference if isinstance(reference, Image.Image) else Image.open(reference)
    timings: dict[str, float] = {}

    with _timed(timings, "export_mesh"):
        mesh = export_mesh(mesh_blend, work_dir)
    front = View.framing(mesh, name="front", resolution=resolution)
    with _timed(timings, "render_view"):
        renders = render_view(mesh_blend, mesh, front, work_dir)
    with _timed(timings, "generate"):
        generated = generate_view(client, renders, reference, prompt, seed,
                                  out=work_dir / "views" / front.name / "generated.png")
    with _timed(timings, "project_bake"):
        bakes = {front.name: project_view(mesh, front, generated, bake_size=bake_size)}
    with _timed(timings, "psx_reduce"):
        texture = finish_texture(bakes)
        texture_path = work_dir / "texture_psx.png"
        texture.save(texture_path)
    previews = {
        "front": View.framing(mesh, name="front", resolution=PREVIEW_SIZE),
        "three_quarter": View.framing(mesh, name="three_quarter", azimuth=35, elevation=10, resolution=PREVIEW_SIZE),
    }
    with _timed(timings, "pack_and_preview"):
        preview_paths = pack(Path(mesh_blend), texture_path, Path(out_blend), previews)

    preview = work_dir / "preview.png"
    _sheet([generated, *(Image.open(p) for p in preview_paths), texture], PREVIEW_SIZE).save(preview)
    (work_dir / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    return TexturingResult(blend=Path(out_blend), texture=texture_path, preview=preview, bakes=bakes,
                           timings=timings)
