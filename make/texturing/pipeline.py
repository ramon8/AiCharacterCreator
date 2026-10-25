"""Texturing pipeline: single view (#18, `texture_single_view`) and multiview
(#19, `texture_multiview`, which loops `render_view` / generate / `project_view`
over several `View`s and reuses `finish_texture` and `pack`).

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
import numpy as np

from make.texturing.projection import (
    BAKE_SIZE,
    ViewBake,
    merge_views,
    project_view,
    render_painted,
    surface_mask,
    to_texture,
)
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
    views: list[View] = field(default_factory=list)
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


def render_view(mesh_blend: Path, mesh: MeshData, view: View, work_dir: Path,
                smooth_angle: float | None = None) -> ViewRenders:
    """Conditioning renders. `smooth_angle` (degrees): smooth the normal pass across
    edges flatter than that, so low-poly curved parts read as curved."""
    out_dir = Path(work_dir) / "views" / view.name
    near, far = view.depth_range(mesh)
    paths = blender.run("render_view", {"view": _blender_view(view), "near": near, "far": far,
                                        "out_dir": str(out_dir), "smooth_angle": smooth_angle},
                        blend=mesh_blend)
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


def finish_texture(bakes: Mapping[str, ViewBake], sharpness: float = 1.0,
                   surface: np.ndarray | None = None) -> Image.Image:
    """Blend the view bakes, bleed them past the UV island borders (and over any
    `surface` texel no view painted) and reduce to PSX."""
    color, covered = merge_views(list(bakes.values()), sharpness=sharpness)
    return reduce_to_psx(to_texture(color, covered, surface=surface))


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


# Multiview (#19) ------------------------------------------------------------

SHARPNESS = 4.0  # merge weight = facing**SHARPNESS: the camera that faces a texel most dominates
KEEP_FROM, KEEP_FULLY = 0.3, 0.6  # earlier-view confidence where repainting starts to fade / stops
NEUTRAL = (128, 128, 128)
SMOOTH_ANGLE = 45.0  # conditioning normals are smoothed across edges flatter than this (degrees)
# View-dependent prompting: tell the model which side it is painting, so the back
# does not get a second face or chest emblem.
VIEW_PROMPTS = {"front": "front view", "right": "side view from the right", "left": "side view from the left",
                "back": "back view, seen from behind", "top": "top view, seen from above"}


def view_prompt(prompt: str, view: View) -> str:
    phrase = VIEW_PROMPTS.get(view.name)
    return f"{prompt}, {phrase}" if phrase else prompt


def default_views(mesh: MeshData, resolution: int = 1024) -> list[View]:
    """Projection cameras, in generation order: the front first (it sets the look),
    then the sides that overlap it, the back, and the top of the head."""
    return [View.framing(mesh, name=name, azimuth=az, elevation=el, resolution=resolution)
            for name, az, el in (("front", 0, 0), ("right", 90, 0), ("left", -90, 0),
                                 ("back", 180, 0), ("top", 0, 89))]


def multiview_previews(mesh: MeshData, size: int = PREVIEW_SIZE) -> dict[str, View]:
    """Preview angles: the cameras and the angles halfway between them."""
    angles = {"front": (0, 0), "front_right": (45, 0), "right": (90, 0), "back_right": (135, 0),
              "back": (180, 0), "back_left": (-135, 0), "left": (-90, 0), "front_left": (-45, 0),
              "above_front": (20, 45), "above_back": (200, 45)}
    return {name: View.framing(mesh, name=name, azimuth=az, elevation=el, resolution=size)
            for name, (az, el) in angles.items()}


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def continuation(mesh: MeshData, view: View, bakes: Mapping[str, ViewBake]) -> tuple[Image.Image, Image.Image]:
    """(init, repaint) for generating `view` after `bakes`: the paint so far seen
    from `view`, and a soft mask that keeps what earlier views saw well (white =
    generate). Where the paint is uncertain the mask fades, so the new view
    blends into it instead of cutting against it."""
    color, _ = merge_views(list(bakes.values()), sharpness=SHARPNESS)
    best = np.max([b.weight for b in bakes.values()], axis=0)
    image, confidence = render_painted(mesh, view, color, best)
    image[confidence <= 0] = NEUTRAL
    repaint = 1 - _smoothstep(KEEP_FROM, KEEP_FULLY, confidence)
    init = Image.fromarray(np.clip(np.round(image), 0, 255).astype(np.uint8))
    return init, Image.fromarray(np.round(repaint * 255).astype(np.uint8), "L")


def _on_neutral(image: Image.Image, mask: Image.Image) -> Image.Image:
    """The generated object alone on a neutral background (for use as a reference)."""
    background = Image.new("RGB", image.size, NEUTRAL)
    return Image.composite(image.convert("RGB"), background, mask.convert("L").resize(image.size))


def texture_multiview(mesh_blend: Path, reference: Path | Image.Image, prompt: str, seed: int,
                      out_blend: Path, client: GenerationClient, work_dir: Path,
                      resolution: int = 1024, bake_size: int = BAKE_SIZE,
                      views: list[View] | None = None) -> TexturingResult:
    """Texture `mesh_blend` from several cameras and save it as `out_blend`.

    Coherence: the first view is generated from the reference alone; every later
    view gets the first generated view as a second reference (both through
    IP-Adapter "style transfer", so they pass on palette and style but not the
    composition: with "linear" every view repeated the front's chest emblem) and
    continues the paint already projected (repaint workflow with differential
    diffusion: keep what earlier views saw well, generate the rest, soft in between).
    Each view's prompt names the side it shows. Seams: views blend by
    facing**SHARPNESS with normals smoothed across edges flatter than SMOOTH_ANGLE,
    so the hand-over is gradual inside faces; then the 128 px box reduction smooths
    what is left. Surface no camera saw is filled from the painted surface around it."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reference = reference if isinstance(reference, Image.Image) else Image.open(reference)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    with _timed(timings, "export_mesh"):
        mesh = export_mesh(mesh_blend, work_dir)
    views = views or default_views(mesh, resolution)
    bakes: dict[str, ViewBake] = {}
    generated: dict[str, Image.Image] = {}
    first_view: Image.Image | None = None
    for view in views:
        view_dir = work_dir / "views" / view.name
        with _timed(timings, f"render_view:{view.name}"):
            renders = render_view(mesh_blend, mesh, view, work_dir, smooth_angle=SMOOTH_ANGLE)
        conditions = dict(depth=renders.depth, normal=renders.normal, mask=renders.mask,
                          prompt=view_prompt(prompt, view), seed=seed)
        if first_view is None:
            conditions["references"] = (reference,)
        else:
            init, repaint = continuation(mesh, view, bakes)
            init.save(view_dir / "init.png")
            repaint.save(view_dir / "repaint.png")
            conditions.update(references=(reference, first_view), init=init, repaint=repaint)
        with _timed(timings, f"generate:{view.name}"):
            image = client.generate(Conditions(**conditions))
        image.save(view_dir / "generated.png")
        generated[view.name] = image
        if first_view is None:
            first_view = _on_neutral(image, renders.mask)
        with _timed(timings, f"project_bake:{view.name}"):
            bakes[view.name] = project_view(mesh, view, image, bake_size=bake_size, smooth_angle=SMOOTH_ANGLE)

    surface = surface_mask(mesh, bake_size)
    seen = np.max([b.weight for b in bakes.values()], axis=0) > 0
    coverage = {"surface_texels": int(surface.sum()), "seen_by_a_camera": int((surface & seen).sum()),
                "filled_from_neighbours": int((surface & ~seen).sum())}
    (work_dir / "coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")
    with _timed(timings, "psx_reduce"):
        texture = finish_texture(bakes, sharpness=SHARPNESS, surface=surface)
        texture_path = work_dir / "texture_psx.png"
        texture.save(texture_path)
    with _timed(timings, "pack_and_preview"):
        preview_paths = pack(Path(mesh_blend), texture_path, Path(out_blend), multiview_previews(mesh))

    preview = work_dir / "preview.png"
    _sheet([*generated.values(), texture], PREVIEW_SIZE // 2).save(work_dir / "generated.png")
    _sheet([Image.open(p) for p in preview_paths], PREVIEW_SIZE // 2).save(preview)
    timings["total"] = round(time.perf_counter() - started, 2)
    (work_dir / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    return TexturingResult(blend=Path(out_blend), texture=texture_path, preview=preview, bakes=bakes,
                           views=views, timings=timings)
