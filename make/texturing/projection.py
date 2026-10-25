"""Camera projection of a generated view onto the mesh UV ("projection + bake").

Done in numpy instead of a Cycles bake: for every texel of the UV layout we
know its 3D point, project it with the same `View` that rendered the
conditioning images, test it against a depth buffer of the mesh for
occlusion and sample the generated image. Deterministic, GPU-free and
testable without Blender.

Each view gives a `ViewBake` (color + weight per texel). Weight is the
facing ratio (cosine between face normal and view direction) for visible
texels and 0 elsewhere, so several views can be blended (#19) with
`merge_views` before `to_texture` fills the gaps.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from make.texturing.mesh import MeshData
from make.texturing.views import View

BAKE_SIZE = 512
MIN_FACING = 0.05  # grazing faces get no paint: their projection is smeared


@dataclass(frozen=True)
class ViewBake:
    color: np.ndarray  # (H, W, 3) float, 0..255
    weight: np.ndarray  # (H, W) float, 0 = not painted by this view


def _rasterize(tris: np.ndarray, width: int, height: int) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """For each triangle (T, 3, 2) in pixel coordinates, yield (index, ys, xs, barycentrics)
    of the pixel centers it covers. Degenerate triangles cover nothing."""
    for i, (a, b, c) in enumerate(tris):
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c) - 0.5).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c) + 0.5).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-12:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
        w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area
        w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area
        w2 = 1 - w0 - w1
        bary = np.stack([w0, w1, w2], axis=1)
        inside = (bary >= -1e-9).all(axis=1)
        if inside.any():
            yield i, ys.ravel()[inside], xs.ravel()[inside], bary[inside]


def depth_buffer(mesh: MeshData, view: View) -> np.ndarray:
    """(res, res) nearest depth of the mesh per camera pixel, inf where there is none."""
    res = view.resolution
    x, y, depth = view.project(mesh.positions)
    zbuf = np.full((res, res), np.inf)
    for i, ys, xs, bary in _rasterize(np.stack([x, y], axis=-1), res, res):
        np.minimum.at(zbuf, (ys, xs), bary @ depth[i])
    return zbuf


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1)
    y = np.clip(y - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def project_view(mesh: MeshData, view: View, image: Image.Image, bake_size: int = BAKE_SIZE) -> ViewBake:
    """Project `image`, seen from `view`, onto the mesh UV at `bake_size` px."""
    pixels = np.asarray(image.convert("RGB").resize((view.resolution, view.resolution)), dtype=np.float64)
    zbuf = depth_buffer(mesh, view)
    pixel_world = view.ortho_scale / view.resolution
    facing = mesh.normals @ -view.basis()[2]

    color = np.zeros((bake_size, bake_size, 3))
    weight = np.zeros((bake_size, bake_size))
    uv_px = np.stack([mesh.uvs[..., 0] * bake_size, (1 - mesh.uvs[..., 1]) * bake_size], axis=-1)
    for i, ys, xs, bary in _rasterize(uv_px, bake_size, bake_size):
        if facing[i] < MIN_FACING:
            continue
        x, y, depth = view.project(bary @ mesh.positions[i])
        px = np.clip(x.astype(int), 0, view.resolution - 1)
        py = np.clip(y.astype(int), 0, view.resolution - 1)
        # Depth tolerance grows with the slope of the face across one camera pixel.
        slope = np.sqrt(max(1 - facing[i] ** 2, 0)) / facing[i]
        visible = depth <= zbuf[py, px] + 1.5 * pixel_world * (1 + slope)
        visible &= (x >= 0) & (x < view.resolution) & (y >= 0) & (y < view.resolution)
        ys, xs = ys[visible], xs[visible]
        color[ys, xs] = _sample_bilinear(pixels, x[visible], y[visible])
        weight[ys, xs] = facing[i]
    return ViewBake(color=color, weight=weight)


def merge_views(bakes: Sequence[ViewBake]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted blend of several view bakes -> (color, coverage mask)."""
    total = sum(b.weight for b in bakes)
    color = sum(b.color * b.weight[..., None] for b in bakes)
    covered = total > 0
    color[covered] /= total[covered][:, None]
    return color, covered


def to_texture(color: np.ndarray, covered: np.ndarray, padding: int = 8,
               fill: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Bleed painted texels `padding` px into their surroundings (so UV island borders
    don't pick up the fill once downscaled), then paint what is left with `fill`."""
    color, covered = color.copy(), covered.copy()
    h, w = covered.shape
    for _ in range(padding):
        acc = np.zeros_like(color)
        count = np.zeros((h, w))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            src = np.zeros((h, w), bool)
            shifted = np.zeros_like(color)
            ys = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            src[ys[1], xs[1]] = covered[ys[0], xs[0]]
            shifted[ys[1], xs[1]] = color[ys[0], xs[0]]
            acc += shifted * src[..., None]
            count += src
        grow = ~covered & (count > 0)
        if not grow.any():
            break
        color[grow] = acc[grow] / count[grow][:, None]
        covered |= grow
    color[~covered] = fill
    return Image.fromarray(np.clip(np.round(color), 0, 255).astype(np.uint8))
