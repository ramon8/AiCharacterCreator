"""Hits the real local ComfyUI. Run with: uv run pytest -m comfyui -s

Needs `uv run make comfyui start` first. Outputs are kept in
ACC_WORK_DIR/generations for visual review.
"""

import math
import time

import pytest
from PIL import Image, ImageChops, ImageDraw

from make import settings
from make.comfyui.availability import check
from make.generation import ComfyUIClient, Conditions

pytestmark = pytest.mark.comfyui

SIZE = 1024


def sphere_renders(size=SIZE, radius=0.42):
    """Depth (near = white) and tangent-space normal renders of a sphere, plus its mask."""
    depth = Image.new("RGB", (size, size))
    normal = Image.new("RGB", (size, size), (128, 128, 255))
    mask = Image.new("L", (size, size), 0)
    dp, np_, mp = depth.load(), normal.load(), mask.load()
    for y in range(size):
        ny = (0.5 - (y + 0.5) / size) / radius
        for x in range(size):
            nx = ((x + 0.5) / size - 0.5) / radius
            r2 = nx * nx + ny * ny
            if r2 >= 1:
                continue
            nz = math.sqrt(1 - r2)
            d = int(80 + 175 * nz)
            dp[x, y] = (d, d, d)
            np_[x, y] = (int((nx + 1) * 127.5), int((ny + 1) * 127.5), int((nz + 1) * 127.5))
            mp[x, y] = 255
    return depth, normal, mask


def reference_image():
    """A red and gold diagonal-striped fabric swatch."""
    image = Image.new("RGB", (512, 512), (170, 20, 30))
    draw = ImageDraw.Draw(image)
    for offset in range(-512, 1024, 64):
        draw.line([(offset, 0), (offset + 512, 512)], fill=(230, 180, 40), width=18)
    return image


@pytest.fixture(scope="module")
def client():
    check()
    return ComfyUIClient()


@pytest.fixture(scope="module")
def renders():
    return sphere_renders()


def conditions(renders, seed):
    depth, normal, mask = renders
    return Conditions(
        depth=depth, normal=normal, mask=mask, references=(reference_image(),),
        prompt="an ornate ball wrapped in red and gold striped silk fabric, studio lighting",
        seed=seed,
    )


def identical(a, b):
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None


def timed_uncached(client, conds):
    client.free()  # otherwise ComfyUI serves an identical graph from its cache
    time.sleep(1)
    started = time.perf_counter()
    image = client.generate(conds)
    return image, time.perf_counter() - started


def test_same_seed_and_workflow_give_the_same_image(client, renders):
    first, t1 = timed_uncached(client, conditions(renders, seed=7))
    second, t2 = timed_uncached(client, conditions(renders, seed=7))
    print(f"\nrun 1 {t1:.1f}s, run 2 {t2:.1f}s; outputs in {settings.work_dir() / 'generations'}")

    assert first.size == (SIZE, SIZE)
    assert t2 > 2, "second run came from ComfyUI's cache, not a recomputation"
    assert identical(first, second)


def test_other_seed_gives_another_image(client, renders):
    a = client.generate(conditions(renders, seed=7))
    b = client.generate(conditions(renders, seed=8))
    assert not identical(a, b)


def test_the_reference_image_conditions_the_result(client, renders):
    depth, normal, mask = renders
    blue = Image.new("RGB", (512, 512), (20, 60, 200))
    ImageDraw.Draw(blue).ellipse((128, 128, 384, 384), fill=(240, 240, 240))
    striped = client.generate(conditions(renders, seed=7))
    other = client.generate(Conditions(
        depth=depth, normal=normal, mask=mask, references=(blue,),
        prompt=conditions(renders, seed=7).prompt, seed=7,
    ))
    assert not identical(striped, other)
    sheet = Image.new("RGB", (SIZE * 4, SIZE), (255, 255, 255))
    for i, image in enumerate([depth, reference_image().resize((SIZE, SIZE)), striped, other]):
        sheet.paste(image.convert("RGB"), (i * SIZE, 0))
    sheet.save(settings.work_dir() / "generations" / "integration_sheet.png")
