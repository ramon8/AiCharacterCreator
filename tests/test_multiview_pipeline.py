"""Multiview texturing end to end with real Blender and a fake generation client.

Skipped when Blender is not installed.
"""

import numpy as np
import pytest
from PIL import Image

from make import blender
from make.generation import FakeGenerationClient
from make.texturing import pipeline
from make.texturing.mesh import MeshData
from make.texturing.projection import surface_mask
from make.texturing.testmesh import build_test_mesh

pytestmark = pytest.mark.skipif(not blender.available(), reason="Blender not installed")

RESOLUTION = 256  # conditioning render size; small to keep the test fast
PAINT = (30, 160, 90)


@pytest.fixture(scope="module")
def mesh_blend(tmp_path_factory):
    return build_test_mesh(tmp_path_factory.mktemp("mesh") / "test_mesh.blend")


@pytest.fixture(scope="module")
def reference():
    return Image.new("RGB", (64, 64), (200, 30, 40))


def run(mesh_blend, reference, work, client=None):
    client = client or FakeGenerationClient(color=PAINT)
    result = pipeline.texture_multiview(
        mesh_blend=mesh_blend, reference=reference, prompt="test", seed=7,
        out_blend=work / "textured.blend", client=client, work_dir=work, resolution=RESOLUTION,
    )
    return result, client


@pytest.fixture(scope="module")
def textured(mesh_blend, reference, tmp_path_factory):
    return run(mesh_blend, reference, tmp_path_factory.mktemp("multiview"))


def test_no_surface_texel_is_left_unpainted(textured):
    result, _ = textured
    mesh = MeshData.load(result.blend.parent / "mesh.npz")
    texture = np.asarray(Image.open(result.texture).convert("RGB"))
    surface = surface_mask(mesh, texture.shape[0])
    assert surface.sum() > 1000
    assert (texture[surface] == PAINT).all(axis=1).all()


def test_texture_is_psx_and_packed_in_the_blend(textured):
    result, _ = textured
    [image] = blender.inspect_textures(result.blend)["images"]
    assert image["size"] == [128, 128]
    assert image["packed"]
    assert image["colors"] <= 256
    assert image["interpolation"] == ["Closest"]
    assert image["used_by_meshes"] == ["TestMesh"]


def test_the_first_view_is_generated_fresh_and_the_rest_continue_from_it(textured, reference):
    result, client = textured
    first, *rest = client.calls
    assert len(client.calls) == len(result.views) >= 4
    assert first.init is None and first.references == (reference,)
    for call in rest:
        assert call.init is not None and call.repaint is not None
        own_reference, first_view = call.references
        assert own_reference is reference
        # The first generated view (all PAINT in this fake) is the extra reference.
        w, h = first_view.size
        assert first_view.convert("RGB").getpixel((w // 2, h // 2)) == PAINT


def test_a_continuing_view_keeps_what_earlier_views_saw_well_and_generates_the_rest(textured):
    _, client = textured
    side = client.calls[1]
    repaint = np.asarray(side.repaint.convert("L"))
    on_mesh = np.asarray(side.mask.convert("L")) > 128
    assert (repaint[on_mesh] < 10).mean() > 0.05, "faces the first view saw head-on are kept"
    assert (repaint[on_mesh] > 245).mean() > 0.05, "faces no earlier view saw are generated"
    painted = np.asarray(side.init.convert("RGB"))[repaint < 10]
    assert (painted == PAINT).all(axis=1).mean() > 0.95


def test_same_seed_and_workflow_give_the_same_texture(textured, mesh_blend, reference, tmp_path):
    first, _ = textured
    second, _ = run(mesh_blend, reference, tmp_path)
    a = blender.inspect_textures(first.blend)["images"][0]["pixels_sha256"]
    b = blender.inspect_textures(second.blend)["images"][0]["pixels_sha256"]
    assert a == b


def test_each_view_prompt_says_which_side_of_the_character_it_shows(textured):
    _, client = textured
    prompts = [call.prompt for call in client.calls]
    assert all(p.startswith("test") for p in prompts)
    assert "front view" in prompts[0]
    assert "back view" in prompts[3] and "front" not in prompts[3]
