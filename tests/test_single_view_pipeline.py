"""Single-view texturing end to end with real Blender and a fake generation client.

The produced .blend is inspected through Blender (stand-in for the PSX
contract validator, #2). Skipped when Blender is not installed.
"""

import numpy as np
import pytest
from PIL import Image

from make import blender
from make.generation import Conditions, FakeGenerationClient
from make.texturing import pipeline
from make.texturing.testmesh import build_test_mesh

pytestmark = pytest.mark.skipif(not blender.available(), reason="Blender not installed")

RESOLUTION = 256  # conditioning render size; small to keep the test fast


class EchoMask:
    """Generation client double that "generates" the object mask it is given:
    white exactly where the camera saw the mesh, black elsewhere."""

    def generate(self, conditions: Conditions) -> Image.Image:
        return conditions.mask.convert("RGB")


@pytest.fixture(scope="module")
def mesh_blend(tmp_path_factory):
    return build_test_mesh(tmp_path_factory.mktemp("mesh") / "test_mesh.blend")


@pytest.fixture(scope="module")
def reference():
    return Image.new("RGB", (64, 64), (200, 30, 40))


def run(mesh_blend, reference, work, client=None, seed=7):
    return pipeline.texture_single_view(
        mesh_blend=mesh_blend, reference=reference, prompt="test", seed=seed,
        out_blend=work / "textured.blend", client=client or FakeGenerationClient(color=(30, 160, 90)),
        work_dir=work, resolution=RESOLUTION,
    )


def test_texture_is_psx_and_packed_in_the_blend(mesh_blend, reference, tmp_path):
    result = run(mesh_blend, reference, tmp_path)
    report = blender.inspect_textures(result.blend)
    assert len(report["images"]) == 1
    image = report["images"][0]
    assert image["size"] == [128, 128]
    assert image["packed"]
    assert image["colors"] <= 256
    assert image["interpolation"] == ["Closest"]
    assert image["used_by_meshes"] == ["TestMesh"]
    assert result.preview.exists()


def test_generated_color_is_on_the_mesh(mesh_blend, reference, tmp_path):
    result = run(mesh_blend, reference, tmp_path)
    texture = np.asarray(Image.open(result.texture).convert("RGB")).reshape(-1, 3)
    painted = (texture == (30, 160, 90)).all(axis=1).mean()
    assert painted > 0.1, "the front view should paint a good part of the UV"


def test_projection_is_aligned_with_the_rendered_geometry(mesh_blend, reference, tmp_path):
    bake = run(mesh_blend, reference, tmp_path, client=EchoMask()).bakes["front"]
    seen = bake.weight > 0.3
    assert seen.sum() > 1000
    sampled = bake.color[seen].min(axis=1)
    # Only the antialiased 1 px border of the silhouette may be partly grey; nothing the
    # camera sees may sample the background.
    assert (sampled < 5).sum() == 0
    assert (sampled > 128).mean() > 0.99
    assert (sampled > 250).mean() > 0.9


def test_same_seed_and_workflow_give_the_same_texture(mesh_blend, reference, tmp_path):
    first = blender.inspect_textures(run(mesh_blend, reference, tmp_path / "a").blend)
    second = blender.inspect_textures(run(mesh_blend, reference, tmp_path / "b").blend)
    assert first["images"][0]["pixels_sha256"] == second["images"][0]["pixels_sha256"]
