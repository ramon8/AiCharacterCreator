"""Projection of a generated view onto the mesh UV, on hand-built meshes (no Blender, no GPU)."""

import numpy as np
from PIL import Image

from make.texturing.mesh import MeshData
from make.texturing.projection import project_view
from make.texturing.views import View

RED, BLUE, GREEN, YELLOW = (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)


def quad(y, x0=-1.0, x1=1.0, z0=-1.0, z1=1.0, uv=((0, 0), (1, 0), (1, 1), (0, 1)), normal=(0, -1, 0)):
    """An axis-aligned quad in the plane Y = y, as two triangles."""
    corners = np.array([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], float)
    uv = np.array(uv, float)
    tris = [(0, 1, 2), (0, 2, 3)]
    return corners[tris], uv[tris], np.array([normal, normal], float)


def mesh(*quads):
    positions, uvs, normals = (np.concatenate(parts) for parts in zip(*quads))
    return MeshData(positions=positions, uvs=uvs, normals=normals)


def quadrants(size=256):
    """Top-left red, top-right blue, bottom-left green, bottom-right yellow."""
    image = Image.new("RGB", (size, size), RED)
    image.paste(BLUE, (size // 2, 0, size, size // 2))
    image.paste(GREEN, (0, size // 2, size // 2, size))
    image.paste(YELLOW, (size // 2, size // 2, size, size))
    return image


FRONT = View(name="front", target=(0.0, 0.0, 0.0), azimuth=0.0, elevation=0.0, ortho_scale=2.0, resolution=256)


def texel(bake, u, v):
    """Bake texel at UV (u, v), v up as in Blender."""
    h, w = bake.weight.shape
    return int(u * w), int((1 - v) * h)


def color_at(bake, u, v):
    x, y = texel(bake, u, v)
    return tuple(np.round(bake.color[y, x]).astype(int))


def test_front_view_lands_on_the_uv_with_the_same_orientation():
    bake = project_view(mesh(quad(0.0)), FRONT, quadrants(), bake_size=64)
    assert color_at(bake, 0.25, 0.75) == RED
    assert color_at(bake, 0.75, 0.75) == BLUE
    assert color_at(bake, 0.25, 0.25) == GREEN
    assert color_at(bake, 0.75, 0.25) == YELLOW
    assert bake.weight[texel(bake, 0.5, 0.5)[::-1]] > 0.99


def test_faces_turned_away_from_the_camera_are_not_painted():
    bake = project_view(mesh(quad(0.0, normal=(0, 1, 0))), FRONT, quadrants(), bake_size=64)
    assert not bake.weight.any()


def test_faces_hidden_behind_others_are_not_painted():
    occluder = quad(-0.5, x0=-1.0, x1=0.0, uv=((0, 0),) * 4)  # covers the left half; zero UV area
    bake = project_view(mesh(quad(0.0), occluder), FRONT, quadrants(), bake_size=64)
    left, right = texel(bake, 0.25, 0.5), texel(bake, 0.75, 0.5)
    assert bake.weight[left[::-1]] == 0
    assert bake.weight[right[::-1]] > 0.99


def test_oblique_faces_weigh_less_than_frontal_ones():
    # A quad turned 60 degrees around Z: facing the camera by cos(60) = 0.5.
    a = np.radians(60)
    tangent, normal = np.array([np.cos(a), np.sin(a), 0.0]), np.array([np.sin(a), -np.cos(a), 0.0])
    corners = np.array([-0.5 * tangent - (0, 0, 1), 0.5 * tangent - (0, 0, 1),
                        0.5 * tangent + (0, 0, 1), -0.5 * tangent + (0, 0, 1)])
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
    tris = [(0, 1, 2), (0, 2, 3)]
    tilted = MeshData(positions=corners[tris], uvs=uv[tris], normals=np.array([normal, normal]))
    bake = project_view(tilted, FRONT, quadrants(), bake_size=64)
    assert abs(bake.weight[texel(bake, 0.5, 0.5)[::-1]] - 0.5) < 1e-6
