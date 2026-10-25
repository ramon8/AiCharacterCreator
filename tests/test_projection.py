"""Projection of a generated view onto the mesh UV, on hand-built meshes (no Blender, no GPU)."""

import numpy as np
from PIL import Image

from make.texturing.mesh import MeshData
from make.texturing.projection import ViewBake, merge_views, project_view, render_painted, surface_mask, to_texture
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


def test_what_earlier_views_painted_renders_back_from_their_camera():
    surface = mesh(quad(0.0))
    bake = project_view(surface, FRONT, quadrants(), bake_size=128)
    image, confidence = render_painted(surface, FRONT, bake.color, bake.weight)
    assert image.shape == (256, 256, 3) and confidence.shape == (256, 256)
    assert tuple(np.round(image[64, 64]).astype(int)) == RED
    assert tuple(np.round(image[64, 192]).astype(int)) == BLUE
    assert tuple(np.round(image[192, 64]).astype(int)) == GREEN
    assert tuple(np.round(image[192, 192]).astype(int)) == YELLOW
    assert confidence[128, 128] > 0.99


def test_parts_no_earlier_view_painted_have_no_confidence():
    surface = mesh(quad(0.0, x0=-0.5, x1=0.5))  # half the frame wide: the rest is background
    weight = np.zeros((64, 64))
    weight[:, :32] = 1.0  # only the left half of the UV (= left of the quad) is painted
    image, confidence = render_painted(surface, FRONT, np.full((64, 64, 3), 200.0), weight)
    assert confidence[128, 100] == 1.0  # left part of the quad
    assert confidence[128, 156] == 0.0  # right part, unpainted
    assert confidence[128, 10] == 0.0  # background


def test_sharper_blending_favours_the_view_that_faces_the_surface_most():
    def bake(color, w):
        return ViewBake(color=np.full((1, 1, 3), color, float), weight=np.full((1, 1), w))

    front, side = bake((100, 0, 0), 0.8), bake((0, 0, 0), 0.4)
    linear, _ = merge_views([front, side])
    sharp, _ = merge_views([front, side], sharpness=2)
    assert round(linear[0, 0, 0], 3) == 66.667
    assert round(sharp[0, 0, 0], 3) == 80.0


def test_surface_mask_is_where_the_uv_layout_has_triangles():
    half = quad(0.0, uv=((0, 0), (0.5, 0), (0.5, 1), (0, 1)))
    surface = surface_mask(mesh(half), 64)
    assert surface[:, :31].all() and not surface[:, 33:].any()


def test_surface_nobody_painted_is_filled_from_its_painted_neighbours_not_grey():
    color = np.zeros((64, 64, 3))
    color[:, :4] = (10, 200, 30)
    covered = np.zeros((64, 64), bool)
    covered[:, :4] = True
    surface = np.zeros((64, 64), bool)
    surface[:, :40] = True  # painted strip + 36 px of unpainted surface, beyond the padding
    texture = np.asarray(to_texture(color, covered, padding=2, surface=surface))
    assert tuple(texture[32, 39]) == (10, 200, 30)
    assert tuple(texture[32, 60]) == (128, 128, 128)  # off the surface, past the padding


def ridge():
    """Two quads meeting at a vertical ridge on X = 0 that points at the front camera,
    each turned 30 degrees away from it; UV: left quad u 0..0.5, right quad u 0.5..1."""
    a = np.radians(30)
    def side(sign):
        outer = np.array([sign * np.cos(a), np.sin(a), 0.0])
        normal = np.array([sign * np.sin(a), -np.cos(a), 0.0])
        corners = np.array([(0, 0, -1), outer - (0, 0, 1), outer + (0, 0, 1), (0, 0, 1)], float)
        if sign < 0:
            corners = corners[[1, 0, 3, 2]]
        u0 = 0.0 if sign < 0 else 0.5
        uv = np.array([(u0, 0), (u0 + 0.5, 0), (u0 + 0.5, 1), (u0, 1)], float)
        tris = [(0, 1, 2), (0, 2, 3)]
        return corners[tris], uv[tris], np.array([normal, normal])
    return mesh(side(-1), side(1))


def test_flat_weights_are_constant_across_a_face():
    bake = project_view(ridge(), FRONT, quadrants(), bake_size=64)
    near_ridge, near_outer = bake.weight[32, 30], bake.weight[32, 2]
    assert abs(near_ridge - np.cos(np.radians(30))) < 1e-6
    assert abs(near_outer - np.cos(np.radians(30))) < 1e-6


def test_smooth_weights_vary_across_the_face_so_views_hand_over_gradually():
    bake = project_view(ridge(), FRONT, quadrants(), bake_size=64, smooth_angle=70)
    near_ridge, near_outer = bake.weight[32, 31], bake.weight[32, 1]
    assert near_ridge > 0.99  # the welded normal on the ridge faces the camera
    assert abs(near_outer - np.cos(np.radians(30))) < 0.02
    sharp = project_view(ridge(), FRONT, quadrants(), bake_size=64, smooth_angle=45)
    assert abs(sharp.weight[32, 31] - np.cos(np.radians(30))) < 1e-6  # normals 60 degrees apart: a sharp edge
