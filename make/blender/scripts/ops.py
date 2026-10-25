"""Blender-side ops of the pipeline. Runs inside Blender 5.0 (bpy, bmesh, numpy only).

Invoked by `make.blender.run`:
    blender --background --factory-startup [file.blend] --python ops.py -- <op> <params.json>
Each op reads its params JSON and writes a JSON result to params["result"].
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

SHADER_SAMPLES = 16  # emission-only scenes: samples only antialias the edges
RENDER_SEED = 0


# --- helpers ---------------------------------------------------------------------------

def the_mesh() -> bpy.types.Object:
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected exactly one mesh object in the scene, found {[o.name for o in meshes]}")
    return meshes[0]


def node(tree, type_name, **props):
    n = tree.nodes.new(type_name)
    for key, value in props.items():
        setattr(n, key, value)
    return n


def emission_material(name: str, build_color) -> bpy.types.Material:
    """Material whose surface is an emission of strength 1; `build_color(tree)` returns
    the output socket that drives the emission color."""
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    tree.nodes.clear()
    out = node(tree, "ShaderNodeOutputMaterial")
    emit = node(tree, "ShaderNodeEmission")
    tree.links.new(build_color(tree), emit.inputs["Color"])
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def set_world_color(color) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    tree = world.node_tree
    if tree is None:
        world.use_nodes = True
        tree = world.node_tree
    background = next(n for n in tree.nodes if n.type == "BACKGROUND")
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = 1.0


def setup_render(resolution: int, view_transform: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SHADER_SAMPLES
    scene.cycles.seed = RENDER_SEED
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def place_camera(view: dict) -> None:
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        cam = bpy.data.objects.new("ProjectionCamera", bpy.data.cameras.new("ProjectionCamera"))
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = view["ortho_scale"]
    cam.data.sensor_fit = "AUTO"
    cam.data.shift_x = cam.data.shift_y = 0.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = view["distance"] * 4
    cam.matrix_world = Matrix(view["matrix_world"])


def render_to(path: str, material, world_color) -> None:
    scene = bpy.context.scene
    bpy.context.view_layer.material_override = material
    set_world_color(world_color)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# --- ops -------------------------------------------------------------------------------

def build_test_mesh(params: dict) -> dict:
    """A low-poly bust (tapered 10-sided torso + icosphere head with a nose), smart-UV'd."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.42, radius2=0.3, depth=0.9,
                          matrix=Matrix.Translation((0, 0, 0.45)))
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.26, matrix=Matrix.Translation((0, 0, 1.17)))
    bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.06, radius2=0.0, depth=0.12,
                          matrix=Matrix.Translation((0, -0.27, 1.15)) @ Matrix.Rotation(math.radians(90), 4, "X"))
    mesh = bpy.data.meshes.new("TestMesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("TestMesh", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.save_as_mainfile(filepath=params["out"], check_existing=False)
    return {"blend": params["out"], "triangles": sum(len(p.vertices) - 2 for p in mesh.polygons)}


def export_mesh(params: dict) -> dict:
    """World-space triangle soup with UVs and face normals, as .npz (see make.texturing.mesh)."""
    obj = the_mesh()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    if not mesh.uv_layers.active:
        raise RuntimeError(f"mesh {obj.name} has no UV map")
    tris = mesh.loop_triangles
    n = len(tris)
    tri_verts = np.empty(n * 3, np.int64)
    tri_loops = np.empty(n * 3, np.int64)
    tri_normals = np.empty(n * 3, np.float64)
    tris.foreach_get("vertices", tri_verts)
    tris.foreach_get("loops", tri_loops)
    tris.foreach_get("normal", tri_normals)
    co = np.empty(len(mesh.vertices) * 3, np.float64)
    mesh.vertices.foreach_get("co", co)
    uv = np.empty(len(mesh.loops) * 2, np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", uv)

    world = np.array(obj.matrix_world)
    co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    normals = tri_normals.reshape(-1, 3) @ normal_matrix.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    np.savez(params["out"], positions=co[tri_verts].reshape(n, 3, 3),
             uvs=uv.reshape(-1, 2)[tri_loops].reshape(n, 3, 2), normals=normals)
    evaluated.to_mesh_clear()
    return {"mesh": obj.name, "triangles": n}


def render_view(params: dict) -> dict:
    """Depth (near = white), normal (camera space, (n+1)/2, +Z toward the camera) and mask
    renders of the mesh from one orthographic view. Raw view transform: pixel = shader value."""
    view = params["view"]
    out = Path(params["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    setup_render(view["resolution"], "Raw")
    place_camera(view)

    def depth(tree):
        cam = node(tree, "ShaderNodeCameraData")
        remap = node(tree, "ShaderNodeMapRange", clamp=True)
        remap.inputs["From Min"].default_value = params["near"]
        remap.inputs["From Max"].default_value = params["far"]
        remap.inputs["To Min"].default_value = 1.0
        remap.inputs["To Max"].default_value = params.get("far_value", 0.25)
        tree.links.new(cam.outputs["View Z Depth"], remap.inputs["Value"])
        return remap.outputs["Result"]

    def normal(tree):
        geo = node(tree, "ShaderNodeNewGeometry")
        to_cam = node(tree, "ShaderNodeVectorTransform", vector_type="NORMAL", convert_from="WORLD",
                      convert_to="CAMERA")
        scale = node(tree, "ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        # Cycles camera space looks down +Z; flip Z so +Z points toward the camera.
        scale.inputs[1].default_value = (0.5, 0.5, -0.5)
        scale.inputs[2].default_value = (0.5, 0.5, 0.5)
        tree.links.new(geo.outputs["Normal"], to_cam.inputs["Vector"])
        tree.links.new(to_cam.outputs["Vector"], scale.inputs[0])
        return scale.outputs["Vector"]

    def white(tree):
        rgb = node(tree, "ShaderNodeRGB")
        rgb.outputs[0].default_value = (1, 1, 1, 1)
        return rgb.outputs[0]

    paths = {name: str(out / f"{name}.png") for name in ("depth", "normal", "mask")}
    render_to(paths["depth"], emission_material("acc_depth", depth), (0, 0, 0))
    render_to(paths["normal"], emission_material("acc_normal", normal), (0.5, 0.5, 1.0))
    render_to(paths["mask"], emission_material("acc_mask", white), (0, 0, 0))
    return paths


def pack_texture(params: dict) -> dict:
    """Assign the texture to the mesh (nearest sampling), pack it into the saved .blend,
    then render shadeless previews (not saved)."""
    obj = the_mesh()
    image = bpy.data.images.load(params["texture"])
    image.name = params.get("image_name", "psx_texture")
    image.pack()

    mat = bpy.data.materials.new(params.get("material_name", "PSXTexture"))
    tree = mat.node_tree
    principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    tex = node(tree, "ShaderNodeTexImage", image=image, interpolation="Closest")
    tree.links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 1.0
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.material_index = 0
    for cam in [o for o in bpy.context.scene.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(cam)
    bpy.ops.wm.save_as_mainfile(filepath=params["out"], check_existing=False, compress=True)

    previews = []
    if params.get("previews"):
        def textured(tree):
            t = node(tree, "ShaderNodeTexImage", image=image, interpolation="Closest")
            return t.outputs["Color"]
        shadeless = emission_material("acc_preview", textured)
        for preview in params["previews"]:
            setup_render(preview["view"]["resolution"], "Standard")
            place_camera(preview["view"])
            render_to(preview["path"], shadeless, (0.18, 0.18, 0.2))
            previews.append(preview["path"])
    return {"blend": params["out"], "previews": previews}


def inspect_textures(params: dict) -> dict:
    images = []
    for image in bpy.data.images:
        if image.type != "IMAGE":
            continue
        w, h = image.size
        pixels = np.empty(w * h * image.channels, np.float32)
        image.pixels.foreach_get(pixels)
        rgb = np.round(pixels.reshape(h, w, image.channels)[..., :3] * 255).astype(np.uint8)
        materials = [m for m in bpy.data.materials if m.node_tree and any(
            n.type == "TEX_IMAGE" and n.image == image for n in m.node_tree.nodes)]
        interpolation = sorted({n.interpolation for m in materials for n in m.node_tree.nodes
                                if n.type == "TEX_IMAGE" and n.image == image})
        users = sorted(o.name for o in bpy.data.objects if o.type == "MESH"
                       and any(s.material in materials for s in o.material_slots))
        images.append({
            "name": image.name, "size": [w, h], "packed": image.packed_file is not None,
            "colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
            "pixels_sha256": hashlib.sha256(rgb[::-1].tobytes()).hexdigest(),
            "interpolation": interpolation, "used_by_meshes": users,
        })
    return {"images": images}


OPS = {f.__name__: f for f in (build_test_mesh, export_mesh, render_view, pack_texture, inspect_textures)}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    op, params_file = argv[0], Path(argv[1])
    params = json.loads(params_file.read_text(encoding="utf-8"))
    result = OPS[op](params)
    Path(params["result"]).write_text(json.dumps(result), encoding="utf-8")


main()
