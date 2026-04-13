from __future__ import annotations

import bpy

from ..utils.logging import get_logger
from .models import LayerPart

logger = get_logger("alpha_mesh_adapter")


def _pixel_to_plane(x: float, y: float, canvas_size: tuple[int, int]) -> tuple[float, float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    canvas_w, canvas_h = canvas_size
    return (
        (x - canvas_w * 0.5) * scale,
        0.0,
        (canvas_h * 0.5 - y) * scale,
    )


def _uv_bounds(part: LayerPart) -> tuple[float, float, float, float]:
    image_w = max(1, part.image_size[0])
    image_h = max(1, part.image_size[1])
    local = part.local_alpha_bbox
    u0 = local[0] / image_w
    u1 = local[2] / image_w
    v_top = 1.0 - (local[1] / image_h)
    v_bottom = 1.0 - (local[3] / image_h)
    return (u0, u1, v_bottom, v_top)


def _build_grid_geometry(
    part: LayerPart,
    resolution: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]], int, int]:
    bbox = part.alpha_bbox if any(part.alpha_bbox) else (
        part.canvas_offset[0],
        part.canvas_offset[1],
        part.canvas_offset[0] + part.image_size[0],
        part.canvas_offset[1] + part.image_size[1],
    )
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    x_steps = max(1, min(resolution, width))
    y_steps = max(1, min(resolution, height))

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []

    for row in range(y_steps + 1):
        y = y0 + (height * row / y_steps)
        for col in range(x_steps + 1):
            x = x0 + (width * col / x_steps)
            verts.append(_pixel_to_plane(x, y, part.canvas_size))

    for row in range(y_steps):
        for col in range(x_steps):
            base = row * (x_steps + 1) + col
            faces.append((base, base + 1, base + x_steps + 2, base + x_steps + 1))

    return verts, faces, x_steps, y_steps


def _ensure_image_material(part: LayerPart) -> bpy.types.Material:
    material_name = f"HAVATAR_MAT_{part.layer_name}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)

    material.use_nodes = True
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.interpolation = "Closest"
    tex.image = bpy.data.images.load(part.temp_image_path, check_existing=True) if part.temp_image_path else None

    output.location = (300, 0)
    bsdf.location = (60, 0)
    tex.location = (-240, 0)

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_layer_mesh(
    context: bpy.types.Context,
    part: LayerPart,
    collection: bpy.types.Collection,
    *,
    grid_resolution: int = 12,
) -> bpy.types.Object:
    verts, faces, x_steps, y_steps = _build_grid_geometry(part, grid_resolution)
    mesh = bpy.data.meshes.new(f"{part.layer_name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    u0, u1, v0, v1 = _uv_bounds(part)
    face_index = 0
    for row in range(y_steps):
        for col in range(x_steps):
            if face_index >= len(mesh.polygons):
                break
            face = mesh.polygons[face_index]
            u_left = u0 + (u1 - u0) * (col / x_steps)
            u_right = u0 + (u1 - u0) * ((col + 1) / x_steps)
            v_top = v1 - (v1 - v0) * (row / y_steps)
            v_bottom = v1 - (v1 - v0) * ((row + 1) / y_steps)
            coords = ((u_left, v_top), (u_right, v_top), (u_right, v_bottom), (u_left, v_bottom))
            for loop_offset, loop_index in enumerate(face.loop_indices):
                uv_layer.data[loop_index].uv = coords[loop_offset]
            face_index += 1

    material = _ensure_image_material(part)
    if mesh.materials:
        mesh.materials[0] = material
    else:
        mesh.materials.append(material)

    obj_name = f"{part.draw_index:03d}_{part.layer_name}"
    obj = bpy.data.objects.new(obj_name, mesh)
    collection.objects.link(obj)
    obj["hallway_avatar_generated"] = True
    obj["hallway_avatar_layer_path"] = part.layer_path
    obj["hallway_avatar_layer_name"] = part.layer_name
    obj["hallway_avatar_semantic_label"] = part.semantic_label
    obj["hallway_avatar_side_guess"] = part.side_guess
    obj["hallway_avatar_confidence"] = part.confidence
    return obj
