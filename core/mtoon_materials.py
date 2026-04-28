from __future__ import annotations

from pathlib import Path

import bpy

from ..utils.logging import get_logger
from .models import LayerPart

logger = get_logger("mtoon_materials")


def _material_image(material: bpy.types.Material, part: LayerPart) -> bpy.types.Image | None:
    temp_image_path = part.temp_image_path or ""
    if temp_image_path and Path(temp_image_path).exists():
        try:
            return bpy.data.images.load(temp_image_path, check_existing=True)
        except Exception as exc:
            logger.warning("Unable to load MToon texture image %s: %s", temp_image_path, exc)

    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            image = getattr(node, "image", None)
            if image is not None:
                return image
    return None


def _mtoon_alpha_transparent_identifier(mtoon1: object) -> str:
    enum = getattr(mtoon1, "alpha_mode_enum", None)
    identifiers = set(enum.identifiers()) if enum is not None and hasattr(enum, "identifiers") else set()
    if "TRANSPARENT" in identifiers:
        return "TRANSPARENT"
    if "BLEND" in identifiers:
        return "BLEND"
    blend = getattr(mtoon1, "ALPHA_MODE_BLEND", None)
    blend_identifier = getattr(blend, "identifier", None)
    if isinstance(blend_identifier, str):
        return blend_identifier
    return "TRANSPARENT"


def _safe_set_mtoon_texture(texture_info: object, image: bpy.types.Image) -> None:
    index = getattr(texture_info, "index", None)
    if index is not None and hasattr(index, "source"):
        index.source = image


def _setup_mtoon_material(
    material: bpy.types.Material,
    image: bpy.types.Image,
    render_queue_offset: int,
) -> bool:
    extension = getattr(material, "vrm_addon_extension", None)
    if extension is None or not hasattr(extension, "mtoon1"):
        logger.info("VRM material extension unavailable for %s; skipping MToon setup", material.name)
        return False

    mtoon1 = extension.mtoon1
    try:
        mtoon1.enabled = True
    except Exception as exc:
        logger.warning("Unable to enable MToon1 on %s: %s", material.name, exc)
        return False

    mtoon = mtoon1.extensions.vrmc_materials_mtoon
    alpha_identifier = _mtoon_alpha_transparent_identifier(mtoon1)
    try:
        mtoon1.alpha_mode = alpha_identifier
    except TypeError:
        if hasattr(mtoon1, "ALPHA_MODE_BLEND"):
            mtoon1.alpha_mode = mtoon1.ALPHA_MODE_BLEND.identifier
    mtoon1.pbr_metallic_roughness.base_color_factor = (1.0, 1.0, 1.0, 1.0)
    _safe_set_mtoon_texture(mtoon1.pbr_metallic_roughness.base_color_texture, image)

    mtoon.shade_color_factor = (1.0, 1.0, 1.0)
    _safe_set_mtoon_texture(mtoon.shade_multiply_texture, image)
    mtoon.render_queue_offset_number = max(-9, min(9, int(render_queue_offset)))
    material["hallway_avatar_mtoon_render_queue_offset"] = mtoon.render_queue_offset_number
    return True


def _render_queue_offsets(objects: list[bpy.types.Object]) -> dict[str, int]:
    if not objects:
        return {}
    ordered = sorted(objects, key=lambda obj: obj.matrix_world.translation.y, reverse=True)
    if len(ordered) == 1:
        return {ordered[0].name: 0}
    offsets: dict[str, int] = {}
    for rank, obj in enumerate(ordered):
        offsets[obj.name] = round(-9 + (18 * rank / (len(ordered) - 1)))
    return offsets


def configure_avatar_mtoon_materials(parts: list[LayerPart]) -> int:
    objects: list[bpy.types.Object] = []
    part_by_object_name: dict[str, LayerPart] = {}
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None or obj.type != "MESH":
            continue
        objects.append(obj)
        part_by_object_name[obj.name] = part

    offsets = _render_queue_offsets(objects)
    configured = 0
    seen_materials: set[str] = set()
    for obj in objects:
        part = part_by_object_name[obj.name]
        render_queue_offset = offsets.get(obj.name, 0)
        for material in obj.data.materials:
            if material is None or not material.name.startswith("HAVATAR_MAT_"):
                continue
            if material.name in seen_materials:
                continue
            image = _material_image(material, part)
            if image is None:
                logger.info("No image found for avatar material %s", material.name)
                continue
            if _setup_mtoon_material(material, image, render_queue_offset):
                seen_materials.add(material.name)
                configured += 1

    logger.info("Configured %s avatar MToon materials", configured)
    return configured
