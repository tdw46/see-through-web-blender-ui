from __future__ import annotations

import bpy

from .. import properties
from ..utils import blender as blender_utils
from ..utils import env
from ..utils.logging import get_logger
from . import alpha_mesh_adapter, armature_builder, heuristic_rigger, part_classifier, psd_io, weighting

logger = get_logger("pipeline")
ADDON_ID = env.addon_package_id(__package__)


def _cache_dir_from_context(context: bpy.types.Context) -> str:
    addon = context.preferences.addons.get(ADDON_ID)
    if not addon:
        return ""
    prefs = addon.preferences
    return getattr(prefs, "cache_dir", "")


def import_psd_scene(context: bpy.types.Context, filepath: str) -> list:
    scene = context.scene
    state = scene.hallway_avatar_state
    cache_dir = _cache_dir_from_context(context)

    parts = psd_io.load_psd_layer_parts(
        filepath,
        ignore_hidden_layers=state.ignore_hidden_layers,
        ignore_empty_layers=state.ignore_empty_layers,
        min_visible_pixels=state.min_visible_pixels,
        keep_tiny_named_parts=state.keep_tiny_named_parts,
        configured_cache_dir=cache_dir,
    )
    part_classifier.classify_parts(parts)

    collection = blender_utils.clear_collection(state.imported_collection_name) if state.replace_existing else blender_utils.ensure_collection(state.imported_collection_name)

    for part in parts:
        if part.skipped:
            continue
        obj = alpha_mesh_adapter.build_layer_mesh(
            context,
            part,
            collection,
            grid_resolution=state.mesh_grid_resolution,
        )
        part.imported_object_name = obj.name
        obj["hallway_avatar_semantic_label"] = part.semantic_label
        obj["hallway_avatar_side_guess"] = part.side_guess
        obj["hallway_avatar_confidence"] = part.confidence

    state.source_psd_path = filepath
    properties.set_layer_items(scene, parts)
    state.last_report = f"Imported {state.imported_count} layers, skipped {state.skipped_count}"
    logger.info(state.last_report)
    return parts


def reclassify_scene(context: bpy.types.Context) -> list:
    scene = context.scene
    parts = properties.get_parts(scene)
    part_classifier.classify_parts(parts)
    properties.set_layer_items(scene, parts)
    scene.hallway_avatar_state.last_report = f"Classified {scene.hallway_avatar_state.classified_count} layers"
    return parts


def build_armature_scene(context: bpy.types.Context, *, bind_weights: bool = False):
    scene = context.scene
    state = scene.hallway_avatar_state
    parts = properties.get_parts(scene)
    if not parts:
        raise RuntimeError("No imported layers found. Import a PSD first.")

    part_classifier.classify_parts(parts)
    properties.set_layer_items(scene, parts)

    rig_plan = heuristic_rigger.estimate_rig(parts)
    if not rig_plan.bones:
        raise RuntimeError("Unable to estimate a rig from the current layers.")

    if state.replace_existing:
        blender_utils.clear_collection(state.rig_collection_name)

    armature_obj = armature_builder.build_armature(context, rig_plan, state.rig_collection_name)
    state.armature_object_name = armature_obj.name

    if bind_weights:
        weighting.bind_parts(context, armature_obj, parts)

    state.last_report = f"Built rig with {len(rig_plan.bones)} bones (confidence {rig_plan.confidence:.2f})"
    logger.info(state.last_report)
    return armature_obj, rig_plan


def bind_weights_scene(context: bpy.types.Context) -> None:
    scene = context.scene
    state = scene.hallway_avatar_state
    armature_obj = bpy.data.objects.get(state.armature_object_name)
    if armature_obj is None:
        raise RuntimeError("No generated armature found. Build the armature first.")

    parts = properties.get_parts(scene)
    weighting.bind_parts(context, armature_obj, parts)
    state.last_report = f"Bound {len([part for part in parts if not part.skipped])} layers to {armature_obj.name}"
