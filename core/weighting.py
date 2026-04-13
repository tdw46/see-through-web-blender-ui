from __future__ import annotations

import bpy
from mathutils import Vector

from ..utils.logging import get_logger
from .models import LayerPart

logger = get_logger("weighting")


def _ensure_armature_modifier(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    modifier = obj.modifiers.get("HallwayAvatarArmature")
    if modifier is None:
        modifier = obj.modifiers.new("HallwayAvatarArmature", "ARMATURE")
    modifier.object = armature_obj


def _clear_generated_groups(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    bone_names = {bone.name for bone in armature_obj.data.bones}
    for group in list(obj.vertex_groups):
        if group.name in bone_names:
            obj.vertex_groups.remove(group)


def _ensure_group(obj: bpy.types.Object, bone_name: str) -> bpy.types.VertexGroup:
    group = obj.vertex_groups.get(bone_name)
    if group is None:
        group = obj.vertex_groups.new(name=bone_name)
    return group


def _assign_rigid(obj: bpy.types.Object, bone_name: str) -> None:
    group = _ensure_group(obj, bone_name)
    indices = [vertex.index for vertex in obj.data.vertices]
    group.add(indices, 1.0, "REPLACE")


def _assign_two_bone_gradient(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    upper_bone: str,
    lower_bone: str,
) -> None:
    upper = armature_obj.data.bones[upper_bone]
    lower = armature_obj.data.bones[lower_bone]
    upper_group = _ensure_group(obj, upper_bone)
    lower_group = _ensure_group(obj, lower_bone)

    start = armature_obj.matrix_world @ upper.head_local
    end = armature_obj.matrix_world @ lower.tail_local
    axis = end - start
    axis_len_sq = max(axis.length_squared, 1e-8)

    for vertex in obj.data.vertices:
        world_co = obj.matrix_world @ vertex.co
        t = max(0.0, min(1.0, (world_co - start).dot(axis) / axis_len_sq))
        upper_group.add([vertex.index], 1.0 - t, "REPLACE")
        lower_group.add([vertex.index], t, "REPLACE")


def _assign_torso(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    groups = {
        "root": _ensure_group(obj, "root"),
        "torso": _ensure_group(obj, "torso"),
        "chest": _ensure_group(obj, "chest"),
    }

    z_values = [vertex.co.z for vertex in obj.data.vertices]
    min_z = min(z_values)
    max_z = max(z_values)
    span = max(1e-8, max_z - min_z)

    for vertex in obj.data.vertices:
        t = (vertex.co.z - min_z) / span
        root_weight = max(0.0, 0.5 - t)
        chest_weight = max(0.0, t - 0.5)
        torso_weight = 1.0 - abs(t - 0.5) * 2.0
        total = max(1e-8, root_weight + torso_weight + chest_weight)
        groups["root"].add([vertex.index], root_weight / total, "REPLACE")
        groups["torso"].add([vertex.index], torso_weight / total, "REPLACE")
        groups["chest"].add([vertex.index], chest_weight / total, "REPLACE")


def _binding_mode(part: LayerPart) -> tuple[str, tuple[str, ...]]:
    label = part.semantic_label
    if label == "arm" and part.side_guess in {"L", "R"}:
        label = f"arm_{part.side_guess.lower()}"
    if label == "leg" and part.side_guess in {"L", "R"}:
        label = f"leg_{part.side_guess.lower()}"
    if label == "foot" and part.side_guess in {"L", "R"}:
        label = f"foot_{part.side_guess.lower()}"
    if label == "torso" or label == "pelvis":
        return "torso", ("root", "torso", "chest")
    if label == "neck":
        return "rigid", ("neck",)
    if label == "head" or label.startswith("hair_"):
        return "rigid", ("head",)
    if label == "arm_l":
        return "gradient", ("upper_arm.L", "lower_arm.L")
    if label == "arm_r":
        return "gradient", ("upper_arm.R", "lower_arm.R")
    if label == "hand_l":
        return "rigid", ("hand.L",)
    if label == "hand_r":
        return "rigid", ("hand.R",)
    if label == "leg_l":
        return "gradient", ("upper_leg.L", "lower_leg.L")
    if label == "leg_r":
        return "gradient", ("upper_leg.R", "lower_leg.R")
    if label == "foot_l":
        return "rigid", ("foot.L",)
    if label == "foot_r":
        return "rigid", ("foot.R",)
    return "rigid", ("chest",)


def bind_parts(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    parts: list[LayerPart],
) -> None:
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None or obj.type != "MESH":
            continue

        _ensure_armature_modifier(obj, armature_obj)
        _clear_generated_groups(obj, armature_obj)

        mode, bones = _binding_mode(part)
        if mode == "torso":
            _assign_torso(obj, armature_obj)
        elif mode == "gradient":
            _assign_two_bone_gradient(obj, armature_obj, bones[0], bones[1])
        else:
            _assign_rigid(obj, bones[0])

    logger.info("Bound %s layer objects to %s", len(parts), armature_obj.name)
