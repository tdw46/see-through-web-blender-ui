from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_build_armature(Operator):
    bl_idname = "hallway_avatar.build_armature"
    bl_label = "Build Armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        self.report({"INFO"}, "2.5-D generation and rigging are coming later. This version currently focuses on importing See-through layers.")
        return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_build_armature,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
