from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_classify_parts(Operator):
    bl_idname = "hallway_avatar.classify_parts"
    bl_label = "Classify Parts"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            pipeline.reclassify_scene(context)
            self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_classify_parts,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
