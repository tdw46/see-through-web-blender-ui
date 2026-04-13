from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_bind_weights(Operator):
    bl_idname = "hallway_avatar.bind_weights"
    bl_label = "Bind Weights"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        self.report({"INFO"}, "Weight binding is part of the later 2.5-D generation phase and is not exposed in this importer-first release.")
        return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_bind_weights,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
