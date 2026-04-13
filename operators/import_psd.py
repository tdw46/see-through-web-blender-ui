from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..core import pipeline


class HALLWAYAVATAR_OT_import_psd(Operator, ImportHelper):
    bl_idname = "hallway_avatar.import_psd"
    bl_label = "Import PSD Avatar"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".psd"
    filter_glob: StringProperty(default="*.psd", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context):
        try:
            pipeline.import_psd_scene(context, self.filepath)
            self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_import_psd,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
