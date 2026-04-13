from __future__ import annotations

import bpy
from bpy.types import Operator

from ..utils import env


class HALLWAYAVATAR_OT_install_psd_backend(Operator):
    bl_idname = "hallway_avatar.install_psd_backend"
    bl_label = "Install PSD Backend"
    bl_description = "Install bundled psd-tools, Pillow, and silhouette tracing wheels into the extension-local vendor directory"

    def execute(self, context: bpy.types.Context):
        try:
            message = env.install_bundled_psd_backend()
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_install_psd_backend,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
