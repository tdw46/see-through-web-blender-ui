from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator


def _resolve_context_path(context: bpy.types.Context, data_path: str):
    target = context
    for part in data_path.split("."):
        target = getattr(target, part)
    return target


class HALLWAYAVATAR_OT_set_bool(Operator):
    bl_idname = "hallway_avatar.set_bool"
    bl_label = "Set Toggle"
    bl_options = {"INTERNAL"}

    data_path: StringProperty(name="Data Path")
    prop_name: StringProperty(name="Property Name")
    value: BoolProperty(name="Value")

    @classmethod
    def description(cls, context: bpy.types.Context, properties) -> str:
        try:
            owner = _resolve_context_path(context, properties.data_path)
            prop = owner.bl_rna.properties[properties.prop_name]
            prop_desc = getattr(prop, "description", "") or prop.name
            state = "ON" if properties.value else "OFF"
            return f"Set {prop.name} to {state}. {prop_desc}"
        except Exception:
            state = "ON" if getattr(properties, "value", False) else "OFF"
            return f"Set this option to {state}."

    def execute(self, context: bpy.types.Context):
        owner = _resolve_context_path(context, self.data_path)
        setattr(owner, self.prop_name, self.value)
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_set_bool,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
