from __future__ import annotations

import bpy


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def clear_collection(name: str) -> bpy.types.Collection:
    collection = ensure_collection(name)
    for obj in list(collection.objects):
        collection.objects.unlink(obj)
        if obj.users == 0:
            bpy.data.objects.remove(obj)
    for child in list(collection.children):
        collection.children.unlink(child)
        if child.users == 0:
            bpy.data.collections.remove(child)
    return collection


def set_active_object(context: bpy.types.Context, obj: bpy.types.Object | None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    if obj is None:
        return
    obj.select_set(True)
    context.view_layer.objects.active = obj


def generated_layer_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    state = scene.hallway_avatar_state
    collection = bpy.data.collections.get(state.imported_collection_name)
    if not collection:
        return []
    return list(collection.objects)
