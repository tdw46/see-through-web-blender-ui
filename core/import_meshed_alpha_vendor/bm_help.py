from typing import Iterable
from itertools import chain
from collections import deque

import bmesh
from bmesh.types import BMesh, BMVert, BMFace
from mathutils import Vector


def get_faces_from_verts(verts: Iterable[BMVert]) -> set[BMFace]:
    faces = set()
    for vert in verts:
        faces = faces.union(vert.link_faces)
    return faces


def calc_total_face_area(faces: Iterable[BMFace]) -> float:
    total = 0.0
    for face in faces:
        total += face.calc_area()
    return total

def get_vert_islands(target: BMesh) -> list[set[BMVert]]:
    def _walk_connected(init_vert: BMVert):
        island_verts: set[BMVert] = set()

        to_check = deque((init_vert,))
        while to_check:
            test_vert = to_check.pop()
            island_verts.add(test_vert)
            linked_edges = test_vert.link_edges
            edge_vert_pairs = (edge.verts for edge in linked_edges)
            connected_verts = set(chain.from_iterable(edge_vert_pairs))
            unchecked_connected = connected_verts - island_verts
            to_check.extend(unchecked_connected)
        return island_verts

    target.verts.ensure_lookup_table()

    islands: list[set[BMVert]] = []
    ungrouped_verts = set(target.verts)

    while ungrouped_verts:
        island_verts = _walk_connected(ungrouped_verts.pop())
        ungrouped_verts -= island_verts
        islands.append(island_verts)

    return islands


def calc_bbox(bm: bmesh.types.BMesh) -> tuple[Vector, Vector]:
    # Calc bounding box
    bound_min = [float("inf")] * 3
    bound_max = [float("-inf")] * 3

    for v in bm.verts:
        for i in range(3):
            bound_min[i] = min(bound_min[i], v.co[i])
            bound_max[i] = max(bound_max[i], v.co[i])

    return Vector(bound_min), Vector(bound_max)

