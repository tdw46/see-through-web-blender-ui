from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Literal

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def _convert_hex_color(value: str) -> float:
    srgb = int(value, 16) / 255.0
    return ((srgb + 0.055) / 1.055) ** 2.4


def _hex_string_to_color(hex_str: str = "FFFFFF") -> tuple[float, float, float, float]:
    return (
        _convert_hex_color(hex_str[:2]),
        _convert_hex_color(hex_str[2:4]),
        _convert_hex_color(hex_str[4:6]),
        1.0,
    )


@dataclass
class CoordSequence:
    origin: np.ndarray
    knots: np.ndarray
    fill: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    control_points_a: np.ndarray | None = None
    control_points_b: np.ndarray | None = None
    coord_type: Literal["LINEAR", "CURVE"] = field(init=False)
    npoints: int = field(init=False)

    def __post_init__(self) -> None:
        self.coord_type = "LINEAR" if self.control_points_a is None else "CURVE"
        self.npoints = self.knots.shape[0]


Shape = list[CoordSequence]


@dataclass
class ParsedTrace:
    dimensions: tuple[int, int]
    shapes: list[Shape]


def _parse_path_line(line: str, fill: str, transform: str) -> list[CoordSequence]:
    offset_x_str, offset_y_str = transform.split("(")[-1].rstrip(")").split(",")
    offset = (float(offset_x_str), -float(offset_y_str), 0.0)

    line = line.strip().lower()
    is_linear = len(line) > 2 and line[2] == ","
    if is_linear:
        line = line.replace(",", " ")

    n_coord_points = 1 if is_linear else 3
    split_char = "l" if is_linear else "c"
    coord_sequences: list[CoordSequence] = []

    for path_section in line.split("z"):
        path_section = path_section.strip().replace(split_char, "")
        if not path_section:
            continue

        arr = np.fromstring(path_section[1:], dtype=np.float64, sep=" ")
        if is_linear:
            origin = arr[:2]
        else:
            origin, arr = np.split(arr, [2])
            arr = np.roll(arr, 4)

        n_values = arr.shape[0]
        arr = arr.reshape(n_values // 2, 2)
        arr *= np.array((1.0, -1.0))
        arr += np.array(offset[:2])

        z = np.zeros((arr.shape[0], 1), dtype=np.float64)
        arr = np.hstack((arr, z))
        arr = np.reshape(arr, (arr.shape[0] // n_coord_points, n_coord_points, 3))

        if is_linear:
            coord_sequences.append(
                CoordSequence(
                    origin=origin,
                    knots=arr,
                    fill=_hex_string_to_color(fill),
                    offset=offset,
                )
            )
        else:
            coord_sequences.append(
                CoordSequence(
                    origin=origin,
                    knots=arr[:, 1, :],
                    control_points_a=arr[:, 0, :],
                    control_points_b=arr[:, 2, :],
                    fill=_hex_string_to_color(fill),
                    offset=offset,
                )
            )

    return coord_sequences


def parse_trace(data: str) -> ParsedTrace:
    tree = ET.ElementTree(ET.fromstring(data))
    root = tree.getroot()
    dimensions = (int(root.get("width")), int(root.get("height")))

    shapes: list[Shape] = []
    for child in root:
        path_string = child.attrib.get("d", "")
        if not path_string:
            continue
        fill = child.attrib.get("fill", "#FFFFFF").lstrip("#")
        transform = child.attrib.get("transform", "translate(0,0)")
        shapes.append(_parse_path_line(path_string, fill, transform))

    return ParsedTrace(dimensions=dimensions, shapes=shapes)


def create_bmesh(trace_result: ParsedTrace) -> bmesh.types.BMesh:
    curves = _create_curves(trace_result)
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    uv_scale = Vector((1.0 / float(trace_result.dimensions[0]), 1.0 / float(trace_result.dimensions[1])))
    uv_offset = Vector((0.5, 0.5))

    for curve in curves:
        curve.data.offset = -0.00001
        mesh = curve.to_mesh()
        bm.from_mesh(mesh)
        curve.to_mesh_clear()
        curve_data = curve.data
        bpy.data.objects.remove(curve)
        bpy.data.curves.remove(curve_data)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.001)

    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv = loop.vert.co.xy * uv_scale + uv_offset

    return bm


def _create_curves(trace_result: ParsedTrace) -> list[bpy.types.Object]:
    new_objects: list[bpy.types.Object] = []
    origin_offset = np.array(
        (-trace_result.dimensions[0] / 2.0, trace_result.dimensions[1] / 2.0, 0.0),
        dtype=np.float64,
    )

    for index, shape in enumerate(trace_result.shapes):
        if not shape:
            continue
        data = bpy.data.curves.new(str(index).zfill(4), "CURVE")
        data.dimensions = "2D"
        data.fill_mode = "FRONT"
        obj = bpy.data.objects.new(data.name, object_data=data)
        obj.color = shape[0].fill
        new_objects.append(obj)

        for path in shape:
            is_linear = path.coord_type == "LINEAR"
            spline = data.splines.new("POLY" if is_linear else "BEZIER")
            spline.use_cyclic_u = True
            n_points = path.npoints

            if is_linear:
                spline.points.add(n_points - 1)
                knots = path.knots + origin_offset
                padding = np.ones((n_points, 1, 1), dtype=np.float64)
                padded = np.dstack((knots, padding))
                spline.points.foreach_set("co", padded.ravel())
            else:
                spline.bezier_points.add(n_points - 1)
                knots = path.knots + origin_offset
                control_a = path.control_points_a + origin_offset
                control_b = path.control_points_b + origin_offset
                spline.bezier_points.foreach_set("co", knots.ravel())
                spline.bezier_points.foreach_set("handle_left", control_a.ravel())
                spline.bezier_points.foreach_set("handle_right", control_b.ravel())

    return new_objects
