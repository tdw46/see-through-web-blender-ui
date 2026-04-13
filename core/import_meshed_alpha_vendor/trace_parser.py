"""
Optimized parsing of VTracer output. Will not do anything worthwhile given other input.

This is specifically for the output of vtracer and uses the following assumptions
    Per file, path interpolation prefix will always be either L or C
    Point sequences start with M (move) and end with Z (close path), Unsure if C paths need M so they don't use it, but L require it.
    Path per line
    All paths are closed

I think exterior paths are also typically clockwise where interior are ccw, but likely not relevant given other assumptions
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import bpy
import bmesh
from mathutils import Vector

from .bm_help import calc_bbox

def _convert_hex_color(value: str):
    """ Convert length 2 hex string to linear color float """
    srgb = int(value, 16) / 255
    return ((srgb + 0.055) / 1.055) ** 2.4


def _hex_string_to_color(hex_str: str = "FFFFFF") -> tuple[float, float, float, float]:
    """ Convert length 3 color hex string to length 4 lin space float with a white alpha """
    return (
        _convert_hex_color(hex_str[:2]),
        _convert_hex_color(hex_str[2:4]),
        _convert_hex_color(hex_str[4:]),
        1.0,
    )



@dataclass
class CoordSequence:
    origin: np.ndarray
    knots: np.ndarray
    fill: tuple[float, float, float, float] = (0,0,0,0)
    offset: tuple[float, float, float]  = (0, 0, 0)
    control_points_a: np.ndarray | None = None
    control_points_b: np.ndarray | None = None
    coord_type: Literal["LINEAR", "CURVE"] = field(init=False)
    npoints: int = field(init=False)

    def __post_init__(self):
        if self.control_points_a is None:
            self.coord_type = "LINEAR"
        else:
            self.coord_type = "CURVE"
        self.npoints = self.knots.shape[0]


Shape = list[CoordSequence]

@dataclass
class ParsedTrace:
    dimensions: tuple[int, int]
    shapes: list[Shape]


def _parse_path_line(line: str, fill: str, transform: str) -> list[CoordSequence]:
    """ parse a single line of the svg data """
    # NOTE: dirty fix
    # vtracer output is inconsistent
    # Case is not guaranteed so lower() is applied
    # lineto coords are formatted l0,0 1,2
    # curveto are formatted c0 0 1 2
    # curve coords are (control point, knot, control point)
    # transform only appears to has a single translation
    offset_x_str, offset_y_str = transform.split("(")[-1].rstrip(")").split(",")
    offset = (float(offset_x_str), -float(offset_y_str), 0.0)

    line = line.strip(" ").lower()
    is_linear = line[2] == ","
    if is_linear:
        line = line.replace(",", " ")

    lineto = "l"
    curveto = "c"
    close_path_char = "z"
    if is_linear:
        n_coord_points = 1
        split_char = lineto
    else:
        n_coord_points = 3
        split_char = curveto

    coord_sequences: list[CoordSequence] = []

    for path_section in line.split(close_path_char):
        path_section = path_section.strip().replace(split_char, "")
        if not path_section:
            continue

        # note: curved coordinates may be discarding origin point incorrectly
        arr = np.fromstring(path_section[1:], dtype=np.float64, sep=" ")
        if is_linear:
            origin = arr[:2]
            # arr = np.hstack(((0, 0), arr))
        else:
            origin, arr = np.split(arr, [2])
            arr = np.roll(arr, 4)

        n_values = arr.shape[0]
        arr = arr.reshape(n_values // 2, 2)
        arr *= np.array((1, -1)) # flip y
        arr += offset[:2]

        z = np.zeros((arr.shape[0], 1), dtype=np.float64)
        arr = np.hstack((arr, z))
        arr = np.reshape(arr, (arr.shape[0] // n_coord_points, n_coord_points, 3))
        if is_linear:
            coord_sequences.append(CoordSequence(
                origin,
                arr,
                fill=_hex_string_to_color(fill),  # remove hash prefix
                offset=offset,
            ))
        else:
            # note: may be edge cases where this is incorrect?
            coord_sequences.append(CoordSequence(
                origin,
                knots=arr[:,1,:],
                control_points_a=arr[:,0,:],
                control_points_b=arr[:,2,:],
                fill=_hex_string_to_color(fill),  # remove hash prefix
                offset=offset,
            ))

    return coord_sequences


def parse_trace(data: str):
    tree = ET.ElementTree(ET.fromstring(data))

    root = tree.getroot()
    dimensions = (
        int(root.get("width")),
        int(root.get("height"))
    )

    shapes: list[Shape] = []
    for child in root:
        path_string = child.attrib["d"]
        if not path_string:
            continue
        fill = child.attrib["fill"][1:]
        transform = child.attrib["transform"]
        parsed_line = _parse_path_line(path_string, fill, transform)
        shapes.append(parsed_line)

    return ParsedTrace(dimensions, shapes)


def create_bmesh(trace_result: ParsedTrace) -> bmesh.types.BMesh:
    """ Create mesh objects from the parsed svg data """
    curves = _create_curves(trace_result)

    bm = bmesh.new()
    offset = Vector((0.5, 0.5))
    for curve in curves:
        curve.data.offset = -0.00001  # Correct order confusion

        me = curve.to_mesh()
        bm.from_mesh(me)
        # me.polygons.foreach_get
        curve.to_mesh_clear()

        curve_data = curve.data
        bpy.data.objects.remove(curve)
        bpy.data.curves.remove(curve_data)

    # Correct degeneracy
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.001)

    # Calculate_uvs
    uv_layer = bm.loops.layers.uv.verify()
    scaler = Vector((
        1.0 / float(trace_result.dimensions[0]),
        1.0 / float(trace_result.dimensions[1])
    ))

    # NOTE: Ideally this would be done via the buffer protocol and values calculated through a single numpy call
    #       But it should be fast enough
    # Calculate/Generate UVs
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv = loop.vert.co.xy * scaler + offset

    return bm


def _create_curves(trace_result: ParsedTrace) -> list[bpy.types.Object]:
    """ Create curve objects from the parsed svg data """
    new_objects: list[bpy.types.Object] = []

    origin_offset = np.array((
        -trace_result.dimensions[0] / 2,
        trace_result.dimensions[1] / 2, 0
    ), dtype=np.float64)

    for i, shape in enumerate(trace_result.shapes):
        name = str(i).zfill(4)
        data = bpy.data.curves.new(name, "CURVE")
        data.dimensions = "2D"
        data.fill_mode = "FRONT"
        new_object = bpy.data.objects.new(name, object_data=data)
        new_objects.append(new_object)
        new_object.color = shape[0].fill

        for path in shape:
            is_linear = path.coord_type == "LINEAR"
            spline_type = "POLY" if is_linear else "BEZIER"
            spline = data.splines.new(spline_type)
            spline.use_cyclic_u = True
            n_points = path.npoints

            if is_linear:
                # Spline datablock initializes with a single point so - 1
                spline.points.add(n_points - 1)

                knots = path.knots + origin_offset

                # NOTE: Padded to a vec 4 necessary for polysplines
                padding = np.zeros((n_points, 1, 1), dtype=np.float64)
                padded = np.dstack((knots, padding))

                # Populate
                spline.points.foreach_set("co", padded.ravel())
            else:
                # Spline datablock initializes with a single point so - 1
                spline.bezier_points.add(n_points - 1)

                knots = path.knots + origin_offset
                control_points_a = path.control_points_a + origin_offset
                control_points_b = path.control_points_b + origin_offset

                # Populate
                spline.bezier_points.foreach_set("co", knots.ravel())
                spline.bezier_points.foreach_set("handle_left", control_points_a.ravel())
                spline.bezier_points.foreach_set("handle_right", control_points_b.ravel())

    return new_objects
