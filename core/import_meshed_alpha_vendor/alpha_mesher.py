from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from itertools import chain
from typing import Literal

from bpy.types import Object, Image, Context
from mathutils import Vector
import bmesh
import numpy as np
import vtracer

from . import trace_parser
from .bm_help import get_faces_from_verts, calc_total_face_area, get_vert_islands, calc_bbox

from ...utils.logging import get_logger
from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec, ROI

logger = get_logger("import_meshed_alpha_vendor")

@lru_cache(maxsize=2)
def preprocess_image(
    im_path: Path,
    use_luma_as_alpha: bool = False,
    rgb_to_alpha_method: Literal["LUMA", "MAX", "MIN"] | None = None,
    invert_alpha: bool = False,
    dilate: int = 0,
    predivide_alpha: bool = True,
    contrast_remap: tuple[float, float] = (0.1, 0.9),
    resample: float = 1.0,
    output_path: Path | None = None
) -> np.ndarray:
    """ Preprocess image, return as an ndarray of [height, width, channels] """
    preprocessed = _ImagePreprocessor.execute(
        im_path, use_luma_as_alpha, rgb_to_alpha_method,
        invert_alpha, dilate, predivide_alpha, contrast_remap,
        resample, output_path
    )
    return preprocessed


def trace_image(
    pixels: np.ndarray,
    mode:Literal["spline", "polygon", "none"] = "spline",
) -> str:
    """ Trace image with vtracer and produce an svg string """
    # TODO: This needs a disk caching mechanism
    height, width = pixels.shape[:2]
    size = (width, height)
    new_shape = (width * height, 4)
    pixels = pixels.reshape(new_shape).astype(np.int64)
    pixels = tuple(map(tuple, pixels.tolist()))  # pyright: ignore

    try:
        trace_result = vtracer.convert_pixels_to_svg(
            pixels, # pyright: ignore
            size, "color", "stacked",
            # size, "bw", "stacked",
            mode=mode,             # 
            filter_speckle=0,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=8
        )
    except Exception as e:
        logger.error(f"Error tracing: {type(e)}{e}")
        trace_result = ""

    return trace_result


@lru_cache(maxsize=3)
def parse_trace(data: str) -> trace_parser.ParsedTrace:
    """ Parse vtracer svg before meshing """
    return trace_parser.parse_trace(data)


def parsed_to_bmesh(parsed_svg: trace_parser.ParsedTrace, context: Context) -> bmesh.types.BMesh:
    """ Produce bmesh object from parsed trace data """
    return trace_parser.create_bmesh(parsed_svg)


def post_process_mesh(
        bm: bmesh.types.BMesh,
        x_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        y_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        triangulate: bool = True,
        xy_divisions: tuple[int, int] = (5, 5),
        divide_ngons: bool = False,
        remove_small_islands: int = 0
):
    """ Cleanup and filtering of meshed trace data """
    _MeshPostProcessor.execute(
        bm, x_align, y_align, triangulate,
        xy_divisions, divide_ngons, remove_small_islands
    )


class _ImagePreprocessor:
    @classmethod
    def execute(cls,
        image: Path,
        use_luma_as_alpha: bool = False,
        rgb_to_alpha_method: Literal["LUMA", "MAX", "MIN"] | None = None,
        invert_alpha: bool = False, dilate: int = 0,
        predivide_alpha: bool = True,
        contrast_remap: tuple[float, float] = (0.1, 0.9),
        resample: float = 1.0,
        debug_path: Path | None = None
    ) -> np.ndarray:
        """ Apply image preprocessing """
        # Get image and extract its alpha
        buf = ImageBuf(str(image))
        spec: ImageSpec = buf.spec()

        if np.abs(resample-1) > 0.001:
            new_size = ROI(
                0, max(1, int(spec.width * resample)),
                0, max(1, int(spec.height * resample))
            )
            buf = ImageBufAlgo.resample(buf, False, new_size)

        buf = cls._get_alpha(buf, use_luma_as_alpha, rgb_to_alpha_method)

        if contrast_remap[0] > 0.0001 or contrast_remap[1] < 0.9999:
            logger.debug(f"Applying contrast remap {contrast_remap}")
            buf = ImageBufAlgo.contrast_remap(buf, *contrast_remap)
            buf = ImageBufAlgo.clamp(buf, 0, 1)

        # buf = ImageBufAlgo.fillholes_pushpull(buf)

        if invert_alpha:
            logger.debug(f"Inverting alpha")
            buf = ImageBufAlgo.invert(buf)

        if predivide_alpha:
            logger.debug(f"Dividing Alpha")
            buf = ImageBufAlgo.div(buf, buf)

        if dilate != 0:
            logger.debug(f"Dilate/Erode {dilate}")
            buf = cls._morphologic_adjust(buf, dilate)

        # if debug_path is not None:
        #     logger.debug(f"Saving preprocess image as: {debug_path}")
        #     buf.write(str(debug_path))

        # Vtracer expects 4 channels
        logger.debug("Padding channels")
        buf = cls._channel_pad_1_to_4(buf)

        pixels: np.ndarray = buf.get_pixels()

        logger.debug("Image Preprocess Complete")
        return pixels

    @classmethod
    def _get_alpha(
        cls, buf: ImageBuf,
        use_rgb_as_alpha,
        rgb_to_alpha_method: Literal["LUMA", "MAX", "MIN"] | None = "MAX",
    ) -> ImageBuf:
        spec = buf.spec()
        if spec.alpha_channel == -1 or use_rgb_as_alpha:
            converter = {
                "LUMA": cls._rgb_to_luma,
                "MAX": ImageBufAlgo.maxchan,
                "MIN": cls._rgb_min_to_alpha
            }
            return converter[rgb_to_alpha_method](buf)
        else:
            a = spec.alpha_channel
            return ImageBufAlgo.channels(buf, (a,), (a,))

    @staticmethod
    def _rgb_min_to_alpha(buf: ImageBuf) -> ImageBuf:
        return ImageBufAlgo.invert(ImageBufAlgo.minchan(buf))

    @staticmethod
    def _rgb_to_luma(buf: ImageBuf) -> ImageBuf:
        rgb = ImageBufAlgo.channels(buf, (0, 1, 2), (0, 1, 2))
        weights = (.2126, .7152, .0722)
        return ImageBufAlgo.channel_sum(rgb, weights)

    @classmethod
    def _channel_pad_1_to_4(cls, buf: ImageBuf) -> ImageBuf:
        """ Repeat first channel 4 times """
        return ImageBufAlgo.channels(buf, (0, 0, 0, 0), ("R", "G", "B", "A")) 

    @classmethod
    def _morphologic_adjust(cls, buf: ImageBuf, amount: int) -> ImageBuf:
        k_size = int(abs(amount) * 2 + 1)
        if amount > 0:
            return ImageBufAlgo.dilate(buf, k_size, k_size)
        elif amount < 0:
            return ImageBufAlgo.erode(buf, k_size, k_size)
        else:
            return buf


class _MeshPostProcessor:
    @classmethod
    def execute(cls,
        bm: bmesh.types.BMesh,
        x_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        y_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        triangulate: bool = True,
        xy_divisions: tuple[int, int] = (5, 5),
        divide_ngons: bool = False,
        remove_small_islands: int = 0
    ):
        bmesh.ops.dissolve_edges(bm, edges=bm.edges)

        do_alignment = (x_align, y_align) != ("NONE", "NONE")
        do_xy_division = xy_divisions != (1, 1)
        bounds_mod_required = do_alignment or do_xy_division

        if remove_small_islands > 0:
            cls.remove_small_islands(bm, float(remove_small_islands))

        if bounds_mod_required:
            bounds = calc_bbox(bm)
            bounds = cls._axis_align_mesh(bm, x_align, y_align, bounds)
            cls._xy_grid_divide(bm, xy_divisions, bounds)

        if triangulate:
            bmesh.ops.triangulate(bm, faces=bm.faces)

        if divide_ngons:
            ngons = [f for f in bm.faces[:] if len(f.edges) > 4]
            bmesh.ops.triangulate(bm, faces=ngons)

    # @staticmethod
    # def _calc_bbox(bm: bmesh.types.BMesh) -> tuple[Vector, Vector]:
    #     # Calc bounding box
    #     bound_min = [float("inf")] * 3
    #     bound_max = [float("-inf")] * 3
    #
    #     for v in bm.verts:
    #         for i in range(3):
    #             bound_min[i] = min(bound_min[i], v.co[i])
    #             bound_max[i] = max(bound_max[i], v.co[i])
    #
    #     return Vector(bound_min), Vector(bound_max)

    @classmethod
    def _xy_grid_divide(
        cls,
        bm: bmesh.types.BMesh,
        xy_divisions: tuple[int, int] = (5, 5),
        bounds: tuple[Vector, Vector] | None = None,
    ):
        if bounds is None: bounds = calc_bbox(bm)

        cut_normals = Vector((1, 0, 0)), Vector((0, 1, 0))
        for axis, n_division, cut_normal in zip("xy", xy_divisions, cut_normals):
            start, end = getattr(bounds[0], axis), getattr(bounds[1], axis)
            axis_length = abs(start - end)
            step_size = axis_length / n_division
            cut_loc = Vector((0, 0, 0))

            for step in range(1, n_division):
                setattr(cut_loc, axis, step * step_size + start)
                geom = bm.edges[:] + bm.faces[:]
                bmesh.ops.bisect_plane(
                    bm, geom=geom,
                    plane_co=cut_loc,
                    plane_no=cut_normal,
                )

    @classmethod
    def _axis_align_mesh(
        cls,
        bm: bmesh.types.BMesh,
        x_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        y_align: Literal["NONE", "MIN", "CENTER", "MAX"] = "NONE",
        bounds: tuple[Vector, Vector] | None = None,
    ) -> tuple[Vector, Vector]:
        """ Axis align mesh in place and return updated bounding box """
        if bounds is None: bounds = calc_bbox(bm)
        bound_min, bound_max = bounds

        # Calculate offset
        offset = [0.0] * 3
        for i, align_arg in enumerate((x_align, y_align)):
            if align_arg == "NONE":
                continue

            if align_arg == "MIN":
                offset[i] = -bound_min[i]
            elif align_arg == "MAX":
                offset[i] = -bound_max[i]
            elif align_arg == "CENTER":
                offset[i] = -((bound_min[i] + bound_max[i]) * 0.5)
            else:
                logger.error(f"ERROR: Unrecognized mesh alignment arg {align_arg}")

        bmesh.ops.translate(bm, vec=offset, verts=bm.verts)

        # Return updated bounding box
        offset_v = Vector(offset)
        return bound_min + offset_v, bound_max + offset_v

    @classmethod
    def remove_small_islands(cls, bm: bmesh.types.BMesh, threshold: float):
        """ Remove side effects by operating on existing bmesh """
        def _islands_below_threshold(island):
            return calc_total_face_area(island) < threshold

        vert_islands = get_vert_islands(bm)
        face_islands = [get_faces_from_verts(island) for island in vert_islands]
        below_threshold = filter(_islands_below_threshold, face_islands)
        to_delete = list(chain.from_iterable(below_threshold))
        bmesh.ops.delete(bm, geom=to_delete, context="FACES")
        return bm


