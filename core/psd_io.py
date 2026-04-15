from __future__ import annotations

from pathlib import Path
from time import perf_counter

from ..utils import env, paths
from ..utils.logging import get_logger
from .models import LayerPart
from . import psd_layer_filters, seethrough_naming

logger = get_logger("psd_io")


def _coerce_bbox(raw_bbox) -> tuple[int, int, int, int]:
    if raw_bbox is None:
        return (0, 0, 0, 0)
    if hasattr(raw_bbox, "x1"):
        return (int(raw_bbox.x1), int(raw_bbox.y1), int(raw_bbox.x2), int(raw_bbox.y2))
    return tuple(int(value) for value in raw_bbox)


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
    return cleaned or "layer"


def _is_valid_bbox(bbox: tuple[int, int, int, int]) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def load_psd_layer_parts(
    filepath: str,
    *,
    ignore_hidden_layers: bool = True,
    ignore_empty_layers: bool = True,
    min_visible_pixels: int = 8,
    alpha_noise_floor: int = 64,
    visible_alpha_threshold: int = 32,
    auto_alpha_threshold_boost: bool = True,
    keep_tiny_named_parts: bool = True,
    configured_cache_dir: str = "",
) -> list[LayerPart]:
    env.ensure_psd_backend(configured_cache_dir)
    psd_tools = env.import_optional("psd_tools", configured_cache_dir)
    psd = psd_tools.PSDImage.open(filepath)

    source_path = str(Path(filepath).resolve())
    session_dir = paths.import_session_dir(Path(filepath).stem, configured_cache_dir)
    parts: list[LayerPart] = []
    draw_index = 0

    def walk(layer_container, parent_path: tuple[str, ...] = (), hidden_by_parent: bool = False) -> None:
        nonlocal draw_index
        for layer in layer_container:
            layer_name = getattr(layer, "name", "") or f"Layer {draw_index + 1}"
            path_parts = parent_path + (layer_name,)
            layer_path = "/".join(path_parts)
            layer_visible = bool(getattr(layer, "visible", True))
            layer_hidden = hidden_by_parent or (ignore_hidden_layers and not layer_visible)

            if hasattr(layer, "is_group") and layer.is_group():
                walk(layer, path_parts, layer_hidden)
                continue

            draw_index += 1
            bbox = _coerce_bbox(getattr(layer, "bbox", None))
            width = max(0, bbox[2] - bbox[0])
            height = max(0, bbox[3] - bbox[1])

            base_part = LayerPart(
                source_path=source_path,
                source_type="psd",
                document_path=source_path,
                layer_path=layer_path,
                layer_name=layer_name,
                image_size=(width, height),
                canvas_size=(int(psd.width), int(psd.height)),
                canvas_offset=(bbox[0], bbox[1]),
                draw_index=draw_index,
            )

            if layer_hidden:
                base_part.skipped = True
                base_part.skip_reason = "hidden layer"
                parts.append(base_part)
                continue

            if width <= 0 or height <= 0:
                base_part.skipped = True
                base_part.skip_reason = "zero-sized bounds"
                parts.append(base_part)
                continue

            raster_start = perf_counter()
            image = None
            raster_errors: list[str] = []
            try:
                # Prefer bbox-aware compositing so PSD layers behave like standalone images
                # before they enter the meshed-alpha import path.
                image = layer.composite(viewport=bbox)
            except Exception as exc:
                raster_errors.append(f"composite failed: {exc}")

            if image is None:
                try:
                    image = layer.topil()
                except Exception as exc:
                    raster_errors.append(f"topil failed: {exc}")

            if image is None:
                logger.warning(
                    "Failed to rasterize PSD layer %s: %s",
                    layer_path,
                    "; ".join(raster_errors) or "no raster data",
                )
                base_part.skipped = True
                base_part.skip_reason = "; ".join(raster_errors) or "rasterization failed"
                parts.append(base_part)
                continue

            if image is None:
                base_part.skipped = True
                base_part.skip_reason = "no raster data"
                parts.append(base_part)
                continue

            image = psd_layer_filters.ensure_rgba(image)
            stats = psd_layer_filters.visible_pixel_stats(
                image,
                threshold=visible_alpha_threshold,
                noise_floor=alpha_noise_floor,
                auto_boost_threshold=auto_alpha_threshold_boost,
            )
            visible_pixels = int(stats["visible_pixels"])
            local_bbox = tuple(int(value) for value in stats["local_bbox"])
            centroid = tuple(float(value) for value in stats["centroid"])
            alpha_max = int(stats.get("alpha_max", 0))
            alpha_threshold = int(stats.get("alpha_threshold", 0))

            if visible_pixels == 0 or not _is_valid_bbox(local_bbox):
                base_part.skipped = True
                base_part.skip_reason = f"fully transparent or faint alpha noise (max alpha {alpha_max})"
                parts.append(base_part)
                continue

            if visible_pixels < min_visible_pixels and ignore_empty_layers:
                if not (keep_tiny_named_parts and seethrough_naming.is_tiny_named_exception(layer_name, layer_path)):
                    base_part.skipped = True
                    base_part.skip_reason = f"below visible pixel threshold ({visible_pixels})"
                    parts.append(base_part)
                    continue

            expected_layer_size = (width, height)
            is_canvas_sized = image.size == (int(psd.width), int(psd.height))
            cropped = image.crop(local_bbox)
            temp_path = session_dir / f"{draw_index:04d}_{_safe_filename(layer_name)}.png"
            cropped.save(temp_path)

            if image.size == expected_layer_size:
                global_bbox = (
                    bbox[0] + local_bbox[0],
                    bbox[1] + local_bbox[1],
                    bbox[0] + local_bbox[2],
                    bbox[1] + local_bbox[3],
                )
                global_centroid = (bbox[0] + centroid[0], bbox[1] + centroid[1])
            elif is_canvas_sized:
                global_bbox = local_bbox
                global_centroid = centroid
            else:
                global_bbox = (
                    bbox[0] + local_bbox[0],
                    bbox[1] + local_bbox[1],
                    bbox[0] + local_bbox[2],
                    bbox[1] + local_bbox[3],
                )
                global_centroid = (bbox[0] + centroid[0], bbox[1] + centroid[1])

            cropped_bbox = (0, 0, cropped.size[0], cropped.size[1])
            raster_seconds = perf_counter() - raster_start

            part = LayerPart(
                source_path=source_path,
                source_type="psd",
                document_path=source_path,
                layer_path=layer_path,
                layer_name=layer_name,
                temp_image_path=str(temp_path),
                image_size=cropped.size,
                canvas_size=(int(psd.width), int(psd.height)),
                canvas_offset=(global_bbox[0], global_bbox[1]),
                alpha_bbox=global_bbox,
                local_alpha_bbox=cropped_bbox,
                centroid=global_centroid,
                area=visible_pixels,
                perimeter=float(stats["perimeter"]),
                draw_index=draw_index,
            )
            parts.append(part)
            logger.info(
                "Rasterized %s -> crop %sx%s from canvas %sx%s in %.3fs (alpha max %s, bbox threshold %s)",
                layer_path,
                cropped.size[0],
                cropped.size[1],
                psd.width,
                psd.height,
                raster_seconds,
                alpha_max,
                alpha_threshold,
            )

    walk(psd)
    logger.info("Parsed PSD %s into %s layer parts", filepath, len(parts))
    return parts
