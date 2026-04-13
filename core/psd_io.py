from __future__ import annotations

from pathlib import Path

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


def load_psd_layer_parts(
    filepath: str,
    *,
    ignore_hidden_layers: bool = True,
    ignore_empty_layers: bool = True,
    min_visible_pixels: int = 8,
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

            try:
                image = layer.topil()
            except Exception as exc:
                logger.warning("Failed to rasterize PSD layer %s: %s", layer_path, exc)
                base_part.skipped = True
                base_part.skip_reason = f"rasterization failed: {exc}"
                parts.append(base_part)
                continue

            if image is None:
                base_part.skipped = True
                base_part.skip_reason = "no raster data"
                parts.append(base_part)
                continue

            image = psd_layer_filters.ensure_rgba(image)
            stats = psd_layer_filters.visible_pixel_stats(image)
            visible_pixels = int(stats["visible_pixels"])
            local_bbox = tuple(int(value) for value in stats["local_bbox"])
            centroid = tuple(float(value) for value in stats["centroid"])

            if visible_pixels == 0 and ignore_empty_layers:
                base_part.skipped = True
                base_part.skip_reason = "fully transparent"
                parts.append(base_part)
                continue

            if visible_pixels < min_visible_pixels and ignore_empty_layers:
                if not (keep_tiny_named_parts and seethrough_naming.is_tiny_named_exception(layer_name, layer_path)):
                    base_part.skipped = True
                    base_part.skip_reason = f"below visible pixel threshold ({visible_pixels})"
                    parts.append(base_part)
                    continue

            temp_path = session_dir / f"{draw_index:04d}_{_safe_filename(layer_name)}.png"
            image.save(temp_path)

            global_bbox = (
                bbox[0] + local_bbox[0],
                bbox[1] + local_bbox[1],
                bbox[0] + local_bbox[2],
                bbox[1] + local_bbox[3],
            )
            global_centroid = (bbox[0] + centroid[0], bbox[1] + centroid[1])

            part = LayerPart(
                source_path=source_path,
                source_type="psd",
                document_path=source_path,
                layer_path=layer_path,
                layer_name=layer_name,
                temp_image_path=str(temp_path),
                image_size=image.size,
                canvas_size=(int(psd.width), int(psd.height)),
                canvas_offset=(bbox[0], bbox[1]),
                alpha_bbox=global_bbox,
                local_alpha_bbox=local_bbox,
                centroid=global_centroid,
                area=visible_pixels,
                perimeter=float(stats["perimeter"]),
                draw_index=draw_index,
            )
            parts.append(part)

    walk(psd)
    logger.info("Parsed PSD %s into %s layer parts", filepath, len(parts))
    return parts
