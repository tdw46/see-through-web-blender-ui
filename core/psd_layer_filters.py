from __future__ import annotations

from typing import Any


def ensure_rgba(image: Any):
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def visible_pixel_stats(image, threshold: int = 1) -> dict[str, object]:
    rgba = ensure_rgba(image)
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return {
            "visible_pixels": 0,
            "local_bbox": (0, 0, 0, 0),
            "centroid": (0.0, 0.0),
            "perimeter": 0.0,
        }

    width, _height = rgba.size
    visible_pixels = 0
    sum_x = 0.0
    sum_y = 0.0
    for index, value in enumerate(alpha.getdata()):
        if value >= threshold:
            x = index % width
            y = index // width
            visible_pixels += 1
            sum_x += x + 0.5
            sum_y += y + 0.5

    centroid = (sum_x / visible_pixels, sum_y / visible_pixels) if visible_pixels else (0.0, 0.0)
    perimeter = float((bbox[2] - bbox[0]) * 2 + (bbox[3] - bbox[1]) * 2)
    return {
        "visible_pixels": visible_pixels,
        "local_bbox": bbox,
        "centroid": centroid,
        "perimeter": perimeter,
    }


def layer_bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])

