from __future__ import annotations

from typing import Any
import numpy as np


def ensure_rgba(image: Any):
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def visible_pixel_stats(
    image,
    *,
    threshold: int = 32,
    noise_floor: int = 64,
    auto_boost_threshold: bool = True,
) -> dict[str, object]:
    rgba = ensure_rgba(image)
    alpha = rgba.getchannel("A")
    alpha_np = np.asarray(alpha, dtype=np.uint8)
    alpha_max = int(alpha_np.max()) if alpha_np.size else 0

    # Ignore extremely faint PSD noise that otherwise expands the bbox to the full canvas.
    if alpha_max < max(0, int(noise_floor)):
        return {
            "visible_pixels": 0,
            "local_bbox": (0, 0, 0, 0),
            "centroid": (0.0, 0.0),
            "perimeter": 0.0,
            "alpha_max": alpha_max,
            "alpha_threshold": int(threshold),
        }

    base_threshold = max(0, int(threshold))
    if auto_boost_threshold:
        strong_threshold = max(base_threshold, min(max(0, int(noise_floor)), max(base_threshold, alpha_max // 4)))
    else:
        strong_threshold = base_threshold
    mask = alpha_np >= strong_threshold
    visible_pixels = int(mask.sum())
    if visible_pixels:
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        centroid = (float(xs.mean() + 0.5), float(ys.mean() + 0.5))
    else:
        bbox = (0, 0, 0, 0)
        centroid = (0.0, 0.0)

    perimeter = float((bbox[2] - bbox[0]) * 2 + (bbox[3] - bbox[1]) * 2) if visible_pixels else 0.0
    return {
        "visible_pixels": visible_pixels,
        "local_bbox": bbox,
        "centroid": centroid,
        "perimeter": perimeter,
        "alpha_max": alpha_max,
        "alpha_threshold": strong_threshold,
    }


def layer_bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])
