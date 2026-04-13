from __future__ import annotations

from .models import LayerPart
from . import seethrough_naming


def _bbox_width(part: LayerPart) -> float:
    return float(part.alpha_bbox[2] - part.alpha_bbox[0])


def _bbox_height(part: LayerPart) -> float:
    return float(part.alpha_bbox[3] - part.alpha_bbox[1])


def _centerline_x(parts: list[LayerPart]) -> float:
    torso_like = []
    for part in parts:
        if part.skipped:
            continue
        token, _, _ = seethrough_naming.classify_name(part.layer_name, part.layer_path)
        if token in {"topwear", "bottomwear", "face"}:
            torso_like.append(part)
    if torso_like:
        return sum(part.centroid[0] for part in torso_like) / len(torso_like)
    if parts:
        return parts[0].canvas_size[0] * 0.5
    return 0.0


def _infer_side(part: LayerPart, centerline_x: float) -> str:
    if part.centroid[0] < centerline_x:
        return "L"
    if part.centroid[0] > centerline_x:
        return "R"
    return "UNKNOWN"


def _geometry_fallback_label(part: LayerPart, centerline_x: float) -> tuple[str, float]:
    canvas_h = max(1.0, float(part.canvas_size[1]))
    rel_y = part.centroid[1] / canvas_h
    width = max(1.0, _bbox_width(part))
    height = max(1.0, _bbox_height(part))
    aspect = width / height
    side = _infer_side(part, centerline_x)

    if rel_y < 0.28:
        return "head", 0.45
    if 0.20 <= rel_y <= 0.65 and abs(part.centroid[0] - centerline_x) > part.canvas_size[0] * 0.12 and aspect >= 0.65:
        return f"arm_{side.lower()}" if side in {"L", "R"} else "arm", 0.4
    if rel_y > 0.55 and height >= width * 0.8:
        return f"leg_{side.lower()}" if side in {"L", "R"} else "leg", 0.4
    if 0.28 <= rel_y <= 0.70:
        return "torso", 0.35
    return "accessory", 0.2


def classify_parts(parts: list[LayerPart]) -> list[LayerPart]:
    centerline_x = _centerline_x(parts)

    for part in parts:
        if part.skipped:
            continue

        token, side, confidence = seethrough_naming.classify_name(part.layer_name, part.layer_path)
        part.normalized_token = token or seethrough_naming.normalize_name(part.layer_name)
        inferred_side = side
        if inferred_side == "UNKNOWN" and token in seethrough_naming.SIDE_SENSITIVE_TOKENS:
            inferred_side = _infer_side(part, centerline_x)
        part.side_guess = inferred_side

        label = seethrough_naming.map_token_to_label(token, inferred_side)
        if label == "unclassified":
            label, confidence = _geometry_fallback_label(part, centerline_x)
            if label.startswith("arm_") or label.startswith("leg_") or label.startswith("foot_"):
                part.side_guess = label.rsplit("_", 1)[-1].upper()

        part.semantic_label = label
        part.confidence = confidence

    return parts
