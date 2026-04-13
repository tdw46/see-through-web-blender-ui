from __future__ import annotations

from .models import BonePlan, LayerPart, RigPlan


def _visible_parts(parts: list[LayerPart]) -> list[LayerPart]:
    return [part for part in parts if not part.skipped]


def _union_bbox(parts: list[LayerPart], labels: set[str] | None = None) -> tuple[float, float, float, float] | None:
    selected = [part for part in parts if labels is None or part.semantic_label in labels]
    if not selected:
        return None
    return (
        min(part.alpha_bbox[0] for part in selected),
        min(part.alpha_bbox[1] for part in selected),
        max(part.alpha_bbox[2] for part in selected),
        max(part.alpha_bbox[3] for part in selected),
    )


def _pixel_to_plane(x: float, y: float, canvas_size: tuple[int, int]) -> tuple[float, float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    canvas_w, canvas_h = canvas_size
    return ((x - canvas_w * 0.5) * scale, 0.0, (canvas_h * 0.5 - y) * scale)


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _center_of_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _side_bbox(parts: list[LayerPart], semantic_prefix: str, side: str) -> tuple[float, float, float, float] | None:
    target = f"{semantic_prefix}_{side.lower()}"
    direct = _union_bbox(parts, {target})
    if direct:
        return direct
    fallback = [part for part in parts if part.semantic_label == semantic_prefix and part.side_guess == side]
    if not fallback:
        return None
    return (
        min(part.alpha_bbox[0] for part in fallback),
        min(part.alpha_bbox[1] for part in fallback),
        max(part.alpha_bbox[2] for part in fallback),
        max(part.alpha_bbox[3] for part in fallback),
    )


def estimate_rig(parts: list[LayerPart]) -> RigPlan:
    visible = _visible_parts(parts)
    if not visible:
        return RigPlan()

    canvas_size = visible[0].canvas_size
    overall_bbox = _union_bbox(visible)
    torso_bbox = _union_bbox(visible, {"torso", "pelvis"}) or overall_bbox
    head_bbox = _union_bbox(visible, {"head"}) or (
        torso_bbox[0],
        max(0.0, torso_bbox[1] - (torso_bbox[3] - torso_bbox[1]) * 0.9),
        torso_bbox[2],
        torso_bbox[1],
    )
    neck_bbox = _union_bbox(visible, {"neck"})
    centerline_x = _center_of_bbox(torso_bbox)[0]

    torso_w = max(1.0, torso_bbox[2] - torso_bbox[0])
    torso_h = max(1.0, torso_bbox[3] - torso_bbox[1])
    head_h = max(1.0, head_bbox[3] - head_bbox[1])

    pelvis = (centerline_x, torso_bbox[1] + torso_h * 0.72)
    chest = (centerline_x, torso_bbox[1] + torso_h * 0.30)
    root = (centerline_x, torso_bbox[3] + torso_h * 0.08)
    neck = (
        centerline_x,
        neck_bbox[1] + (neck_bbox[3] - neck_bbox[1]) * 0.55 if neck_bbox else (head_bbox[3] + torso_bbox[1]) * 0.5,
    )
    head_base = (centerline_x, head_bbox[3] - head_h * 0.28)
    head_tip = (centerline_x, max(0.0, head_bbox[1] - head_h * 0.08))

    bones: dict[str, BonePlan] = {}

    def add_bone(name: str, head_xy: tuple[float, float], tail_xy: tuple[float, float], parent: str | None) -> None:
        bones[name] = BonePlan(
            name=name,
            head=_pixel_to_plane(head_xy[0], head_xy[1], canvas_size),
            tail=_pixel_to_plane(tail_xy[0], tail_xy[1], canvas_size),
            parent=parent,
        )

    add_bone("root", root, pelvis, None)
    add_bone("torso", pelvis, chest, "root")
    add_bone("chest", chest, neck, "torso")
    add_bone("neck", neck, head_base, "chest")
    add_bone("head", head_base, head_tip, "neck")

    def build_arm(side: str) -> None:
        arm_bbox = _side_bbox(visible, "arm", side)
        hand_bbox = _side_bbox(visible, "hand", side)
        direction = -1.0 if side == "L" else 1.0
        shoulder = (torso_bbox[0] + torso_w * 0.10, chest[1]) if side == "L" else (torso_bbox[2] - torso_w * 0.10, chest[1])
        if arm_bbox:
            wrist = ((arm_bbox[0] if side == "L" else arm_bbox[2]), (arm_bbox[1] + arm_bbox[3]) * 0.5)
        else:
            wrist = (shoulder[0] + torso_w * 0.55 * direction, shoulder[1] + torso_h * 0.10)
        if hand_bbox:
            wrist = ((hand_bbox[2] if side == "L" else hand_bbox[0]), (hand_bbox[1] + hand_bbox[3]) * 0.5)
            hand_tip = ((hand_bbox[0] if side == "L" else hand_bbox[2]), (hand_bbox[1] + hand_bbox[3]) * 0.5)
        else:
            hand_tip = (wrist[0] + torso_w * 0.12 * direction, wrist[1])
        elbow = _lerp(shoulder, wrist, 0.55)
        add_bone(f"upper_arm.{side}", shoulder, elbow, "chest")
        add_bone(f"lower_arm.{side}", elbow, wrist, f"upper_arm.{side}")
        add_bone(f"hand.{side}", wrist, hand_tip, f"lower_arm.{side}")

    def build_leg(side: str) -> None:
        leg_bbox = _side_bbox(visible, "leg", side)
        foot_bbox = _side_bbox(visible, "foot", side)
        direction = -1.0 if side == "L" else 1.0
        hip = (centerline_x + torso_w * 0.18 * direction, pelvis[1])
        if leg_bbox:
            ankle = ((leg_bbox[0] + leg_bbox[2]) * 0.5, leg_bbox[3])
        else:
            ankle = (hip[0], overall_bbox[3])
        knee = _lerp(hip, ankle, 0.55)
        if foot_bbox:
            ankle = ((foot_bbox[0] + foot_bbox[2]) * 0.5, foot_bbox[1])
            foot_tip = ((foot_bbox[0] if side == "L" else foot_bbox[2]), (foot_bbox[1] + foot_bbox[3]) * 0.5)
        else:
            foot_tip = (ankle[0] + torso_w * 0.14 * direction, ankle[1] + torso_h * 0.04)
        add_bone(f"upper_leg.{side}", hip, knee, "root")
        add_bone(f"lower_leg.{side}", knee, ankle, f"upper_leg.{side}")
        add_bone(f"foot.{side}", ankle, foot_tip, f"lower_leg.{side}")

    build_arm("L")
    build_arm("R")
    build_leg("L")
    build_leg("R")

    confidence_values = [part.confidence for part in visible if part.confidence > 0.0]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.25
    return RigPlan(bones=bones, confidence=confidence, centerline_x=centerline_x)
