from __future__ import annotations

import re

TOKEN_ALIASES = {
    "front hair": ("front hair", "front_hair", "hair front", "bangs", "fringe"),
    "back hair": ("back hair", "back_hair", "hair back"),
    "neck": ("neck",),
    "topwear": ("topwear", "top wear", "body", "torso", "shirt", "coat", "upper body"),
    "handwear": ("handwear", "hand wear", "arms", "arm", "hands", "hand", "sleeves", "sleeve"),
    "bottomwear": ("bottomwear", "bottom wear", "hips", "hip", "pelvis", "skirt", "pants", "waist"),
    "legwear": ("legwear", "leg wear", "legs", "leg", "thigh", "calf"),
    "footwear": ("footwear", "foot wear", "feet", "foot", "shoes", "shoe", "boots", "boot"),
    "tail": ("tail",),
    "wings": ("wings", "wing"),
    "objects": ("objects", "object", "prop", "props", "accessory", "accessories"),
    "headwear": ("headwear", "head wear", "hat", "hood", "helmet"),
    "face": ("face", "head", "skin"),
    "irides": ("irides", "iris", "pupil", "pupils"),
    "eyebrow": ("eyebrow", "eyebrows", "brow", "brows"),
    "eyewhite": ("eyewhite", "eye white", "sclera"),
    "eyelash": ("eyelash", "eyelashes", "lash", "lashes"),
    "eyewear": ("eyewear", "glasses", "goggles"),
    "ears": ("ears", "ear"),
    "earwear": ("earwear", "ear wear", "earring", "earrings"),
    "nose": ("nose",),
    "mouth": ("mouth", "lips", "teeth", "tongue"),
}

SIDE_SENSITIVE_TOKENS = {"handwear", "legwear", "footwear"}
TINY_EXCEPTIONS = {"irides", "eyebrow", "eyewhite", "eyelash", "nose", "mouth", "ears", "earwear"}


def normalize_name(name: str) -> str:
    lowered = (name or "").lower()
    lowered = re.sub(r"[_\-./]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def detect_side(name: str) -> str:
    normalized = f" {normalize_name(name)} "
    if any(token in normalized for token in (" left ", " l ", " _l ", ".l ", " lft ")):
        return "L"
    if any(token in normalized for token in (" right ", " r ", " _r ", ".r ", " rgt ")):
        return "R"
    return "UNKNOWN"


def classify_name(name: str, group_path: str = "") -> tuple[str, str, float]:
    merged = normalize_name(f"{group_path} {name}")
    side = detect_side(merged)
    for canonical, aliases in TOKEN_ALIASES.items():
        for alias in aliases:
            if alias in merged:
                confidence = 1.0 if canonical in merged else 0.88
                return canonical, side, confidence
    return "", side, 0.0


def is_tiny_named_exception(name: str, group_path: str = "") -> bool:
    token, _, _ = classify_name(name, group_path)
    return token in TINY_EXCEPTIONS


def map_token_to_label(token: str, side: str) -> str:
    if token == "front hair":
        return "hair_front"
    if token == "back hair":
        return "hair_back"
    if token == "neck":
        return "neck"
    if token == "topwear":
        return "torso"
    if token == "bottomwear":
        return "pelvis"
    if token == "handwear":
        return f"arm_{side.lower()}" if side in {"L", "R"} else "arm"
    if token == "legwear":
        return f"leg_{side.lower()}" if side in {"L", "R"} else "leg"
    if token == "footwear":
        return f"foot_{side.lower()}" if side in {"L", "R"} else "foot"
    if token in {"face", "headwear", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear", "ears", "earwear", "nose", "mouth"}:
        return "head"
    if token == "tail":
        return "tail"
    if token == "wings":
        return "wings"
    if token == "objects":
        return "accessory"
    return "unclassified"

