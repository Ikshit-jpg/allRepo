from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EditorPersonality:
    name: str
    cut_density: float
    transition_density: float
    motion_intensity: float
    hold_bias: float
    grade_bias: str


PERSONALITIES: dict[str, EditorPersonality] = {
    "anime": EditorPersonality("anime", 1.08, 1.08, 1.18, 0.86, "anime_bloom"),
    "cinematic": EditorPersonality("cinematic", 0.78, 0.7, 0.82, 1.25, "film"),
    "hyper_phonk": EditorPersonality("hyper_phonk", 1.35, 1.22, 1.35, 0.72, "hard_contrast"),
    "sam_kolder": EditorPersonality("sam_kolder", 0.94, 1.28, 0.98, 1.0, "teal_orange"),
    "tiktok_brainrot": EditorPersonality("tiktok_brainrot", 1.45, 1.18, 1.28, 0.66, "punchy"),
    "minimal": EditorPersonality("minimal", 0.58, 0.42, 0.55, 1.55, "clean"),
}


def infer_editor_personality(prompt: str, mood: str) -> EditorPersonality:
    text = str(prompt or "").lower()
    if any(k in text for k in ("brainrot", "tiktok", "shorts", "reels")):
        return PERSONALITIES["tiktok_brainrot"]
    if any(k in text for k in ("hyper phonk", "phonk", "aggressive", "hype")) or mood == "hype":
        return PERSONALITIES["hyper_phonk"]
    if any(k in text for k in ("sam kolder", "travel edit", "smooth transition")):
        return PERSONALITIES["sam_kolder"]
    if any(k in text for k in ("minimal", "clean", "simple", "subtle")):
        return PERSONALITIES["minimal"]
    if any(k in text for k in ("cinematic", "film", "movie", "trailer")):
        return PERSONALITIES["cinematic"]
    return PERSONALITIES["anime"]


def apply_personality_to_style(style: dict[str, Any], personality: EditorPersonality) -> dict[str, Any]:
    out = dict(style)
    out["editor_style"] = personality.name
    out["cut_density"] = personality.cut_density
    out["transition_density"] = personality.transition_density
    out["motion_intensity"] = personality.motion_intensity
    out["hold_bias"] = personality.hold_bias
    out["grade_bias"] = personality.grade_bias
    if personality.name == "minimal":
        out["intensity"] = min(int(out.get("intensity", 1)), 1)
        out["simple_mode"] = True
    elif personality.name in {"hyper_phonk", "tiktok_brainrot"}:
        out["shake"] = True
        out["zoom"] = True
        out["flash"] = True
        out["auto_timefx"] = True
    elif personality.name == "cinematic":
        out["color_grade"] = True
        out["ken_burns"] = True
        out["overlap_transitions"] = True
    return out
