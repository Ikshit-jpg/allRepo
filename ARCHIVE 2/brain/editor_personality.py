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


# Probabilistic effect pool weights per personality.
# These are SAMPLING WEIGHTS for taste_system.TasteFilter — NOT global
# on/off flags. A weight of 2.0 means "twice as likely when eligible",
# never "always on". This is what kills the same-effect-everywhere look.
PERSONALITY_EFFECT_WEIGHTS: dict[str, dict[str, float]] = {
    "anime": {
        "impact_flash_combo": 1.6, "zoom_burst_shake": 1.4, "energy_aura": 1.5,
        "god_of_war": 1.2, "zoom_pulse": 1.3, "cinematic_grade": 1.0,
        "smear_cut": 1.4,
    },
    "cinematic": {
        "cinematic_grade": 2.0, "atmospheric_grade": 1.8, "ken_burns_slow": 1.6,
        "push_in_slow": 1.5, "subtle_drift": 1.4, "impact_flash_combo": 0.4,
        "zoom_burst_shake": 0.4,
    },
    "hyper_phonk": {
        "zoom_burst_shake": 2.0, "impact_flash_combo": 1.8, "camera_slam": 1.7,
        "handheld_shake": 1.6, "glitch_cut": 1.5, "subtle_drift": 0.5,
    },
    "sam_kolder": {
        "parallax_zoom": 1.8, "smear_cut": 1.7, "zoom_pulse": 1.5,
        "cinematic_grade": 1.4, "whip_pan_right": 1.6,
    },
    "tiktok_brainrot": {
        "zoom_pulse": 1.8, "impact_flash_combo": 1.6, "camera_slam": 1.5,
        "handheld_shake": 1.5, "energy_aura": 1.3,
    },
    "minimal": {
        "cinematic_grade": 2.0, "subtle_drift": 1.8, "ken_burns_slow": 1.5,
        "impact_flash_combo": 0.2, "zoom_burst_shake": 0.2, "camera_slam": 0.2,
    },
}


def get_personality_effect_weights(name: str) -> dict[str, float]:
    """Sampling weights for TasteFilter(personality_weights=...)."""
    return dict(PERSONALITY_EFFECT_WEIGHTS.get(name, PERSONALITY_EFFECT_WEIGHTS["anime"]))


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
    out["effect_weights"] = get_personality_effect_weights(personality.name)
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
