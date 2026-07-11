from __future__ import annotations

from typing import Any, Dict, List


EFFECT_GRAMMAR: Dict[str, List[Dict[str, Any]]] = {
    "cinematic_grade": [{"op": "color_grade"}, {"op": "soft_sharpen"}, {"op": "grain"}],
    "atmospheric_grade": [{"op": "color_grade"}, {"op": "vignette"}, {"op": "grain"}],
    "portrait_enhance": [{"op": "color_grade"}, {"op": "glow", "target": "subject"}],
    "subtle_drift": [{"op": "transform", "motion": "drift"}],
    "ken_burns_slow": [{"op": "transform", "motion": "slow_push"}],
    "parallax_zoom": [{"op": "transform", "motion": "parallax"}],
    "push_in_slow": [{"op": "transform", "motion": "slow_push"}],
    "handheld_shake": [{"op": "shake", "mode": "handheld"}],
    "camera_slam": [{"op": "shake", "mode": "slam"}, {"op": "drop_punch"}],
    "energy_aura": [{"op": "glow"}, {"op": "saturation_pulse"}, {"op": "overlay", "asset": "aura"}],
    "zoom_pulse": [{"op": "transform", "motion": "pulse"}],
    "orbit_right": [{"op": "transform", "motion": "orbit_right"}],
    "impact_flash_combo": [{"op": "flash"}, {"op": "contrast_punch"}, {"op": "rgb_split"}],
    "zoom_burst_shake": [{"op": "transform", "motion": "snap_zoom"}, {"op": "shake", "mode": "burst"}],
    "god_of_war": [{"op": "contrast_punch"}, {"op": "vignette"}, {"op": "grain"}],
    "smear_cut": [{"op": "transition", "kind": "smear_cut"}],
    "whip_pan_right": [{"op": "transition", "kind": "whip_pan_right"}],
}


class EffectGrammar:
    """Maps editorial effect names into renderer-operation intent."""

    def apply(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for shot in timeline:
            shot = dict(shot)
            compiled = []
            for effect in shot.get("effects", []):
                name = str(effect.get("name") or "")
                ops = [dict(op) for op in EFFECT_GRAMMAR.get(name, [{"op": "color_grade"}])]
                for op in ops:
                    op["strength"] = effect.get("strength", 0.0)
                    op["source_effect"] = name
                compiled.extend(ops)
            shot["effect_grammar"] = compiled
            out.append(shot)
        return out
