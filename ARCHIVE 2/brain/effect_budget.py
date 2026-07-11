from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


HEAVY_EFFECTS = {
    "impact_flash_combo",
    "zoom_burst_shake",
    "god_of_war",
    "camera_slam",
    "flash_cut",
    "glitch_cut",
}


@dataclass
class BudgetDecision:
    shot_index: int
    act_type: str
    budget: float
    spent_before: float
    spent_after: float
    limited: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EffectBudget:
    """
    Power allocation stage. It prevents the edit from spending drop-level energy
    before the moment deserves it.
    """

    ACT_BUDGETS = {
        "opening": 0.34,
        "build": 0.56,
        "climax": 1.0,
        "climax_resolution": 0.72,
        "empty": 0.0,
    }

    def __init__(self, global_ceiling: float = 1.0) -> None:
        self.global_ceiling = max(0.1, min(1.0, global_ceiling))
        self.last_decisions: List[BudgetDecision] = []

    def apply(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        decisions: List[BudgetDecision] = []

        for idx, shot in enumerate(timeline):
            shot = dict(shot)
            effects = [dict(effect) for effect in shot.get("effects", [])]
            act = ((shot.get("narrative") or {}).get("act") or {}).get("act_type", "build")
            strategy = shot.get("strategy") or {}
            budget = min(
                self.global_ceiling,
                float(strategy.get("intensity_ceiling") or self.ACT_BUDGETS.get(act, 0.55)),
            )
            if shot.get("is_drop") or shot.get("intent") == "impact":
                budget = max(budget, 0.78)
            if shot.get("has_speech"):
                budget = min(budget, 0.42)

            spent_before = self._spent(effects)
            reasons = ["Budget available" if spent_before <= budget else "Budget limited"]
            limited = False

            trimmed: List[Dict[str, Any]] = []
            for effect in effects:
                name = str(effect.get("name") or "")
                strength = float(effect.get("strength", 0.0) or 0.0)
                if name in HEAVY_EFFECTS and budget < 0.62:
                    limited = True
                    reasons.append(f"Removed heavy effect before payoff: {name}")
                    continue
                if strength > budget:
                    limited = True
                    strength = budget
                effect["strength"] = round(max(0.08, min(1.0, strength)), 3)
                trimmed.append(effect)

            if not trimmed and effects:
                fallback = dict(effects[0])
                fallback["name"] = "cinematic_grade"
                fallback["strength"] = round(min(0.32, budget), 3)
                trimmed.append(fallback)
                reasons.append("Fallback subtle grade kept the shot alive.")

            spent_after = self._spent(trimmed)
            decision = BudgetDecision(
                shot_index=idx,
                act_type=act,
                budget=round(budget, 3),
                spent_before=round(spent_before, 3),
                spent_after=round(spent_after, 3),
                limited=limited,
                reasons=list(dict.fromkeys(reasons)),
            )
            shot["effects"] = trimmed
            shot["effect_budget"] = decision.to_dict()
            self._append_stage_reason(shot, decision.reasons)
            decisions.append(decision)
            out.append(shot)

        self.last_decisions = decisions
        return out

    @staticmethod
    def _spent(effects: List[Dict[str, Any]]) -> float:
        if not effects:
            return 0.0
        power = sum(float(effect.get("strength", 0.0) or 0.0) for effect in effects)
        heavy = sum(0.18 for effect in effects if effect.get("name") in HEAVY_EFFECTS)
        return min(1.0, power / max(1, len(effects)) + heavy)

    @staticmethod
    def _append_stage_reason(shot: Dict[str, Any], reasons: List[str]) -> None:
        for log in shot.get("explainability_log", []):
            existing = list(log.get("reason") or [])
            for reason in reasons:
                if reason not in existing:
                    existing.append(reason)
            log["reason"] = existing
