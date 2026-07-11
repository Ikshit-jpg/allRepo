from __future__ import annotations

from typing import Any, Dict, List

try:
    from .effect_budget import EffectBudget, HEAVY_EFFECTS
    from .quality_judge import FLASH_EFFECTS
    from .quality_judge import QualityJudge, QualityReport
except ImportError:  # script-style execution from brain/
    from effect_budget import EffectBudget, HEAVY_EFFECTS  # type: ignore
    from quality_judge import FLASH_EFFECTS  # type: ignore
    from quality_judge import QualityJudge, QualityReport  # type: ignore


class RevisionLoop:
    """Repairs timeline problems before expensive preview/final rendering."""

    def __init__(self, passes: int = 3) -> None:
        self.passes = max(1, min(3, int(passes)))
        self.judge = QualityJudge()
        self.budget = EffectBudget()
        self.last_reports: List[QualityReport] = []

    def run(
        self,
        timeline: List[Dict[str, Any]],
        initial_report: QualityReport | None = None,
    ) -> List[Dict[str, Any]]:
        current = [dict(shot) for shot in timeline]
        self.last_reports = []

        for pass_idx in range(self.passes):
            report = initial_report if pass_idx == 0 and initial_report is not None else self.judge.evaluate(current)
            self.last_reports.append(report)
            if report.passed:
                break
            current = self._revise_once(current, report, pass_idx)
            current = self.budget.apply(current)

        final_report = self.judge.evaluate(current)
        self.last_reports.append(final_report)
        for shot in current:
            shot["quality_report"] = final_report.to_dict()
        return current

    def _revise_once(
        self,
        timeline: List[Dict[str, Any]],
        report: QualityReport,
        pass_idx: int,
    ) -> List[Dict[str, Any]]:
        out = [self._deepish_copy(shot) for shot in timeline]
        for issue in report.issues:
            if issue.shot_index < 0 or issue.shot_index >= len(out):
                continue
            shot = out[issue.shot_index]
            if issue.code in {"speech_heavy_effect", "speech_too_loud"}:
                self._make_subtle(shot, "Dialogue safety revision")
            elif issue.code in {"heavy_too_close", "effect_density_high"}:
                self._reduce_power(shot, 0.72, "Reduced density during revision", issue.code)
            elif issue.code == "too_many_flashes":
                self._reduce_flashes(shot, issue.code)
            elif issue.code == "repeated_effect_combo":
                self._swap_last_effect(shot, pass_idx, issue.code)
            elif issue.code == "repeated_transition":
                shot["transition"] = "cut" if shot.get("transition") != "cut" else "dissolve"
                self._record_action(shot, issue.code, "Changed only the repeated transition.")
            elif issue.code == "climax_not_strongest":
                self._boost_climax(shot, issue.code)
        return out

    @staticmethod
    def _deepish_copy(shot: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(shot)
        copied["effects"] = [dict(effect) for effect in shot.get("effects", [])]
        copied["explainability_log"] = [dict(log) for log in shot.get("explainability_log", [])]
        return copied

    def _make_subtle(self, shot: Dict[str, Any], reason: str) -> None:
        shot["effects"] = [
            {"name": "portrait_enhance" if shot.get("has_speech") else "cinematic_grade", "strength": 0.28}
        ]
        shot["transition"] = "cut"
        self._add_reason(shot, reason)
        self._record_action(shot, "speech_safety", "Replaced only the unsafe dialogue treatment.")

    def _reduce_power(self, shot: Dict[str, Any], scale: float, reason: str, issue_code: str) -> None:
        reduced = []
        for effect in shot.get("effects", []):
            if effect.get("name") in HEAVY_EFFECTS and not (shot.get("is_drop") or shot.get("intent") == "impact"):
                continue
            effect = dict(effect)
            strength = self._float_value(effect.get("strength"), 0.0)
            effect["strength"] = 0.0 if strength <= 0.0 else round(max(0.1, strength * scale), 3)
            reduced.append(effect)
        shot["effects"] = reduced or [{"name": "cinematic_grade", "strength": 0.24}]
        self._add_reason(shot, reason)
        self._record_action(shot, issue_code, "Reduced only effect power/density.")

    def _reduce_flashes(self, shot: Dict[str, Any], issue_code: str) -> None:
        kept = []
        removed = False
        for effect in shot.get("effects", []):
            if effect.get("name") in FLASH_EFFECTS:
                removed = True
                continue
            kept.append(effect)
        shot["effects"] = kept or [{"name": "cinematic_grade", "strength": 0.26}]
        if shot.get("transition") in FLASH_EFFECTS:
            shot["transition"] = "cut"
            removed = True
        if removed:
            self._add_reason(shot, "Reduced flashes only")
            action = "Removed flash effects/transitions only."
        else:
            action = "Flash issue already resolved by an earlier targeted fix."
        self._record_action(shot, issue_code, action)

    def _swap_last_effect(self, shot: Dict[str, Any], pass_idx: int, issue_code: str) -> None:
        alternatives = ["atmospheric_grade", "subtle_drift", "parallax_zoom"]
        effects = shot.get("effects", [])
        if effects:
            effects[-1] = {"name": alternatives[pass_idx % len(alternatives)], "strength": 0.30}
        else:
            effects.append({"name": alternatives[pass_idx % len(alternatives)], "strength": 0.30})
        shot["effects"] = effects
        self._add_reason(shot, "Forced variation during revision")
        self._record_action(shot, issue_code, "Changed only the repeated effect combo.")

    def _boost_climax(self, shot: Dict[str, Any], issue_code: str) -> None:
        shot["effects"] = [
            {"name": "impact_flash_combo", "strength": 0.88},
            {"name": "zoom_burst_shake", "strength": 0.78},
        ]
        shot["transition"] = "flash_cut"
        self._add_reason(shot, "Boosted climax after quality judge feedback")
        self._record_action(shot, issue_code, "Boosted only the underpowered climax.")

    @staticmethod
    def _add_reason(shot: Dict[str, Any], reason: str) -> None:
        for log in shot.get("explainability_log", []):
            reasons = list(log.get("reason") or [])
            if reason not in reasons:
                reasons.append(reason)
            log["reason"] = reasons
        if not shot.get("explainability_log"):
            shot["explainability_log"] = [{"effect": "revision", "strength": 0.0, "reason": [reason]}]

    @staticmethod
    def _record_action(shot: Dict[str, Any], issue_code: str, action: str) -> None:
        actions = list(shot.get("revision_actions") or [])
        actions.append({"issue": issue_code, "action": action})
        shot["revision_actions"] = actions

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
