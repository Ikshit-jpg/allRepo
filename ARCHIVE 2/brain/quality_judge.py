from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


HEAVY_EFFECTS = {
    "impact_flash_combo",
    "zoom_burst_shake",
    "god_of_war",
    "camera_slam",
    "flash_cut",
    "glitch_cut",
}

FLASH_EFFECTS = {"impact_flash_combo", "flash_cut", "flash", "white_flash"}


@dataclass
class QualityIssue:
    code: str
    shot_index: int
    severity: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    score: float
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= 0.82 and not any(issue.severity >= 0.75 for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


class QualityJudge:
    """Transparent, measurable quality judge for pre-render timeline checks."""

    def evaluate(self, timeline: List[Dict[str, Any]]) -> QualityReport:
        issues: List[QualityIssue] = []
        if not timeline:
            return QualityReport(score=0.0, issues=[QualityIssue("empty", 0, 1.0, "Timeline is empty.")])

        self._check_repetition(timeline, issues)
        self._check_speech_safety(timeline, issues)
        self._check_heavy_spacing(timeline, issues)
        self._check_flash_overuse(timeline, issues)
        self._check_climax_power(timeline, issues)
        self._check_density(timeline, issues)

        penalty = sum(issue.severity for issue in issues) / max(4.0, len(timeline))
        score = max(0.0, min(1.0, 1.0 - penalty))
        metrics = {
            "issue_count": float(len(issues)),
            "average_density": round(self._average_density(timeline), 4),
            "heavy_count": float(sum(self._has_heavy(shot) for shot in timeline)),
        }
        return QualityReport(score=score, issues=issues, metrics=metrics)

    def _check_repetition(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        last_sets: List[tuple[str, ...]] = []
        last_transitions: List[str] = []
        for idx, shot in enumerate(timeline):
            effect_set = tuple(sorted(str(effect.get("name") or "") for effect in shot.get("effects", [])))
            transition = str(shot.get("transition") or "cut")
            last_sets.append(effect_set)
            last_transitions.append(transition)
            if idx >= 2 and last_sets[-1] == last_sets[-2] == last_sets[-3] and effect_set:
                issues.append(QualityIssue("repeated_effect_combo", idx, 0.72, "Same effect combo repeated 3 shots in a row."))
            if idx >= 2 and last_transitions[-1] == last_transitions[-2] == last_transitions[-3]:
                issues.append(QualityIssue("repeated_transition", idx, 0.58, "Same transition repeated 3 shots in a row."))

    def _check_speech_safety(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        for idx, shot in enumerate(timeline):
            if not shot.get("has_speech"):
                continue
            if self._has_heavy(shot):
                issues.append(QualityIssue("speech_heavy_effect", idx, 0.86, "Heavy effect used during speech/dialogue."))
            if self._effect_power(shot) > 0.52:
                issues.append(QualityIssue("speech_too_loud", idx, 0.62, "Dialogue shot effect power is too high."))

    def _check_heavy_spacing(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        last_heavy = -99
        for idx, shot in enumerate(timeline):
            if not self._has_heavy(shot):
                continue
            if idx - last_heavy < 2:
                issues.append(QualityIssue("heavy_too_close", idx, 0.66, "Heavy effects are too close together."))
            last_heavy = idx

    def _check_flash_overuse(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        recent: List[bool] = []
        for idx, shot in enumerate(timeline):
            has_flash = self._has_flash(shot)
            recent.append(has_flash)
            window = recent[-4:]
            if len(window) >= 3 and sum(1 for item in window if item) >= 3:
                issues.append(QualityIssue("too_many_flashes", idx, 0.74, "Too many flash effects in a short window."))

    def _check_climax_power(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        climax_idx = max(range(len(timeline)), key=lambda i: self._climax_score(timeline[i]))
        climax_power = self._effect_power(timeline[climax_idx])
        earlier_peak = max((self._effect_power(shot) for shot in timeline[:climax_idx]), default=0.0)
        if earlier_peak > climax_power + 0.08:
            issues.append(QualityIssue("climax_not_strongest", climax_idx, 0.82, "Earlier shot spends more power than the climax."))

    def _check_density(self, timeline: List[Dict[str, Any]], issues: List[QualityIssue]) -> None:
        for idx, shot in enumerate(timeline):
            count = len(shot.get("effects", []))
            power = self._effect_power(shot)
            if count >= 3 and power > 0.76 and not (shot.get("is_drop") or shot.get("intent") == "impact"):
                issues.append(QualityIssue("effect_density_high", idx, 0.52, "Non-impact shot has too many strong effects."))

    @staticmethod
    def _has_heavy(shot: Dict[str, Any]) -> bool:
        return any(effect.get("name") in HEAVY_EFFECTS for effect in shot.get("effects", []))

    @staticmethod
    def _has_flash(shot: Dict[str, Any]) -> bool:
        if str(shot.get("transition") or "") in FLASH_EFFECTS:
            return True
        return any(effect.get("name") in FLASH_EFFECTS for effect in shot.get("effects", []))

    @staticmethod
    def _effect_power(shot: Dict[str, Any]) -> float:
        effects = shot.get("effects", [])
        if not effects:
            return 0.0
        base = sum(float(effect.get("strength", 0.0) or 0.0) for effect in effects) / max(1, len(effects))
        heavy_bonus = 0.18 if any(effect.get("name") in HEAVY_EFFECTS for effect in effects) else 0.0
        return min(1.0, base + heavy_bonus)

    @staticmethod
    def _climax_score(shot: Dict[str, Any]) -> float:
        act = ((shot.get("narrative") or {}).get("act") or {}).get("act_type", "")
        return (
            float(shot.get("energy", 0.0) or 0.0)
            + (0.35 if shot.get("is_drop") else 0.0)
            + (0.25 if shot.get("intent") == "impact" else 0.0)
            + (0.25 if act == "climax" else 0.0)
        )

    def _average_density(self, timeline: List[Dict[str, Any]]) -> float:
        return sum(self._effect_power(shot) for shot in timeline) / max(1, len(timeline))
