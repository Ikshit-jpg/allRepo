from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .director import (
        DirectingDecision,
        EntityTimeline,
        NarrativePlan,
        ShotFeatures,
        VisualMotif,
    )
except ImportError:  # script-style execution from brain/
    from director import (  # type: ignore
        DirectingDecision,
        EntityTimeline,
        NarrativePlan,
        ShotFeatures,
        VisualMotif,
    )


@dataclass
class NarrativeStrategy:
    """Per-shot strategy. This says why the edit should behave a certain way."""

    shot_start: float
    shot_end: float
    act_type: str
    payoff_state: str = "neutral"
    intensity_ceiling: float = 0.55
    hold_priority: float = 0.0
    surprise: bool = False
    contrast: bool = False
    decision: DirectingDecision = field(default_factory=DirectingDecision)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.to_dict()
        return data


class NarrativeDirector:
    """
    Strategy layer between Story Arc Planner and Editorial Planner.

    It does not choose exact effects. It chooses restraint, payoff timing,
    hold/surprise behavior, and the amount of power this moment is allowed to spend.
    """

    def __init__(self) -> None:
        self.last_strategies: Dict[float, NarrativeStrategy] = {}

    def direct(
        self,
        shots: List[ShotFeatures],
        story: NarrativePlan,
        entities: EntityTimeline,
        motifs: List[VisualMotif],
    ) -> Dict[float, NarrativeStrategy]:
        strategies: Dict[float, NarrativeStrategy] = {}
        total = max(1, len(shots))

        for idx, shot in enumerate(shots):
            act = story.act_for(shot)
            distance_to_climax = story.climax_index - idx
            relationship = entities.relationship_at(shot.start)
            callback = self._callback_motif(shot, motifs)
            arc_pos = idx / max(1, total - 1)

            decision = DirectingDecision(relationship_mode=relationship)
            strategy = NarrativeStrategy(
                shot_start=shot.start,
                shot_end=shot.end,
                act_type=act.act_type,
                payoff_state="build" if distance_to_climax > 0 else "release",
                intensity_ceiling=self._base_ceiling(act.act_type, arc_pos),
                decision=decision,
            )

            if 0 < distance_to_climax <= 2:
                decision.save_best_moment = True
                decision.delay_climax = True
                decision.build_suspense = True
                strategy.payoff_state = "delay"
                strategy.intensity_ceiling = min(strategy.intensity_ceiling, 0.46)
                strategy.contrast = shot.audio_energy < 0.72
                strategy.reasons.append("Save the strongest effects for the upcoming climax.")

            if idx == story.climax_index:
                strategy.payoff_state = "climax"
                strategy.intensity_ceiling = 1.0
                strategy.surprise = shot.audio_energy >= 0.72 or shot.is_drop
                strategy.reasons.append("Release the reserved impact at the story climax.")

            if shot.has_speech or shot.duration >= 1.8 or shot.composition_score > 0.78:
                decision.hold_shot = True
                strategy.hold_priority = max(strategy.hold_priority, 0.72)
                strategy.intensity_ceiling = min(strategy.intensity_ceiling, 0.58)
                strategy.reasons.append("Hold the shot so the viewer can read the moment.")

            if relationship == "confrontation" and strategy.payoff_state != "climax":
                decision.build_suspense = True
                strategy.payoff_state = "tension"
                strategy.reasons.append("Build suspense because the entities imply confrontation.")

            if callback:
                decision.callback_motif = callback
                strategy.reasons.append(f"Callback visual motif: {callback}.")

            if strategy.contrast:
                decision.create_contrast = True
                strategy.reasons.append("Create contrast before the payoff.")

            if not strategy.reasons:
                strategy.reasons.append("Support the current story act without stealing focus.")

            decision.reason = " ".join(strategy.reasons)
            strategies[shot.start] = strategy

        self.last_strategies = strategies
        return strategies

    @staticmethod
    def _base_ceiling(act_type: str, arc_pos: float) -> float:
        if act_type == "climax":
            return 1.0
        if act_type == "climax_resolution":
            return 0.72
        if act_type == "build":
            return 0.42 + min(0.24, arc_pos * 0.30)
        if act_type == "opening":
            return 0.34
        return 0.48

    @staticmethod
    def _callback_motif(shot: ShotFeatures, motifs: List[VisualMotif]) -> Optional[str]:
        for motif in motifs:
            if motif.motif_id == "dark" and shot.brightness < 0.45:
                return motif.motif_id
            if motif.motif_id == "bright" and shot.brightness > 0.55:
                return motif.motif_id
            if motif.motif_id == "high_motion" and shot.motion > 0.55:
                return motif.motif_id
        return None
