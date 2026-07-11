from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .director import (
        CinematicGrammarEngine,
        Director,
        IntentInference,
        MemoryState,
        NarrativeEntity,
        NarrativeMemory,
        NarrativePlan,
        SegmentSemantics,
        ShotFeatures,
        TimelinePlanner,
        VisualMotif,
    )
    from .narrative_director import NarrativeStrategy
except ImportError:  # script-style execution from brain/
    from director import (  # type: ignore
        CinematicGrammarEngine,
        Director,
        IntentInference,
        MemoryState,
        NarrativeEntity,
        NarrativeMemory,
        NarrativePlan,
        SegmentSemantics,
        ShotFeatures,
        TimelinePlanner,
        VisualMotif,
    )
    from narrative_director import NarrativeStrategy  # type: ignore


class EditorialPlanner:
    """
    Execution layer. It turns narrative strategy into cuts, intents, camera moves,
    effects, transitions and renderer-friendly timeline JSON.
    """

    def __init__(self, bpm: float = 130.0, seed: int = 42) -> None:
        self.bpm = bpm
        self.seed = seed
        self.intent_model = IntentInference()
        self.grammar = CinematicGrammarEngine()
        self.timeline_planner = TimelinePlanner(bpm=bpm, seed=seed)

    def plan(
        self,
        shots: List[ShotFeatures],
        story: NarrativePlan,
        strategies: Dict[float, NarrativeStrategy],
        motifs: List[VisualMotif],
        entities_by_start: Optional[Dict[float, List[NarrativeEntity]]] = None,
        entities_by_index: Optional[List[List[NarrativeEntity]]] = None,
    ) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []
        memory = NarrativeMemory()
        total = len(shots)
        entities_by_start = entities_by_start or {}

        for idx, shot in enumerate(shots):
            mem = memory.read(idx, total)
            semantics = Director._semantics(shot, idx, total, mem)
            strategy = strategies.get(shot.start)
            directing = strategy.decision if strategy else None
            intent = self._intent_for(shot, semantics, mem, strategy)
            actions = self.grammar.actions_for(intent, shot, semantics, directing)
            self._apply_strategy_to_actions(actions, shot, strategy)

            segment = self.timeline_planner.build_segment(
                shot=shot,
                intent=intent,
                semantics=semantics,
                memory=mem,
                actions=actions,
                narrative_plan=story,
                entities=self._entities_for(idx, shot, entities_by_start, entities_by_index),
                directing=directing,  # type: ignore[arg-type]
                motifs=motifs,
            )
            segment["strategy"] = strategy.to_dict() if strategy else {}
            segment["pipeline_stage"] = "editorial_planner"
            timeline.append(segment)
            memory.update(intent, segment["effects"], shot.audio_energy)

        return self.timeline_planner.finalize(timeline)

    def _intent_for(
        self,
        shot: ShotFeatures,
        semantics: SegmentSemantics,
        memory: MemoryState,
        strategy: Optional[NarrativeStrategy],
    ) -> str:
        if strategy is None:
            return self.intent_model.infer(shot, semantics, memory, None)
        if strategy.payoff_state == "climax":
            return "impact"
        if strategy.payoff_state == "delay":
            return "anticipation"
        if strategy.payoff_state == "tension":
            return "tension"
        return self.intent_model.infer(shot, semantics, memory, strategy.decision)

    @staticmethod
    def _apply_strategy_to_actions(
        actions: Dict[str, Any],
        shot: ShotFeatures,
        strategy: Optional[NarrativeStrategy],
    ) -> None:
        if strategy is None:
            return

        ceiling = max(0.1, min(1.0, strategy.intensity_ceiling))
        for effect in actions.get("effects", []):
            effect["strength"] = round(min(float(effect.get("strength", 0.0)), ceiling), 3)

        if strategy.hold_priority >= 0.70:
            actions["transition"] = "cut" if shot.has_speech else actions.get("transition", "cut")
            camera = actions.setdefault("camera", {})
            if shot.motion < 0.35:
                camera["motion"] = "static"
                actions["camera_move"] = "static"

        if strategy.surprise:
            actions["transition"] = "flash_cut"

        if strategy.contrast and actions.get("effects"):
            for effect in actions["effects"]:
                effect["strength"] = round(float(effect.get("strength", 0.0)) * 0.82, 3)

    @staticmethod
    def _entities_for(
        idx: int,
        shot: ShotFeatures,
        entities_by_start: Dict[float, List[NarrativeEntity]],
        entities_by_index: Optional[List[List[NarrativeEntity]]],
    ) -> List[NarrativeEntity]:
        if entities_by_index is not None and idx < len(entities_by_index):
            return entities_by_index[idx]
        return entities_by_start.get(shot.start, [])
