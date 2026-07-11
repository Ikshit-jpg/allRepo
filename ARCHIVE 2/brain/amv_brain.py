from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from .director import (
        EditingExperience,
        MotifMemory,
        NarrativeEntityTracker,
        ShotFeatures,
        StoryArcPlanner,
        attach_audio_features,
        extract_shot_features,
    )
    from .editorial_planner import EditorialPlanner
    from .effect_budget import EffectBudget
    from .effect_grammar import EffectGrammar
    from .narrative_director import NarrativeDirector
    from .quality_judge import QualityJudge
    from .revision_loop import RevisionLoop
    from .taste_system import TasteFilter
except ImportError:  # script-style execution from brain/
    from director import (  # type: ignore
        EditingExperience,
        MotifMemory,
        NarrativeEntityTracker,
        ShotFeatures,
        StoryArcPlanner,
        attach_audio_features,
        extract_shot_features,
    )
    from editorial_planner import EditorialPlanner  # type: ignore
    from effect_budget import EffectBudget  # type: ignore
    from effect_grammar import EffectGrammar  # type: ignore
    from narrative_director import NarrativeDirector  # type: ignore
    from quality_judge import QualityJudge  # type: ignore
    from revision_loop import RevisionLoop  # type: ignore
    from taste_system import TasteFilter  # type: ignore


class AMVBrain:
    """
    Full AMV brain pipeline:

    Scene + Music Analysis -> Story Arc Planner -> Narrative Director ->
    Editorial Planner -> Effect Budget -> Effect Grammar -> Taste Filter ->
    Quality Judge -> Revision Loop.
    """

    def __init__(
        self,
        seed: int = 42,
        personality_weights: Optional[Dict[str, float]] = None,
        revision_passes: int = 3,
        experience_dir: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.personality_weights = personality_weights or {}
        self.story_arc_planner = StoryArcPlanner()
        self.narrative_entity_tracker = NarrativeEntityTracker()
        self.motif_memory = MotifMemory()
        self.narrative_director = NarrativeDirector()
        self.effect_budget = EffectBudget()
        self.effect_grammar = EffectGrammar()
        self.quality_judge = QualityJudge()
        self.revision_loop = RevisionLoop(passes=revision_passes)
        self.quality_threshold = 0.82
        self.experience = EditingExperience(experience_dir) if experience_dir else None
        self.last_pipeline_report: Dict[str, Any] = {}

    def plan(self, shots: List[ShotFeatures], bpm: float = 130.0) -> List[Dict[str, Any]]:
        story = self.story_arc_planner.plan(shots)
        entity_timeline = self.narrative_entity_tracker.track(shots)
        motifs = self.motif_memory.observe(shots)
        strategies = self.narrative_director.direct(shots, story, entity_timeline, motifs)
        entities_by_index = self._entities_for_shots(shots, entity_timeline)

        timeline = EditorialPlanner(bpm=bpm, seed=self.seed).plan(
            shots=shots,
            story=story,
            entities_by_index=entities_by_index,
            strategies=strategies,
            motifs=motifs,
        )

        timeline = self.effect_budget.apply(timeline)
        timeline = self.effect_grammar.apply(timeline)
        energy_arc = [self._float_value(shot.get("energy"), 0.5) for shot in timeline]
        timeline = TasteFilter(
            seed=self.seed,
            personality_weights=self.personality_weights,
        ).filter_timeline(timeline, energy_arc)
        self._guard_taste_with_budget(timeline)
        self._annotate_taste_verdicts(timeline)
        self._refresh_explainability(timeline)
        timeline = self.effect_grammar.apply(timeline)
        first_report = self.quality_judge.evaluate(timeline)

        if first_report.score < self.quality_threshold:
            timeline = self.revision_loop.run(timeline, initial_report=first_report)
            self._guard_taste_with_budget(timeline)
            self._annotate_taste_verdicts(timeline)
            self._refresh_explainability(timeline)
            timeline = self.effect_grammar.apply(timeline)
            self._guard_taste_with_budget(timeline)
            self._annotate_taste_verdicts(timeline)
            final_report = self.quality_judge.evaluate(timeline)
        else:
            self.revision_loop.last_reports = [first_report]
            final_report = first_report

        self.last_pipeline_report = {
            "architecture": [
                "scene_music_analysis",
                "story_arc_planner",
                "narrative_director",
                "editorial_planner",
                "effect_budget",
                "effect_grammar",
                "taste_filter",
                "quality_judge",
                "revision_loop",
            ],
            "initial_quality": first_report.to_dict(),
            "final_quality": final_report.to_dict(),
            "revision_passes": [report.to_dict() for report in self.revision_loop.last_reports],
        }

        for shot in timeline:
            shot["brain_pipeline"] = self.last_pipeline_report["architecture"]
            shot["quality_report"] = final_report.to_dict()

        if self.experience:
            session_id = self.experience.record_session(timeline, self.personality_weights)
            for shot in timeline:
                shot["experience_session_id"] = session_id

        return timeline

    def run(self, shots: List[ShotFeatures], bpm: float = 130.0) -> List[Dict[str, Any]]:
        return self.plan(shots, bpm=bpm)

    def to_json(self, timeline: List[Dict[str, Any]]) -> str:
        return json.dumps(timeline, indent=2)

    @staticmethod
    def _entities_for_shots(shots: List[ShotFeatures], entity_timeline: Any) -> List[List[Any]]:
        if not shots:
            return []

        sorted_states = sorted(entity_timeline.states.keys())
        shot_positions = sorted(enumerate(shots), key=lambda item: item[1].start)
        entities_by_index: List[List[Any]] = [[] for _ in shots]
        state_idx = 0
        prev_time: Optional[float] = None
        prev_entities: List[Any] = []

        for shot_idx, shot in shot_positions:
            while state_idx < len(sorted_states) and sorted_states[state_idx] <= shot.start:
                prev_time = sorted_states[state_idx]
                prev_entities = list(entity_timeline.states.get(prev_time, []))
                state_idx += 1

            next_time = sorted_states[state_idx] if state_idx < len(sorted_states) else None
            if prev_time is None:
                chosen = list(entity_timeline.states.get(next_time, [])) if next_time is not None else []
            elif next_time is None:
                chosen = list(prev_entities)
            elif shot.start - prev_time <= next_time - shot.start:
                chosen = list(prev_entities)
            else:
                chosen = list(entity_timeline.states.get(next_time, []))

            entities_by_index[shot_idx] = chosen

        return entities_by_index

    @staticmethod
    def _annotate_taste_verdicts(timeline: List[Dict[str, Any]]) -> None:
        for shot in timeline:
            for effect in shot.get("effects", []):
                if effect.get("budget_guarded"):
                    effect["taste_verdict"] = "adjusted"
                elif effect.get("taste_verdict") in {"approved", "adjusted", "passthrough"}:
                    continue
                else:
                    effect["taste_verdict"] = "approved"

    @staticmethod
    def _refresh_explainability(timeline: List[Dict[str, Any]]) -> None:
        for shot in timeline:
            strategy = shot.get("strategy") or {}
            budget = shot.get("effect_budget") or {}
            reasons = []
            reasons.extend(strategy.get("reasons") or [])
            if shot.get("is_drop") or shot.get("intent") == "impact":
                reasons.append("Beat drop")
            act = ((shot.get("narrative") or {}).get("act") or {}).get("act_type")
            if act == "climax":
                reasons.append("Story climax")
            if budget:
                reasons.extend(budget.get("reasons") or [])
            if not reasons:
                reasons.append("Passed editorial planning")

            shot["explainability_log"] = [
                {
                    "effect": effect.get("name", "effect"),
                    "strength": effect.get("strength", 0.0),
                    "reason": list(dict.fromkeys(reasons + [
                        {
                            "approved": "Passed taste filter",
                            "adjusted": "Taste filter reduced strength",
                            "passthrough": "Not evaluated by taste filter",
                        }.get(effect.get("taste_verdict", "unreviewed"), "Unreviewed")
                    ])),
                }
                for effect in shot.get("effects", [])
            ]

    @staticmethod
    def _guard_taste_with_budget(timeline: List[Dict[str, Any]]) -> None:
        for shot in timeline:
            budget = shot.get("effect_budget") or {}
            ceiling = AMVBrain._float_value(budget.get("budget"), 1.0)
            if shot.get("is_drop") or shot.get("intent") == "impact":
                ceiling = max(ceiling, 0.78)
            if shot.get("has_speech"):
                ceiling = min(ceiling, 0.42)
            for effect in shot.get("effects", []):
                strength = AMVBrain._float_value(effect.get("strength"), 0.0)
                if strength > ceiling:
                    effect["strength"] = round(ceiling, 3)
                    effect["budget_guarded"] = True

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


def plan_amv_edit(
    video_path: str,
    beat_analysis: Any = None,
    bpm: float = 130.0,
    seed: int = 42,
    personality_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    shots = extract_shot_features(video_path)
    if beat_analysis is not None:
        shots = attach_audio_features(shots, beat_analysis)
        if isinstance(beat_analysis, dict):
            bpm = float(beat_analysis.get("bpm") or beat_analysis.get("tempo") or bpm)
        else:
            bpm = float(getattr(beat_analysis, "bpm", None) or getattr(beat_analysis, "tempo", None) or bpm)
    return AMVBrain(seed=seed, personality_weights=personality_weights).plan(shots, bpm=bpm)
