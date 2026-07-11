"""
director.py
===========
THE DIRECTOR MODEL (heuristic v1, model-ready interface).

Input:  per-shot features + music beat/energy analysis
Output: structured JSON timeline, one entry per shot:
        {start, end, intent, energy, is_drop, has_speech,
         camera_move, transition, effects:[{name, strength}]}

The raw plan is then passed through taste_system.TasteFilter, which
enforces variety, intensity ramps and restraint. The renderer
(engine/ or archive_pyav_renderer.py) consumes the JSON directly.

Designed so the heuristic `plan()` can later be swapped for a
Transformer sequence model without changing callers.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    from .taste_system import TasteFilter, rhythm_durations
except ImportError:  # script-style execution
    from taste_system import TasteFilter, rhythm_durations


# ---------------------------------------------------------------------------
# Shot features (the "brain input")
# ---------------------------------------------------------------------------

@dataclass
class ShotFeatures:
    start: float
    end: float
    motion: float = 0.5        # 0..1 optical-flow intensity
    brightness: float = 0.5
    audio_energy: float = 0.5  # 0..1 RMS in this window
    on_beat: bool = False
    is_drop: bool = False
    has_faces: bool = False
    has_speech: bool = False
    face_count: int = 0
    subject_confidence: float = 0.0
    composition_score: float = 0.5
    visual_importance: float = 0.5
    shot_type: str = "wide"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SegmentSemantics:
    """Higher-level meaning derived from scene, audio and context."""

    tension: float = 0.0
    anticipation: float = 0.0
    narrative_importance: float = 0.0
    viewer_focus: float = 0.5
    fatigue: float = 0.0
    mood: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoryAct:
    start: float
    end: float
    act_type: str
    intensity_budget: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativePlan:
    """Global story arc: where to save energy and where to spend it."""

    acts: List[StoryAct]
    climax_index: int
    climax_point: float
    climax_budget: float

    def act_for(self, shot: ShotFeatures) -> StoryAct:
        center = (shot.start + shot.end) / 2.0
        for act in self.acts:
            if act.start <= center <= act.end:
                return act
        return self.acts[-1]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["acts"] = [act.to_dict() for act in self.acts]
        return data


@dataclass
class NarrativeEntity:
    entity_id: str
    entity_type: str
    prominence: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityTimeline:
    """Tracks characters, weapons, important objects, symbols and locations."""

    states: Dict[float, List[NarrativeEntity]] = field(default_factory=dict)
    relationships: Dict[float, str] = field(default_factory=dict)

    def entities_at(self, start: float) -> List[NarrativeEntity]:
        if not self.states:
            return []
        closest = min(self.states, key=lambda t: abs(t - start))
        return self.states.get(closest, [])

    def relationship_at(self, start: float) -> str:
        if not self.relationships:
            return "none"
        closest = min(self.relationships, key=lambda t: abs(t - start))
        return self.relationships.get(closest, "none")


@dataclass
class VisualMotif:
    motif_id: str
    motif_type: str
    first_seen: float
    last_seen: float
    strength: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DirectingDecision:
    save_best_moment: bool = False
    delay_climax: bool = False
    build_suspense: bool = False
    hold_shot: bool = False
    create_contrast: bool = False
    callback_motif: Optional[str] = None
    relationship_mode: str = "none"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryState:
    last_intents: List[str] = field(default_factory=list)
    last_effects: List[str] = field(default_factory=list)
    fatigue: float = 0.0
    arc_position: float = 0.0
    scene_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NarrativeMemory:
    """Short/medium context used by the director before taste filtering."""

    def __init__(self, short_window: int = 3, scene_window: int = 12) -> None:
        self.short_window = max(1, short_window)
        self.scene_window = max(self.short_window, scene_window)
        self._intents: deque[str] = deque(maxlen=self.scene_window)
        self._effects: deque[str] = deque(maxlen=self.scene_window)
        self._energies: deque[float] = deque(maxlen=self.scene_window)
        self._fatigue = 0.0

    def read(self, idx: int, total: int) -> MemoryState:
        arc_position = idx / max(1, total - 1)
        energy = sum(self._energies) / len(self._energies) if self._energies else 0.0
        return MemoryState(
            last_intents=list(self._intents)[-self.short_window:],
            last_effects=list(self._effects)[-self.short_window:],
            fatigue=round(self._fatigue, 4),
            arc_position=round(arc_position, 4),
            scene_energy=round(energy, 4),
        )

    def update(self, intent: str, effects: List[Dict[str, Any]], energy: float) -> None:
        self._intents.append(intent)
        for effect in effects:
            name = str(effect.get("name") or "")
            if name:
                self._effects.append(name)
        self._energies.append(float(max(0.0, min(1.0, energy))))
        effort = float(max(0.0, min(1.0, energy)))
        effort += 0.16 * sum(1 for effect in effects if float(effect.get("strength", 0.0) or 0.0) >= 0.65)
        self._fatigue = max(0.0, min(1.0, (self._fatigue * 0.72) + (effort * 0.28) - 0.08))


class StoryArcPlanner:
    """Pre-pass story planner. Predicts payoff instead of only detecting it."""

    def __init__(self) -> None:
        self.last_plan: Optional[NarrativePlan] = None

    def plan(self, shots: List[ShotFeatures]) -> NarrativePlan:
        if not shots:
            plan = NarrativePlan([StoryAct(0.0, 0.0, "empty", 0.0)], 0, 0.0, 0.0)
            self.last_plan = plan
            return plan

        scores = [
            (shot.audio_energy * 0.45)
            + (shot.motion * 0.25)
            + (0.20 if shot.is_drop else 0.0)
            + (shot.visual_importance * 0.10)
            for shot in shots
        ]
        climax_index = max(range(len(shots)), key=lambda idx: scores[idx])
        climax = shots[climax_index]
        total_start, total_end = shots[0].start, shots[-1].end
        duration = max(0.001, total_end - total_start)
        first_cut = total_start + duration * 0.25
        second_cut = total_start + duration * 0.72

        acts = [
            StoryAct(total_start, first_cut, "opening", 0.18),
            StoryAct(first_cut, second_cut, "build", 0.42),
            StoryAct(second_cut, total_end, "climax_resolution", 0.78),
        ]
        for act in acts:
            if act.start <= climax.start <= act.end:
                act.intensity_budget = 1.0
                act.act_type = "climax"

        plan = NarrativePlan(
            acts=acts,
            climax_index=climax_index,
            climax_point=round((climax.start + climax.end) / 2.0, 3),
            climax_budget=1.0,
        )
        self.last_plan = plan
        return plan


class NarrativeEntityTracker:
    """Tracks characters plus weapons, objects, symbols and locations."""

    def __init__(self) -> None:
        self.last_timeline: Optional[EntityTimeline] = None

    def track(self, shots: List[ShotFeatures]) -> EntityTimeline:
        timeline = EntityTimeline()
        for shot in shots:
            entities: List[NarrativeEntity] = []
            if shot.has_faces or shot.face_count > 0:
                count = max(1, shot.face_count)
                for idx in range(count):
                    entities.append(NarrativeEntity(
                        entity_id=f"character_{idx + 1}",
                        entity_type="character",
                        prominence=max(shot.subject_confidence, 0.45),
                        attributes={"shot_type": shot.shot_type},
                    ))
            if shot.motion >= 0.72 and shot.visual_importance >= 0.55:
                entities.append(NarrativeEntity(
                    entity_id="weapon_or_power_object",
                    entity_type="weapon",
                    prominence=round(min(1.0, shot.motion), 3),
                    attributes={"inferred": True},
                ))
            if shot.brightness < 0.30:
                entities.append(NarrativeEntity(
                    entity_id="dark_symbol",
                    entity_type="symbol",
                    prominence=round(1.0 - shot.brightness, 3),
                ))
            if shot.shot_type in {"wide", "establishing"}:
                entities.append(NarrativeEntity(
                    entity_id="location",
                    entity_type="location",
                    prominence=max(0.3, shot.composition_score),
                    attributes={"shot_type": shot.shot_type},
                ))

            timeline.states[shot.start] = entities
            if len([e for e in entities if e.entity_type == "character"]) >= 2 and shot.motion > 0.55:
                timeline.relationships[shot.start] = "confrontation"
            elif shot.has_speech:
                timeline.relationships[shot.start] = "dialogue"
            elif len(entities) == 1:
                timeline.relationships[shot.start] = "isolation"
            else:
                timeline.relationships[shot.start] = "environment"
        self.last_timeline = timeline
        return timeline


class MotifMemory:
    """Long-term visual storytelling memory for recurring looks and callbacks."""

    def __init__(self) -> None:
        self.motifs: List[VisualMotif] = []

    def observe(self, shots: List[ShotFeatures]) -> List[VisualMotif]:
        self.motifs = []
        if not shots:
            return self.motifs
        buckets: Dict[str, List[ShotFeatures]] = {"dark": [], "bright": [], "high_motion": []}
        for shot in shots:
            if shot.brightness < 0.35:
                buckets["dark"].append(shot)
            if shot.brightness > 0.68:
                buckets["bright"].append(shot)
            if shot.motion > 0.65:
                buckets["high_motion"].append(shot)
        for name, seen in buckets.items():
            if len(seen) >= 2:
                self.motifs.append(VisualMotif(
                    motif_id=name,
                    motif_type="color_grade" if name in {"dark", "bright"} else "motion",
                    first_seen=seen[0].start,
                    last_seen=seen[-1].start,
                    strength=round(min(1.0, len(seen) / max(2, len(shots))), 3),
                ))
        return self.motifs

    def callback_for(self, shot: ShotFeatures, plan: NarrativePlan) -> Optional[VisualMotif]:
        act = plan.act_for(shot)
        if act.act_type not in {"climax", "climax_resolution"}:
            return None
        for motif in self.motifs:
            if motif.motif_id == "dark" and shot.brightness < 0.45:
                return motif
            if motif.motif_id == "bright" and shot.brightness > 0.55:
                return motif
            if motif.motif_id == "high_motion" and shot.motion > 0.55:
                return motif
        return None


class NarrativeDirector:
    """Makes directing decisions before the intent generator chooses effects."""

    def __init__(self) -> None:
        self.last_decisions: Dict[float, DirectingDecision] = {}

    def decide(
        self,
        shots: List[ShotFeatures],
        plan: NarrativePlan,
        entities: EntityTimeline,
        motifs: List[VisualMotif],
    ) -> Dict[float, DirectingDecision]:
        decisions: Dict[float, DirectingDecision] = {}
        for idx, shot in enumerate(shots):
            distance_to_climax = plan.climax_index - idx
            relationship = entities.relationship_at(shot.start)
            callback = next((m.motif_id for m in motifs if m.last_seen >= shot.start), None)
            decision = DirectingDecision(relationship_mode=relationship)
            if 0 < distance_to_climax <= 2:
                decision.save_best_moment = True
                decision.delay_climax = True
                decision.build_suspense = True
                decision.create_contrast = shot.audio_energy < 0.72
                decision.reason = "Saving intensity for the upcoming climax."
            elif idx == plan.climax_index:
                decision.reason = "Climax: release the reserved impact."
            elif shot.has_speech or shot.duration >= 1.8 or shot.composition_score > 0.78:
                decision.hold_shot = True
                decision.reason = "Hold the shot so the moment can breathe."
            elif relationship == "confrontation":
                decision.build_suspense = True
                decision.reason = "Entity relationship suggests confrontation."
            if callback:
                decision.callback_motif = callback
                if not decision.reason:
                    decision.reason = f"Callback motif: {callback}."
            decisions[shot.start] = decision
        self.last_decisions = decisions
        return decisions


class IntentInference:
    """Model-ready multimodal intent inference.

    This is deterministic today, but the class boundary mirrors a future
    classifier/LLM step: scene + audio + memory -> viewer intent.
    """

    VALID_INTENTS = {
        "introduce",
        "build-up",
        "anticipation",
        "tension",
        "impact",
        "release",
        "calm",
        "dialogue",
        "transition",
    }

    def infer(
        self,
        shot: ShotFeatures,
        semantics: SegmentSemantics,
        memory: MemoryState,
        directing: Optional[DirectingDecision] = None,
    ) -> str:
        directing = directing or DirectingDecision()
        if shot.has_speech:
            return "dialogue"
        if directing.delay_climax and not shot.is_drop:
            return "anticipation" if directing.build_suspense else "build-up"
        if directing.hold_shot and shot.audio_energy < 0.70:
            return "release"
        if directing.relationship_mode == "confrontation" and shot.motion >= 0.45:
            return "tension"
        if memory.fatigue >= 0.62 and shot.audio_energy < 0.72:
            return "release"
        if shot.is_drop or (shot.audio_energy >= 0.76 and shot.motion >= 0.5):
            return "impact"
        if semantics.anticipation >= 0.66:
            return "anticipation"
        if semantics.tension >= 0.62:
            return "tension"
        if memory.arc_position <= 0.08:
            return "introduce"
        if shot.audio_energy >= 0.48 and memory.arc_position <= 0.78:
            return "build-up"
        if memory.arc_position >= 0.90 or shot.audio_energy < 0.32:
            return "calm"
        return "transition"


class CinematicGrammarEngine:
    """Maps viewer intent to edit actions with film-grammar defaults."""

    _GRAMMAR: Dict[str, Dict[str, Any]] = {
        "introduce": {
            "camera": {"motion": "steady_wide", "angle": "eye"},
            "transition": "dissolve",
            "effects": {"cinematic_grade": 0.28},
        },
        "dialogue": {
            "camera": {"motion": "locked_portrait", "angle": "eye"},
            "transition": "cut",
            "effects": {"portrait_enhance": 0.28},
        },
        "anticipation": {
            "camera": {"motion": "slow_push", "angle": "low"},
            "transition": "cut",
            "effects": {"push_in_slow": 0.42, "atmospheric_grade": 0.22},
        },
        "tension": {
            "camera": {"motion": "slow_push", "angle": "dutch"},
            "transition": "cut",
            "effects": {"handheld_shake": 0.34, "chromatic": 0.18},
        },
        "build-up": {
            "camera": {"motion": "push_or_orbit", "angle": "eye"},
            "transition": "smear_cut",
            "effects": {"energy_aura": 0.38, "zoom_pulse": 0.24},
        },
        "impact": {
            "camera": {"motion": "snap_zoom", "angle": "eye"},
            "transition": "flash_cut",
            "effects": {"impact_flash_combo": 0.78, "zoom_burst_shake": 0.68},
        },
        "release": {
            "camera": {"motion": "slow_pull", "angle": "eye"},
            "transition": "dissolve",
            "effects": {"cinematic_grade": 0.24, "subtle_drift": 0.2},
        },
        "calm": {
            "camera": {"motion": "static", "angle": "eye"},
            "transition": "fade_black",
            "effects": {"cinematic_grade": 0.24},
        },
        "transition": {
            "camera": {"motion": "pan", "angle": "eye"},
            "transition": "whip_pan_right",
            "effects": {"parallax_zoom": 0.28},
        },
    }

    def actions_for(
        self,
        intent: str,
        shot: ShotFeatures,
        semantics: SegmentSemantics,
        directing: Optional[DirectingDecision] = None,
    ) -> Dict[str, Any]:
        directing = directing or DirectingDecision()
        base = self._GRAMMAR.get(intent, self._GRAMMAR["calm"])
        camera = dict(base["camera"])
        if directing.hold_shot:
            camera["motion"] = "static" if shot.motion < 0.25 else "slow_push"
        if camera["motion"] == "push_or_orbit":
            camera["motion"] = "orbit_right" if shot.motion < 0.42 else "slow_push"
        effects = []
        energy_boost = 0.55 + (0.45 * shot.audio_energy)
        focus_scale = 0.72 + (0.28 * semantics.viewer_focus)
        for name, strength in dict(base["effects"]).items():
            if directing.delay_climax and name in {"impact_flash_combo", "zoom_burst_shake"}:
                continue
            scaled = max(0.05, min(1.0, float(strength) * energy_boost * focus_scale))
            if directing.save_best_moment:
                scaled *= 0.72
            if directing.callback_motif and name in {"cinematic_grade", "atmospheric_grade", "energy_aura"}:
                scaled = min(1.0, scaled * 1.18)
            effects.append({"name": name, "strength": round(scaled, 3)})
        return {
            "camera": camera,
            "camera_move": camera["motion"],
            "transition": base["transition"],
            "effects": effects,
        }


class TimelinePlanner:
    """Collates intent, grammar and policies into renderer-friendly JSON."""

    def __init__(self, bpm: float, seed: int = 42) -> None:
        self.bpm = bpm
        self.seed = seed

    def build_segment(
        self,
        shot: ShotFeatures,
        intent: str,
        semantics: SegmentSemantics,
        memory: MemoryState,
        actions: Dict[str, Any],
        narrative_plan: NarrativePlan,
        entities: List[NarrativeEntity],
        directing: DirectingDecision,
        motifs: List[VisualMotif],
    ) -> Dict[str, Any]:
        act = narrative_plan.act_for(shot)
        effect_log = [
            self._explain_effect(effect, shot, intent, act, directing)
            for effect in actions["effects"]
        ]
        return {
            "start": shot.start,
            "end": shot.end,
            "start_sec": shot.start,
            "end_sec": shot.end,
            "intent": intent,
            "energy": round(float(shot.audio_energy), 4),
            "is_drop": shot.is_drop,
            "has_speech": shot.has_speech,
            "camera": actions["camera"],
            "camera_move": actions["camera_move"],
            "angle": actions["camera"].get("angle", "eye"),
            "transition": actions["transition"],
            "effects": actions["effects"],
            "explainability_log": effect_log,
            "narrative": {
                "act": act.to_dict(),
                "climax_point": narrative_plan.climax_point,
                "entities": [entity.to_dict() for entity in entities],
                "relationship": directing.relationship_mode,
                "directing_decision": directing.to_dict(),
                "motifs": [motif.to_dict() for motif in motifs],
            },
            "semantics": semantics.to_dict(),
            "memory": memory.to_dict(),
        }

    def finalize(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        intents = [t["intent"] for t in timeline]
        for t, dur in zip(timeline, rhythm_durations([], self.bpm, intents, seed=self.seed)):
            t["suggested_duration"] = dur
        return timeline

    @staticmethod
    def _explain_effect(
        effect: Dict[str, Any],
        shot: ShotFeatures,
        intent: str,
        act: StoryAct,
        directing: DirectingDecision,
    ) -> Dict[str, Any]:
        reasons = []
        if shot.is_drop or intent == "impact":
            reasons.append("Big beat")
        if act.act_type == "climax":
            reasons.append("Climax")
        if directing.save_best_moment or directing.delay_climax:
            reasons.append("Saved for this moment" if act.act_type == "climax" else "Climax delayed")
        if directing.hold_shot:
            reasons.append("Held shot")
        if directing.callback_motif:
            reasons.append(f"Motif callback: {directing.callback_motif}")
        if not reasons:
            reasons.append("Supports narrative intent")
        return {
            "effect": effect.get("name", "effect"),
            "strength": effect.get("strength", 0.0),
            "reason": reasons,
        }


def extract_shot_features(video_path: str,
                          shot_bounds: Optional[List[float]] = None,
                          analysis_fps: float = 4.0,
                          analysis_width: int = 320) -> List[ShotFeatures]:
    """Cheap + fast shot analysis (low res, low fps => quick renders).
    Uses OpenCV if available: histogram-diff shot detection + Farneback
    optical flow for motion. Falls back to fixed 2s shots otherwise."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [ShotFeatures(start=t, end=t + 2.0)
                for t in range(0, 60, 2)]  # graceful fallback

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / analysis_fps)))
    prev_small = None
    prev_hist = None
    frame_idx = 0
    cuts: List[float] = [0.0]
    motions: List[tuple] = []  # (time, motion 0..1)
    bright: List[tuple] = []

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = analysis_width / max(1, w)
            small = cv2.resize(frame, (analysis_width, max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            t = frame_idx / fps
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > 0.45 and t - cuts[-1] > 0.4:  # shot boundary
                    cuts.append(t)
            if prev_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
                mag = float(np.mean(np.linalg.norm(flow, axis=2)))
                motions.append((t, min(1.0, mag / 8.0)))
            bright.append((t, float(gray.mean()) / 255.0))
            prev_small, prev_hist = gray, hist
        frame_idx += 1
    duration = frame_idx / fps
    cap.release()

    bounds = shot_bounds or (cuts + [duration])
    shots: List[ShotFeatures] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        in_m = [m for t, m in motions if s <= t < e]
        in_b = [b for t, b in bright if s <= t < e]
        shots.append(ShotFeatures(
            start=round(s, 3), end=round(e, 3),
            motion=round(sum(in_m) / len(in_m), 3) if in_m else 0.5,
            brightness=round(sum(in_b) / len(in_b), 3) if in_b else 0.5,
            composition_score=0.58,
            visual_importance=0.5,
        ))
    return shots


def attach_audio_features(shots: List[ShotFeatures], beat_analysis: Any
                          ) -> List[ShotFeatures]:
    """Annotate shots with beat/drop/energy info from a BeatAnalysis
    (see archive_decision_brain.analyze_music)."""
    if isinstance(beat_analysis, dict):
        beats = list(beat_analysis.get("beats") or beat_analysis.get("beat_times") or [])
        drops = list(beat_analysis.get("drops") or [])
        curve = list(beat_analysis.get("energy_curve") or [])
        bpm = beat_analysis.get("bpm") or beat_analysis.get("tempo")
    else:
        beats = list(getattr(beat_analysis, "beats", []) or getattr(beat_analysis, "beat_times", []) or [])
        drops = list(getattr(beat_analysis, "drops", []) or [])
        curve = list(getattr(beat_analysis, "energy_curve", []) or [])
        bpm = getattr(beat_analysis, "bpm", None) or getattr(beat_analysis, "tempo", None)
    numeric_curve: List[float] = []
    timed_curve: List[tuple[float, float]] = []
    for idx, item in enumerate(curve):
        if isinstance(item, dict):
            try:
                timed_curve.append((float(item.get("t", idx / 10.0)), float(item.get("energy", 0.5))))
            except Exception:
                continue
        else:
            try:
                numeric_curve.append(float(item))
            except Exception:
                continue
    peak = max(numeric_curve) if numeric_curve else max((e for _, e in timed_curve), default=1.0)
    for sh in shots:
        sh.on_beat = any(abs(sh.start - b) < 0.1 for b in beats)
        sh.is_drop = any(sh.start - 0.3 <= d <= sh.end for d in drops)
        if timed_curve:
            seg = [e for t, e in timed_curve if sh.start <= t < sh.end]
            if seg:
                sh.audio_energy = round(min(1.0, (sum(seg) / len(seg)) / max(1e-6, peak)), 3)
        elif numeric_curve:
            i0 = max(0, min(len(numeric_curve) - 1, int(sh.start * 10)))
            i1 = max(i0 + 1, min(len(numeric_curve), int(sh.end * 10)))
            seg = numeric_curve[i0:i1]
            sh.audio_energy = round(min(1.0, (sum(seg) / len(seg)) /
                                        max(1e-6, peak)), 3)
        if bpm:
            sh.on_beat = sh.on_beat or any(abs(sh.start - b) < max(0.08, 30.0 / float(bpm)) for b in beats)
    return shots


class EditingExperience:
    """Explicit-only learning.

    Sessions are recorded for review, but behavior never changes during edits.
    Feedback is submitted separately, learning runs offline, and callers must
    opt in to any returned profile.
    """

    def __init__(self, storage_dir: str = ".director_brain_experience") -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.sessions_path = os.path.join(self.storage_dir, "sessions.jsonl")
        self.profile_path = os.path.join(self.storage_dir, "learned_profile.json")

    def record_session(self, timeline: List[Dict[str, Any]], style: Dict[str, float]) -> str:
        session_id = str(uuid.uuid4())
        record = {
            "session_id": session_id,
            "created_at": time.time(),
            "style": dict(style),
            "timeline": timeline,
            "feedback_score": None,
            "feedback_notes": "",
        }
        with open(self.sessions_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return session_id

    def submit_feedback(self, session_id: str, quality_score: float, notes: str = "") -> None:
        sessions = self._load_sessions()
        found = False
        for session in sessions:
            if session.get("session_id") == session_id:
                session["feedback_score"] = max(0.0, min(1.0, float(quality_score)))
                session["feedback_notes"] = notes
                found = True
                break
        if not found:
            raise ValueError(f"Unknown session_id: {session_id}")
        self._write_sessions(sessions)

    def learn(self, base_profile: Dict[str, float], min_sessions: int = 5) -> Optional[Dict[str, float]]:
        reviewed = [
            session for session in self._load_sessions()
            if session.get("feedback_score") is not None
        ]
        if len(reviewed) < min_sessions:
            return None

        high = [s for s in reviewed if float(s["feedback_score"]) >= 0.7]
        low = [s for s in reviewed if float(s["feedback_score"]) <= 0.4]
        profile = dict(base_profile)
        profile.setdefault("__version__", 1)
        profile["__version__"] = int(profile.get("__version__", 1)) + 1
        profile["__learned_offline__"] = True

        high_effects = self._effect_counts(high)
        low_effects = self._effect_counts(low)
        biases: Dict[str, float] = {}
        for effect in sorted(set(high_effects) | set(low_effects)):
            high_rate = high_effects.get(effect, 0) / max(1, sum(high_effects.values()))
            low_rate = low_effects.get(effect, 0) / max(1, sum(low_effects.values()))
            bias = max(-0.1, min(0.1, (high_rate - low_rate) * 0.5))
            if abs(bias) >= 0.01:
                biases[effect] = round(bias, 4)
        profile["effect_quality_biases"] = biases

        with open(self.profile_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True)
        return profile

    def get_quality_biases(self) -> Dict[str, float]:
        if not os.path.exists(self.profile_path):
            return {}
        try:
            with open(self.profile_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return dict(data.get("effect_quality_biases") or {})
        except Exception:
            return {}

    def _load_sessions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.sessions_path):
            return []
        sessions: List[Dict[str, Any]] = []
        with open(self.sessions_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sessions.append(json.loads(line))
        return sessions

    def _write_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        with open(self.sessions_path, "w", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(session, sort_keys=True) + "\n")

    @staticmethod
    def _effect_counts(sessions: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for session in sessions:
            for shot in session.get("timeline", []):
                for effect in shot.get("effects", []):
                    name = str(effect.get("name") or "")
                    if name:
                        counts[name] = counts.get(name, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Director
# ---------------------------------------------------------------------------

class Director:
    """Plans the edit. Heuristic v1 — swap `plan` internals with a
    Transformer later; the JSON contract stays identical."""

    def __init__(self, seed: int = 42,
                 personality_weights: Optional[Dict[str, float]] = None,
                 experience_dir: Optional[str] = None):
        self.seed = seed
        self.personality_weights = personality_weights or {}
        self.experience = EditingExperience(experience_dir) if experience_dir else None
        self.intent_model = IntentInference()
        self.grammar = CinematicGrammarEngine()
        self.story_arc_planner = StoryArcPlanner()
        self.narrative_entity_tracker = NarrativeEntityTracker()
        self.motif_memory = MotifMemory()
        self.narrative_director = NarrativeDirector()

    def plan(self, shots: List[ShotFeatures],
             bpm: float = 130.0) -> List[Dict[str, Any]]:
        n = len(shots)
        timeline: List[Dict[str, Any]] = []
        narrative_plan = self.story_arc_planner.plan(shots)
        entity_timeline = self.narrative_entity_tracker.track(shots)
        motifs = self.motif_memory.observe(shots)
        directing_decisions = self.narrative_director.decide(
            shots, narrative_plan, entity_timeline, motifs
        )
        memory = NarrativeMemory()
        planner = TimelinePlanner(bpm=bpm, seed=self.seed)
        for i, sh in enumerate(shots):
            mem = memory.read(i, n)
            semantics = self._semantics(sh, i, n, mem)
            directing = directing_decisions.get(sh.start, DirectingDecision())
            intent = self.intent_model.infer(sh, semantics, mem, directing)
            actions = self.grammar.actions_for(intent, sh, semantics, directing)
            segment = planner.build_segment(
                sh,
                intent,
                semantics,
                mem,
                actions,
                narrative_plan,
                entity_timeline.entities_at(sh.start),
                directing,
                motifs,
            )
            timeline.append(segment)
            memory.update(intent, segment["effects"], sh.audio_energy)
        energy_arc = [t["energy"] for t in timeline]
        taste = TasteFilter(seed=self.seed, personality_weights=self.personality_weights)
        timeline = taste.filter_timeline(timeline, energy_arc)
        self._sync_legacy_camera_keys(timeline)
        self._refresh_explainability(timeline)
        if self.experience:
            session_id = self.experience.record_session(timeline, self.personality_weights)
            for entry in timeline:
                entry["experience_session_id"] = session_id
        return planner.finalize(timeline)

    @staticmethod
    def _semantics(sh: ShotFeatures, idx: int, total: int, memory: MemoryState) -> SegmentSemantics:
        del idx, total
        prev_energy = memory.scene_energy if memory.last_intents else sh.audio_energy
        upsurge = max(0.0, sh.audio_energy - prev_energy)
        tension = (sh.motion * 0.34) + (sh.audio_energy * 0.36) + (upsurge * 0.2)
        if sh.brightness < 0.34:
            tension += 0.08
        anticipation = (upsurge * 0.48) + (sh.composition_score * 0.2)
        if not sh.is_drop and sh.audio_energy > 0.44 and memory.arc_position < 0.78:
            anticipation += 0.18
        importance = max(
            sh.visual_importance,
            (0.34 if sh.has_faces or sh.face_count > 0 else 0.0)
            + (0.28 if sh.has_speech else 0.0)
            + (0.22 if sh.is_drop else 0.0)
            + (0.16 * sh.subject_confidence),
        )
        focus = max(sh.composition_score, 0.44 + (0.24 if sh.has_faces else 0.0))
        if sh.audio_energy >= 0.68 and sh.motion >= 0.52:
            mood = "high_energy"
        elif sh.has_speech:
            mood = "dialogue"
        elif tension >= 0.62:
            mood = "tense"
        elif sh.audio_energy < 0.32:
            mood = "reflective"
        else:
            mood = "neutral"
        return SegmentSemantics(
            tension=round(max(0.0, min(1.0, tension)), 4),
            anticipation=round(max(0.0, min(1.0, anticipation)), 4),
            narrative_importance=round(max(0.0, min(1.0, importance)), 4),
            viewer_focus=round(max(0.0, min(1.0, focus)), 4),
            fatigue=memory.fatigue,
            mood=mood,
        )

    @staticmethod
    def _sync_legacy_camera_keys(timeline: List[Dict[str, Any]]) -> None:
        for shot in timeline:
            camera = shot.get("camera")
            if not isinstance(camera, dict):
                camera = {"motion": shot.get("camera_move", "static"), "angle": shot.get("angle", "eye")}
                shot["camera"] = camera
            shot["camera_move"] = str(camera.get("motion") or shot.get("camera_move") or "static")
            shot["angle"] = str(camera.get("angle") or shot.get("angle") or "eye")

    @staticmethod
    def _refresh_explainability(timeline: List[Dict[str, Any]]) -> None:
        for shot in timeline:
            narrative = shot.get("narrative") or {}
            act = narrative.get("act") or {}
            decision = narrative.get("directing_decision") or {}
            reasons = []
            if shot.get("is_drop") or shot.get("intent") == "impact":
                reasons.append("Big beat")
            if act.get("act_type") == "climax":
                reasons.append("Climax")
            if decision.get("save_best_moment") or decision.get("delay_climax"):
                reasons.append("Saved for this moment" if act.get("act_type") == "climax" else "Climax delayed")
            if decision.get("hold_shot"):
                reasons.append("Held shot")
            if decision.get("callback_motif"):
                reasons.append(f"Motif callback: {decision.get('callback_motif')}")
            if not reasons:
                reasons.append("Supports narrative intent")
            shot["explainability_log"] = [
                {
                    "effect": effect.get("name", "effect"),
                    "strength": effect.get("strength", 0.0),
                    "reason": list(reasons),
                }
                for effect in shot.get("effects", [])
            ]

    def to_json(self, timeline: List[Dict[str, Any]]) -> str:
        return json.dumps(timeline, indent=2)


class DirectorBrain(Director):
    """Architecture-friendly wrapper with a `run()` entry point."""

    def run(self, shots: List[ShotFeatures], bpm: float = 130.0) -> List[Dict[str, Any]]:
        return self.plan(shots, bpm=bpm)


def plan_edit(video_path: str, beat_analysis: Any = None,
              bpm: float = 130.0, seed: int = 42,
              personality_weights: Optional[Dict[str, float]] = None
              ) -> List[Dict[str, Any]]:
    """One-call convenience: video -> filtered, human-feeling timeline."""
    shots = extract_shot_features(video_path)
    if beat_analysis is not None:
        shots = attach_audio_features(shots, beat_analysis)
        if isinstance(beat_analysis, dict):
            bpm = float(beat_analysis.get("bpm") or beat_analysis.get("tempo") or bpm)
        else:
            bpm = float(getattr(beat_analysis, "bpm", None) or getattr(beat_analysis, "tempo", None) or bpm)
    return Director(seed=seed,
                    personality_weights=personality_weights).plan(shots, bpm)
