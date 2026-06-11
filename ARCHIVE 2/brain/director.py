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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


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
        ))
    return shots


def attach_audio_features(shots: List[ShotFeatures], beat_analysis: Any
                          ) -> List[ShotFeatures]:
    """Annotate shots with beat/drop/energy info from a BeatAnalysis
    (see archive_decision_brain.analyze_music)."""
    beats = list(getattr(beat_analysis, "beats", []) or [])
    drops = list(getattr(beat_analysis, "drops", []) or [])
    curve = list(getattr(beat_analysis, "energy_curve", []) or [])
    peak = max(curve) if curve else 1.0
    for sh in shots:
        mid = (sh.start + sh.end) / 2
        sh.on_beat = any(abs(sh.start - b) < 0.1 for b in beats)
        sh.is_drop = any(sh.start - 0.3 <= d <= sh.end for d in drops)
        if curve:
            i0 = max(0, min(len(curve) - 1, int(sh.start * 10)))
            i1 = max(i0 + 1, min(len(curve), int(sh.end * 10)))
            seg = curve[i0:i1]
            sh.audio_energy = round(min(1.0, (sum(seg) / len(seg)) /
                                        max(1e-6, peak)), 3)
    return shots


# ---------------------------------------------------------------------------
# Director
# ---------------------------------------------------------------------------

class Director:
    """Plans the edit. Heuristic v1 — swap `plan` internals with a
    Transformer later; the JSON contract stays identical."""

    def __init__(self, seed: int = 42,
                 personality_weights: Optional[Dict[str, float]] = None):
        self.seed = seed
        self.taste = TasteFilter(seed=seed,
                                 personality_weights=personality_weights)

    def plan(self, shots: List[ShotFeatures],
             bpm: float = 130.0) -> List[Dict[str, Any]]:
        n = len(shots)
        timeline: List[Dict[str, Any]] = []
        for i, sh in enumerate(shots):
            intent = self._intent(sh, i, n)
            timeline.append({
                "start": sh.start,
                "end": sh.end,
                "intent": intent,
                "energy": sh.audio_energy,
                "is_drop": sh.is_drop,
                "has_speech": sh.has_speech,
                "camera_move": self._camera(intent, sh),
                "transition": "cut",   # overwritten by taste filter
                "effects": [],          # filled by taste filter
            })
        energy_arc = [t["energy"] for t in timeline]
        timeline = self.taste.filter_timeline(timeline, energy_arc)
        # rhythm: suggested duration per shot (renderer may trim to this)
        intents = [t["intent"] for t in timeline]
        for t, dur in zip(timeline,
                          rhythm_durations([], bpm, intents, seed=self.seed)):
            t["suggested_duration"] = dur
        return timeline

    @staticmethod
    def _intent(sh: ShotFeatures, idx: int, total: int) -> str:
        p = idx / max(1, total - 1)
        if sh.has_speech:
            return "dialogue"
        if sh.is_drop or (sh.audio_energy > 0.72 and sh.motion > 0.55):
            return "impact"
        if sh.audio_energy > 0.45 and p < 0.75:
            return "build-up"
        if p < 0.08 or p > 0.92:
            return "calm"
        return "calm" if sh.audio_energy < 0.4 else "transition"

    @staticmethod
    def _camera(intent: str, sh: ShotFeatures) -> str:
        if intent == "impact":
            return "zoom"
        if intent == "build-up":
            return "pan" if sh.motion < 0.5 else "static"
        return "static"

    def to_json(self, timeline: List[Dict[str, Any]]) -> str:
        return json.dumps(timeline, indent=2)


def plan_edit(video_path: str, beat_analysis: Any = None,
              bpm: float = 130.0, seed: int = 42,
              personality_weights: Optional[Dict[str, float]] = None
              ) -> List[Dict[str, Any]]:
    """One-call convenience: video -> filtered, human-feeling timeline."""
    shots = extract_shot_features(video_path)
    if beat_analysis is not None:
        shots = attach_audio_features(shots, beat_analysis)
        bpm = float(getattr(beat_analysis, "bpm", bpm) or bpm)
    return Director(seed=seed,
                    personality_weights=personality_weights).plan(shots, bpm)
