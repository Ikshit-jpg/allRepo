"""
taste_system.py
===============
THE TASTE LAYER. This is what makes edits feel HUMAN.

Problem it solves: the AI was applying the same effect (or group of
effects) across the whole video. Real editors never do that. They:
  - rotate effects (never the same combo 3 shots in a row)
  - ramp intensity with the music arc (never fixed strength values)
  - go quiet during dialogue / calm shots
  - reserve flash/shake for actual drops so impacts feel EARNED
  - vary transitions (never the same transition 3x in a row)

This module is a deterministic-but-seeded CONSTRAINT FILTER applied
AFTER the planner output. It needs no trained model, so it works today
and can later be replaced/augmented by a learned policy (RL hooks at
the bottom).

Used by: brain/director.py (and can wrap archive_decision_brain output)
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Effect / transition vocabulary (matches archive_decision_brain.py + engine)
# ---------------------------------------------------------------------------

EFFECT_POOLS: Dict[str, List[str]] = {
    "impact":    ["impact_flash_combo", "zoom_burst_shake", "god_of_war",
                  "energy_aura", "camera_slam", "zoom_pulse"],
    "build-up":  ["handheld_shake", "push_in_slow", "zoom_pulse",
                  "energy_aura", "orbit_right"],
    "calm":      ["cinematic_grade", "atmospheric_grade", "ken_burns_slow",
                  "subtle_drift", "parallax_zoom"],
    "dialogue":  ["cinematic_grade", "portrait_enhance"],
    "transition": ["smear_cut", "whip_pan_right", "parallax_zoom", "zoom_pulse"],
}

TRANSITION_POOLS: Dict[str, List[str]] = {
    "high": ["zoom_punch", "flash_cut", "glitch_cut", "whip_pan_right",
             "whip_pan_up", "smear_cut", "cut"],
    "mid":  ["cut", "whip_pan_right", "smear_cut", "zoom_punch", "dissolve"],
    "low":  ["cut", "dissolve", "fade_black", "luma_wipe"],
}

# Effects that scream for attention. Must be rationed.
HEAVY_EFFECTS = {"impact_flash_combo", "zoom_burst_shake", "god_of_war",
                 "camera_slam", "glitch_cut", "flash_cut"}

# Effects allowed (at low strength) during dialogue / quiet shots.
SUBTLE_EFFECTS = {"cinematic_grade", "atmospheric_grade", "portrait_enhance",
                  "subtle_drift", "ken_burns_slow"}


def _energy_tier(energy: float) -> str:
    if energy >= 0.62:
        return "high"
    if energy >= 0.32:
        return "mid"
    return "low"


class TasteFilter:
    """Post-planner constraint filter. Mutates a timeline so that effect
    usage, intensity and transitions feel like a human editor's choices.

    Timeline format (list of dicts, JSON-friendly):
        {
          "start": float, "end": float,
          "intent": "impact" | "build-up" | "calm" | "dialogue" | "transition",
          "energy": float (0..1, optional),
          "is_drop": bool (optional),
          "has_speech": bool (optional),
          "transition": str,
          "effects": [{"name": str, "strength": float}, ...]
        }
    """

    def __init__(self,
                 seed: int = 42,
                 cooldown_shots: int = 4,
                 combo_memory: int = 6,
                 personality_weights: Optional[Dict[str, float]] = None):
        self.rng = random.Random(seed)
        self.cooldown_shots = max(1, cooldown_shots)
        self.combo_memory = max(2, combo_memory)
        self.personality_weights = personality_weights or {}
        # effect name -> shots of cooldown remaining (decays each shot)
        self._cooldown: Dict[str, float] = {}
        # recently used effect SETS (frozensets) -> n-gram combo penalty
        self._combo_history: List[frozenset] = []
        self._transition_history: List[str] = []
        self._last_heavy_idx: int = -10

    # -- public ------------------------------------------------------------

    def filter_timeline(self, timeline: List[Dict[str, Any]],
                        energy_arc: Optional[Sequence[float]] = None
                        ) -> List[Dict[str, Any]]:
        """Rewrites effects/strengths/transitions for every shot."""
        n = len(timeline)
        if n == 0:
            return timeline
        arc = list(energy_arc) if energy_arc else self._default_arc(n)
        out: List[Dict[str, Any]] = []
        for i, shot in enumerate(timeline):
            shot = dict(shot)
            energy = float(shot.get("energy", arc[min(i, len(arc) - 1)]))
            intent = shot.get("intent", "calm")
            shot["energy"] = energy
            shot["effects"] = self._choose_effects(shot, i, energy, intent, n)
            shot["transition"] = self._choose_transition(shot, energy)
            out.append(shot)
            self._tick_cooldowns()
        return out

    # -- effect selection ----------------------------------------------------

    def _choose_effects(self, shot: Dict[str, Any], idx: int,
                        energy: float, intent: str, total: int
                        ) -> List[Dict[str, Any]]:
        has_speech = bool(shot.get("has_speech", intent == "dialogue"))
        is_drop = bool(shot.get("is_drop", False))

        # 1. How many effects does this shot deserve?
        if has_speech or intent == "dialogue":
            max_fx = 1
            pool = list(SUBTLE_EFFECTS & set(EFFECT_POOLS["dialogue"])) or \
                list(EFFECT_POOLS["dialogue"])
        elif intent == "calm":
            max_fx = 1 if energy < 0.45 else 2
            pool = list(EFFECT_POOLS["calm"])
        elif intent == "build-up":
            max_fx = 2
            pool = list(EFFECT_POOLS["build-up"])
        elif intent == "transition":
            max_fx = 1
            pool = list(EFFECT_POOLS["transition"])
        else:  # impact
            max_fx = 3 if (is_drop and energy > 0.7) else 2
            pool = list(EFFECT_POOLS["impact"])

        # 2. Heavy-effect rationing: impacts must feel EARNED.
        heavy_allowed = (
            not has_speech
            and energy > 0.55
            and (idx - self._last_heavy_idx) >= 2
            and (is_drop or intent == "impact")
        )
        if not heavy_allowed:
            pool = [e for e in pool if e not in HEAVY_EFFECTS] or \
                list(SUBTLE_EFFECTS)

        # 3. Weighted diversity sampling with cooldown + combo penalties.
        chosen: List[str] = []
        candidates = list(dict.fromkeys(pool))
        count = self.rng.randint(max(1, max_fx - 1), max_fx) if max_fx > 1 else 1
        for _ in range(count):
            avail = [c for c in candidates if c not in chosen]
            if not avail:
                break
            pick = self._weighted_pick(avail)
            if pick is None:
                break
            chosen.append(pick)

        # 4. Hard guarantee: never the same effect SET 3 shots in a row.
        new_set = frozenset(chosen)
        if (len(self._combo_history) >= 2
                and self._combo_history[-1] == new_set
                and self._combo_history[-2] == new_set):
            chosen = self._force_variation(chosen, intent)
            new_set = frozenset(chosen)

        # 5. Register usage.
        for name in chosen:
            self._cooldown[name] = float(self.cooldown_shots)
            if name in HEAVY_EFFECTS:
                self._last_heavy_idx = idx
        self._combo_history.append(new_set)
        if len(self._combo_history) > self.combo_memory:
            self._combo_history.pop(0)

        # 6. Intensity ramp: strength follows the energy arc, never fixed.
        arc_pos = idx / max(1, total - 1)
        return [{"name": name, "strength": self._strength(name, energy, arc_pos)}
                for name in chosen]

    def _weighted_pick(self, candidates: List[str]) -> Optional[str]:
        weights = []
        for c in candidates:
            w = self.personality_weights.get(c, 1.0)
            cd = self._cooldown.get(c, 0.0)
            # decaying penalty: just-used effects are ~9x less likely
            w *= 1.0 / (1.0 + 2.0 * cd)
            # combo n-gram penalty: punish effects in recently used sets
            for age, combo in enumerate(reversed(self._combo_history[-3:])):
                if c in combo:
                    w *= 0.45 + 0.15 * age
            weights.append(max(1e-6, w))
        total = sum(weights)
        r = self.rng.random() * total
        for c, w in zip(candidates, weights):
            r -= w
            if r <= 0:
                return c
        return candidates[-1] if candidates else None

    def _force_variation(self, chosen: List[str], intent: str) -> List[str]:
        """Swap one effect for the least-recently-used alternative."""
        pool = EFFECT_POOLS.get(intent, EFFECT_POOLS["calm"])
        used = set(chosen)
        alts = sorted((e for e in pool if e not in used),
                      key=lambda e: self._cooldown.get(e, 0.0))
        if alts and chosen:
            chosen = list(chosen)
            chosen[self.rng.randrange(len(chosen))] = alts[0]
        elif alts:
            chosen = [alts[0]]
        else:
            chosen = chosen[:-1] if len(chosen) > 1 else ["cinematic_grade"]
        return chosen

    def _strength(self, name: str, energy: float, arc_pos: float) -> float:
        """Adaptive strength: base follows energy, ramps across the edit,
        with human jitter so two shots never get identical values."""
        base = 0.30 + 0.55 * energy
        # global ramp: edits intensify toward the climax (~70% in), then ease
        ramp = 0.85 + 0.30 * math.sin(min(arc_pos, 1.0) * math.pi * 0.82)
        jitter = 1.0 + self.rng.uniform(-0.12, 0.12)
        s = base * ramp * jitter
        if name in SUBTLE_EFFECTS:
            s = min(s, 0.55)
        return round(max(0.10, min(1.0, s)), 3)

    # -- transitions ---------------------------------------------------------

    def _choose_transition(self, shot: Dict[str, Any], energy: float) -> str:
        tier = _energy_tier(energy)
        pool = list(TRANSITION_POOLS[tier])
        hist = self._transition_history
        # forbid the same transition 3x in a row
        if len(hist) >= 2 and hist[-1] == hist[-2]:
            pool = [t for t in pool if t != hist[-1]] or ["cut"]
        # soft penalty for the most recent transition
        weights = [0.35 if (hist and t == hist[-1]) else 1.0 for t in pool]
        total = sum(weights)
        r = self.rng.random() * total
        pick = pool[-1]
        for t, w in zip(pool, weights):
            r -= w
            if r <= 0:
                pick = t
                break
        hist.append(pick)
        if len(hist) > 8:
            hist.pop(0)
        return pick

    # -- internals -----------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        for k in list(self._cooldown):
            self._cooldown[k] -= 1.0
            if self._cooldown[k] <= 0:
                del self._cooldown[k]

    @staticmethod
    def _default_arc(n: int) -> List[float]:
        """Intro -> build -> climax (~70%) -> release energy arc."""
        arc = []
        for i in range(n):
            p = i / max(1, n - 1)
            arc.append(round(0.25 + 0.7 * math.sin(min(p / 0.7, 1.0)
                                                   * math.pi / 2) ** 2
                             * (1.0 if p <= 0.85 else 0.7), 3))
        return arc


def rhythm_durations(beats: List[float], bpm: float, intent_seq: List[str],
                     seed: int = 42) -> List[float]:
    """Shot-duration control tied to BPM with controlled variation.
    Impact shots = 1-2 beats, build-up = 2 beats, calm/dialogue = 4-8 beats.
    Variation keeps the rhythm stable but never robotic."""
    rng = random.Random(seed)
    beat = 60.0 / max(1.0, bpm)
    durations = []
    for intent in intent_seq:
        if intent == "impact":
            beats_held = rng.choice([1, 1, 2])
        elif intent == "build-up":
            beats_held = rng.choice([2, 2, 4])
        elif intent == "dialogue":
            beats_held = rng.choice([4, 6, 8])
        else:  # calm / transition
            beats_held = rng.choice([4, 4, 8])
        durations.append(round(beats_held * beat, 3))
    return durations


# ---------------------------------------------------------------------------
# Learning hooks (future RL / preference training — interfaces only)
# ---------------------------------------------------------------------------

def log_preference(log_path: str, edit_a_id: str, edit_b_id: str,
                   winner: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Append a human preference comparison ("which feels more cinematic?")
    to a JSONL file. Feed this back into training later (reward model / RL).
    Wire this to rating.html submissions."""
    import json
    import time
    rec = {"ts": time.time(), "a": edit_a_id, "b": edit_b_id,
           "winner": winner, "meta": meta or {}}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
