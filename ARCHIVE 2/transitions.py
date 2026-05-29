from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionPolicy:
    every: int
    phase: int
    overlap_seconds: float


def transition_name_for_mood(mood: str, intensity: int) -> str:
    if intensity >= 3 and mood in {"hype", "dark"}:
        return "impact"
    if mood in {"sad", "chill"}:
        return "soft_dissolve"
    return "cinematic_blend"
