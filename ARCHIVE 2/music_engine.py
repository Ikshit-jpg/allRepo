from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MusicProfile:
    mood: str
    bpm: int
    energy: float


def clamp_bpm(bpm: int, fallback: int = 130) -> int:
    try:
        return max(60, min(220, int(bpm)))
    except Exception:
        return fallback
