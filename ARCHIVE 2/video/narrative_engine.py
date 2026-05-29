from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NarrativeBeat:
    segment_index: int
    phase: str
    energy: float


def phase_for_energy(energy: float) -> str:
    if energy >= 0.78:
        return "impact"
    if energy >= 0.55:
        return "rise"
    if energy <= 0.25:
        return "breath"
    return "flow"
