from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RenderOperation:
    """
    Smallest visual unit the AI should think in.

    Legacy effect names should expand into these operations at the preset layer.
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)
    sync: str = "static"
    backend_hint: str = "auto"

    def stable_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "params": stable_value(self.params),
            "sync": self.sync,
            "backend_hint": self.backend_hint,
        }


def operation_stage(op: str) -> str:
    if op in {"source_clip"}:
        return "decode"
    if op in {"speed", "temporal_blend"}:
        return "temporal"
    if op in {"overlay", "mask", "composite"}:
        return "composite"
    if op in {"transform", "shake"}:
        return "transform"
    if op in {"color_grade", "contrast_punch", "drop_punch", "saturation_pulse", "warm_halation", "desaturate"}:
        return "color"
    return "effect"


def stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [stable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return round(value, 6)
    return value
