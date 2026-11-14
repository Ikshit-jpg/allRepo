"""Shared, renderer-neutral representation of a finished edit decision."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EditPlanShot:
    source_start: float
    source_end: float
    output_start: float
    output_duration: float
    playback_speed: float = 1.0
    beat_times: list[float] = field(default_factory=list)
    is_downbeat: bool = False
    is_drop: bool = False
    music_section: str = "unknown"
    music_relation: str = "off_beat"
    creative_intent: str = "transition"
    effects: list[dict[str, Any]] = field(default_factory=list)
    render_operations: list[dict[str, Any]] = field(default_factory=list)
    transition: str = "cut"
    animation: dict[str, Any] = field(default_factory=dict)
    color_grade: dict[str, Any] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    visual_features: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EditPlanShot":
        start = float(value.get("source_start", value.get("start", 0.0)))
        end = float(value.get("source_end", value.get("end", 0.0)))
        speed = float(value.get("playback_speed", value.get("speed", 1.0)) or 1.0)
        duration = float(value.get("output_duration", value.get("duration", (end - start) / speed)) or 0.0)
        return cls(
            source_start=start,
            source_end=end,
            output_start=float(value.get("output_start", 0.0) or 0.0),
            output_duration=duration,
            playback_speed=speed,
            beat_times=[float(t) for t in value.get("beat_times", value.get("beats", []))],
            is_downbeat=bool(value.get("is_downbeat", False)),
            is_drop=bool(value.get("is_drop", False)),
            music_section=str(value.get("music_section", value.get("section", "unknown"))),
            music_relation=str(value.get("music_relation", "off_beat")),
            creative_intent=str(value.get("creative_intent", value.get("intent", "transition"))),
            effects=[dict(effect) for effect in value.get("effects", []) if isinstance(effect, dict)],
            render_operations=[dict(operation) for operation in value.get("render_operations", []) if isinstance(operation, dict)],
            transition=str(value.get("transition", "cut")),
            animation=dict(value.get("animation", {})) if isinstance(value.get("animation"), dict) else {},
            color_grade=dict(value.get("color_grade", {})) if isinstance(value.get("color_grade"), dict) else {},
            explanation=[str(item) for item in value.get("explanation", [])],
            visual_features=dict(value.get("visual_features", {})) if isinstance(value.get("visual_features"), dict) else {},
            source_id=str(value.get("source_id", "")),
        )

    def validate(self) -> None:
        numbers = (self.source_start, self.source_end, self.output_start, self.output_duration, self.playback_speed)
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError("EditPlan shot contains a non-finite timestamp or speed")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("EditPlan shot must have a positive source interval")
        if self.output_start < 0 or self.output_duration <= 0:
            raise ValueError("EditPlan shot must have a positive output interval")
        if not 0.05 <= self.playback_speed <= 8.0:
            raise ValueError("EditPlan playback speed is outside the supported range")
        if any(not math.isfinite(beat) for beat in self.beat_times):
            raise ValueError("EditPlan beat times must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditPlan:
    source_video: str
    shots: list[EditPlanShot]
    music_timeline: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def validate(self) -> "EditPlan":
        if not self.shots:
            raise ValueError("EditPlan must contain at least one shot")
        last_output_end = 0.0
        for shot in self.shots:
            shot.validate()
            if shot.output_start + 1e-3 < last_output_end:
                raise ValueError("EditPlan output shots must be ordered and non-overlapping")
            last_output_end = shot.output_start + shot.output_duration
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "music_timeline": self.music_timeline,
            "quality_report": self.quality_report,
            "shots": [shot.to_dict() for shot in self.shots],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EditPlan":
        shots = value.get("shots") or []
        if not isinstance(shots, list):
            raise ValueError("EditPlan shots must be a list")
        return cls(
            source_video=str(value.get("source_video", "")),
            music_timeline=dict(value.get("music_timeline", {})),
            quality_report=dict(value.get("quality_report", {})),
            shots=[EditPlanShot.from_dict(shot) for shot in shots if isinstance(shot, dict)],
            version=int(value.get("version", 1)),
        ).validate()
