from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol

from engine.graph.operations import RenderOperation
from engine.graph.render_graph import RenderGraph


@dataclass(frozen=True)
class PreviewSettings:
    width: int = 640
    height: int = 360
    fps: float = 24.0
    crf: int = 28
    preset: str = "ultrafast"


@dataclass(frozen=True)
class ProxySettings:
    width: int = 640
    height: int = 360
    fps: float = 24.0
    crf: int = 30
    preset: str = "ultrafast"


@dataclass
class ClipRenderJob:
    input_path: str
    output_path: str
    trim_start: float = 0.0
    trim_dur: float = 5.0
    mood: str = "hype"
    effect: str = "cinematic_grade"
    intensity: float = 0.75
    section: str = "chorus"
    beat_times: list[float] = field(default_factory=list)
    is_drop: bool = False
    is_climax: bool = False
    is_breath: bool = False
    zoom_start: float = 1.0
    zoom_end: float = 1.05
    zoom_curve: str = "ease"
    speed_mult: float = 1.0
    animation: str = "subtle_drift"
    anim_intensity: float = 0.4
    hold_frames: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    crf: int = 16
    preset: str = "veryfast"
    particle_type: str = ""
    light_leak: float = 0.0
    seed: int = 42
    prefer_gpu_effects: bool = True
    include_audio: bool = True
    operations: list[RenderOperation] = field(default_factory=list)


@dataclass
class TransitionJob:
    clip_a: str
    clip_b: str
    output_path: str
    trans_type: str = "zoom_punch"
    duration_sec: float = 0.4
    offset_a: float = 0.0
    offset_b: float = 0.0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    crf: int = 16
    preset: str = "veryfast"


@dataclass(frozen=True)
class TimelineClip:
    source: str
    start: float
    duration: float
    render_job: ClipRenderJob


@dataclass(frozen=True)
class TimelineTransition:
    index: int
    kind: str
    duration: float


@dataclass
class TimelineGraph:
    clips: list[TimelineClip] = field(default_factory=list)
    transitions: list[TimelineTransition] = field(default_factory=list)
    music_path: Optional[str] = None
    output_path: str = "output.mp4"

    def total_duration(self) -> float:
        base = sum(max(0.0, clip.duration) for clip in self.clips)
        overlap = sum(max(0.0, transition.duration) for transition in self.transitions)
        return max(0.0, base - overlap)


@dataclass
class EffectPlan:
    ffmpeg_filters: list[str] = field(default_factory=list)
    gpu_effects: list[dict[str, Any]] = field(default_factory=list)
    operations: list[RenderOperation] = field(default_factory=list)
    render_graph: Optional[RenderGraph] = None
    cache_key: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def vf(self) -> str:
        return ",".join(part for part in self.ffmpeg_filters if part)


class RenderBackend(Protocol):
    name: str
    available: bool

    def can_render(self, graph: RenderGraph, target: str) -> bool:
        ...

    def render_graph(self, graph: RenderGraph, target: str, output_path: str) -> tuple[bool, str]:
        ...


def parse_fps(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / max(1.0, float(den))
        return float(value)
    except Exception:
        return 30.0


def atempo_chain(speed: float) -> list[str]:
    speed = max(0.05, float(speed))
    parts: list[float] = []
    while speed > 2.0:
        parts.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        parts.append(0.5)
        speed /= 0.5
    parts.append(speed)
    return [f"atempo={part:.3f}" for part in parts]


def copy_job(job: ClipRenderJob, **updates: Any) -> ClipRenderJob:
    return replace(job, **updates)
