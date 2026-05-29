"""
Legacy compatibility facade for the modular render engine.

The old temp engine used to keep graph, compiler, backend, cache, scheduler,
and preset logic in this single file. The real implementation now lives in the
engine package so each layer can evolve independently.
"""

from __future__ import annotations

from archive_engine import (
    FFMPEG,
    FFPROBE,
    ClipRenderJob,
    EffectPlan,
    FFmpegEngine,
    PreviewSettings,
    ProxySettings,
    RenderGraph,
    RenderNode,
    RenderOperation,
    TimelineClip,
    TimelineGraph,
    TimelineTransition,
    TransitionJob,
)
from engine.backends.ffmpeg_backend import XFadeBuilder
from engine.backends.gpu_backend import GPUCompositorBackend
from engine.backends.preview_backend import CPUFallbackBackend
from engine.cache.render_cache import CacheEntry, RenderCache
from engine.compiler.ffmpeg_compiler import HybridFilterCompiler
from engine.compiler.graph_compiler import RenderGraphCompiler
from engine.graph.operations import operation_stage as _operation_stage, stable_value as _stable_value
from engine.graph.render_graph import hash_payload as _hash_payload
from engine.pipeline.render_pipeline import RenderBackend, atempo_chain as _atempo_chain, copy_job as _copy_job, parse_fps as _parse_fps
from engine.presets.operation_presets import OperationPresetMapper
from engine.scheduler.background_scheduler import BackgroundRenderScheduler


__all__ = [
    "FFMPEG",
    "FFPROBE",
    "BackgroundRenderScheduler",
    "CacheEntry",
    "ClipRenderJob",
    "CPUFallbackBackend",
    "EffectPlan",
    "FFmpegEngine",
    "GPUCompositorBackend",
    "HybridFilterCompiler",
    "OperationPresetMapper",
    "PreviewSettings",
    "ProxySettings",
    "RenderBackend",
    "RenderCache",
    "RenderGraph",
    "RenderGraphCompiler",
    "RenderNode",
    "RenderOperation",
    "TimelineClip",
    "TimelineGraph",
    "TimelineTransition",
    "TransitionJob",
    "XFadeBuilder",
    "_atempo_chain",
    "_copy_job",
    "_hash_payload",
    "_operation_stage",
    "_parse_fps",
    "_stable_value",
]
