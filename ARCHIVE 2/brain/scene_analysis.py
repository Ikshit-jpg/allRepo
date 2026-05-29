from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneAnalysisConfig:
    workers: int
    min_segments_for_parallel: int


def scene_analysis_config() -> SceneAnalysisConfig:
    cpu_count = os.cpu_count() or 4
    workers = int(os.getenv("ARCHIVE_SCENE_ANALYSIS_WORKERS", str(min(4, max(1, cpu_count // 2)))))
    min_segments = int(os.getenv("ARCHIVE_PARALLEL_SCENE_MIN_SEGMENTS", "10"))
    return SceneAnalysisConfig(workers=max(1, workers), min_segments_for_parallel=max(2, min_segments))


def should_parallelize_scene_analysis(segment_count: int) -> bool:
    cfg = scene_analysis_config()
    return cfg.workers > 1 and int(segment_count) >= cfg.min_segments_for_parallel
