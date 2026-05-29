from __future__ import annotations

from typing import Any

import numpy as np

from audio_features import energy_at


def score_cut_boundary(
    prev_row: dict[str, Any],
    next_row: dict[str, Any],
    *,
    cut_time: float,
    audio_features: dict[str, Any] | None,
    drop_times: list[float] | None,
    mood: str,
    style: dict[str, Any],
) -> float:
    motion = max(float(prev_row.get("motion", 0.0)), float(next_row.get("motion", 0.0)))
    flow = max(float(prev_row.get("flow_mag", 0.0)), float(next_row.get("flow_mag", 0.0)))
    emotion = max(float(prev_row.get("emotion_weight", 0.0)), float(next_row.get("emotion_weight", 0.0)))
    scene_change = max(float(prev_row.get("scene_change", 0.0)), float(next_row.get("scene_change", 0.0)))
    visual = max(float(prev_row.get("visual_importance", 0.5)), float(next_row.get("visual_importance", 0.5)))
    music = energy_at(audio_features, cut_time)
    drop_bonus = 0.0
    for drop_t in drop_times or []:
        dist = abs(float(cut_time) - float(drop_t))
        if dist <= 1.5:
            drop_bonus = max(drop_bonus, 1.0 - (dist / 1.5))
    personality_density = float(style.get("cut_density", 1.0))
    emotion_weight = 0.32 if mood in {"sad", "dramatic", "chill"} else 0.18
    score = (
        (motion * 0.18)
        + (flow * 0.12)
        + (emotion * emotion_weight)
        + (music * 0.2)
        + (scene_change * 0.16)
        + (visual * 0.14)
        + (drop_bonus * 0.16)
    )
    return float(np.clip(score * personality_density, 0.0, 1.0))


def annotate_cut_scores(
    rows: list[dict[str, Any]],
    cut_points: list[float],
    *,
    audio_features: dict[str, Any] | None,
    drop_times: list[float] | None,
    mood: str,
    style: dict[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    for idx, row in enumerate(rows):
        boundary_time = float(cut_points[idx + 1]) if idx + 1 < len(cut_points) else float(cut_points[-1] if cut_points else 0.0)
        next_row = rows[min(idx + 1, len(rows) - 1)]
        score = score_cut_boundary(
            row,
            next_row,
            cut_time=boundary_time,
            audio_features=audio_features,
            drop_times=drop_times,
            mood=mood,
            style=style,
        )
        row["cut_priority"] = round(score, 4)
        row["hype_score"] = round(
            float(
                np.clip(
                    (score * 0.45)
                    + (float(row.get("scene_energy", 0.0)) * 0.24)
                    + (energy_at(audio_features, boundary_time) * 0.2)
                    + (float(row.get("flow_mag", 0.0)) * 0.11),
                    0.0,
                    1.0,
                )
            ),
            4,
        )
    return rows
