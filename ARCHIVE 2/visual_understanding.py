from __future__ import annotations

import math
from typing import Any

import numpy as np


def classify_motion(motion: float, flow_mag: float) -> str:
    value = (float(motion) * 0.58) + (float(flow_mag) * 0.42)
    if value >= 0.42:
        return "explosive"
    if value >= 0.18:
        return "medium"
    return "low"


def classify_shot_type(face_area: float, face_count: int, subject_conf: float) -> str:
    area = float(max(face_area, 0.0))
    if face_count <= 0 and subject_conf < 0.35:
        return "wide"
    if area >= 0.22:
        return "extreme_close_up"
    if area >= 0.09:
        return "close_up"
    if area >= 0.025 or subject_conf >= 0.5:
        return "medium"
    return "wide"


def infer_emotion_weight(row: dict[str, Any]) -> float:
    face = 1.0 if int(row.get("face_count", 0)) > 0 else 0.0
    close = 1.0 if str(row.get("shot_type", "")) in {"close_up", "extreme_close_up"} else 0.0
    low_motion = 1.0 - float(np.clip(row.get("motion", 0.0), 0.0, 1.0))
    contrast = float(np.clip(row.get("contrast", 0.5), 0.0, 1.0))
    brightness = float(np.clip(row.get("brightness", 0.5), 0.0, 1.0))
    moody_light = 1.0 - abs(brightness - 0.42) * 1.8
    return float(np.clip((face * 0.34) + (close * 0.24) + (low_motion * 0.2) + (contrast * 0.12) + (moody_light * 0.1), 0.0, 1.0))


def composition_score(row: dict[str, Any]) -> float:
    x = float(np.clip(row.get("face_center_x", 0.5), 0.0, 1.0))
    y = float(np.clip(row.get("face_center_y", 0.42), 0.0, 1.0))
    thirds_x = min(abs(x - 1.0 / 3.0), abs(x - 2.0 / 3.0))
    centered = 1.0 - min(1.0, math.hypot(x - 0.5, y - 0.45) / 0.55)
    thirds = 1.0 - min(1.0, thirds_x / 0.34)
    return float(np.clip((centered * 0.62) + (thirds * 0.38), 0.0, 1.0))


def enrich_segment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    prev_x: float | None = None
    prev_y: float | None = None
    for idx, row in enumerate(rows):
        motion = float(row.get("motion", 0.0))
        flow_mag = float(row.get("flow_mag", 0.0))
        face_area = float(row.get("face_area", 0.0))
        face_count = int(row.get("face_count", 0))
        subject_conf = float(row.get("subject_conf", 0.0))
        x = float(np.clip(row.get("face_center_x", 0.5), 0.0, 1.0))
        y = float(np.clip(row.get("face_center_y", 0.42), 0.0, 1.0))

        dx = 0.0 if prev_x is None else x - prev_x
        dy = 0.0 if prev_y is None else y - prev_y
        if abs(dx) < 0.035 and abs(dy) < 0.04:
            direction = "still"
        elif abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"

        row["motion_class"] = classify_motion(motion, flow_mag)
        row["shot_type"] = classify_shot_type(face_area, face_count, subject_conf)
        row["motion_direction"] = direction
        row["motion_direction_conf"] = round(float(np.clip(math.hypot(dx, dy) * 3.2, 0.0, 1.0)), 4)
        row["composition_score"] = round(composition_score(row), 4)
        row["emotion_weight"] = round(infer_emotion_weight(row), 4)
        row["visual_importance"] = round(
            float(
                np.clip(
                    (float(row.get("edit_priority", 0.5)) * 0.34)
                    + (float(row.get("emotion_weight", 0.0)) * 0.24)
                    + (float(row.get("composition_score", 0.5)) * 0.16)
                    + (float(row.get("scene_change", 0.0)) * 0.16)
                    + (flow_mag * 0.1),
                    0.0,
                    1.0,
                )
            ),
            4,
        )
        if face_count > 0 or subject_conf > 0.35:
            prev_x, prev_y = x, y
        elif idx == 0:
            prev_x, prev_y = x, y
    return rows


def transition_hint_from_row(row: dict[str, Any]) -> str:
    direction = str(row.get("motion_direction", "still"))
    motion_class = str(row.get("motion_class", "low"))
    brightness = float(row.get("brightness", 0.5))
    scene_change = float(row.get("scene_change", 0.0))
    if scene_change > 0.34 or motion_class == "explosive":
        return "flash" if brightness > 0.52 else "glitch"
    if direction in {"left", "right"} and float(row.get("motion_direction_conf", 0.0)) > 0.18:
        return f"wipe_{direction}"
    if str(row.get("shot_type", "")) in {"close_up", "extreme_close_up"}:
        return "soft_push"
    return "cut"
