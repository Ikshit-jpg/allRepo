from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EditorMemory:
    style_counts: dict[str, int] = field(default_factory=dict)
    mood_counts: dict[str, int] = field(default_factory=dict)
    preferred_transitions: dict[str, int] = field(default_factory=dict)
    feedback: dict[str, dict[str, float]] = field(default_factory=dict)
    last_updated: float = 0.0

    def remember_render(self, mood: str, style: dict[str, Any], transitions: list[str] | None = None) -> None:
        self.last_updated = time.time()
        if mood:
            self.mood_counts[mood] = self.mood_counts.get(mood, 0) + 1
        for key, enabled in style.items():
            if isinstance(enabled, bool) and enabled:
                self.style_counts[key] = self.style_counts.get(key, 0) + 1
        for transition in transitions or []:
            if transition:
                self.preferred_transitions[transition] = self.preferred_transitions.get(transition, 0) + 1

    def remember_feedback(self, editor_style: str, metrics: dict[str, float]) -> None:
        if not editor_style:
            return
        self.last_updated = time.time()
        current = dict(self.feedback.get(editor_style) or {})
        count = current.get("count", 0.0) + 1.0
        current["count"] = count
        for key, value in metrics.items():
            try:
                v = float(value)
            except Exception:
                continue
            prev = float(current.get(key, v))
            current[key] = prev + ((v - prev) / count)
        self.feedback[editor_style] = current

    def to_dict(self) -> dict[str, Any]:
        return {
            "style_counts": self.style_counts,
            "mood_counts": self.mood_counts,
            "preferred_transitions": self.preferred_transitions,
            "feedback": self.feedback,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditorMemory":
        return cls(
            style_counts=dict(payload.get("style_counts") or {}),
            mood_counts=dict(payload.get("mood_counts") or {}),
            preferred_transitions=dict(payload.get("preferred_transitions") or {}),
            feedback={str(k): dict(v) for k, v in dict(payload.get("feedback") or {}).items() if isinstance(v, dict)},
            last_updated=float(payload.get("last_updated") or 0.0),
        )


def load_editor_memory(path: Path) -> EditorMemory:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return EditorMemory.from_dict(payload)
    except Exception:
        pass
    return EditorMemory()


def save_editor_memory(path: Path, memory: EditorMemory) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(memory.to_dict(), separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
        return True
    except Exception:
        return False
