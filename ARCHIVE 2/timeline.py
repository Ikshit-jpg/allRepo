from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimelineTrack:
    name: str
    layer_type: str
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TimelinePlan:
    tracks: list[TimelineTrack] = field(default_factory=list)

    def add(self, track_name: str, layer_type: str, item: dict[str, Any]) -> None:
        for track in self.tracks:
            if track.name == track_name:
                track.items.append(item)
                return
        self.tracks.append(TimelineTrack(track_name, layer_type, [item]))

    def to_dict(self) -> dict[str, Any]:
        return {"tracks": [{"name": t.name, "type": t.layer_type, "items": t.items} for t in self.tracks]}
