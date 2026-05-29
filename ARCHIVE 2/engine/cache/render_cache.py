from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from engine.graph.render_graph import hash_payload


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: Path
    kind: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RenderCache:
    """Shared cache registry for node, proxy, audio, thumb, and effect caches."""

    def __init__(self, root: str | Path = "runtime/render_cache"):
        self.root = Path(root)
        self._entries: dict[str, CacheEntry] = {}

    def make_key(self, kind: str, payload: dict[str, Any]) -> str:
        return f"{kind}:{hash_payload(payload)}"

    def path_for(self, kind: str, key: str, suffix: str) -> Path:
        safe_key = key.replace(":", "_")
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{safe_key}{suffix}"

    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._entries.get(key)
        if entry and entry.path.exists():
            return entry
        return None

    def put(self, key: str, path: str | Path, kind: str, metadata: Optional[dict[str, Any]] = None) -> CacheEntry:
        entry = CacheEntry(
            key=key,
            path=Path(path),
            kind=kind,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._entries[key] = entry
        return entry
