from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any


DEFAULT_SIGNATURE_BYTES = 2 * 1024 * 1024
_FILE_LOCK = RLock()


def partial_file_sha256(path: Path, *, sample_bytes: int = DEFAULT_SIGNATURE_BYTES) -> str:
    """Hash the first N bytes of a file for safer cache invalidation than mtime alone."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max(1, int(sample_bytes))))
    return h.hexdigest()


def file_cache_signature(path: Path, *, sample_bytes: int = DEFAULT_SIGNATURE_BYTES) -> str:
    stat = path.stat()
    digest = partial_file_sha256(path, sample_bytes=sample_bytes)
    return f"v2:{stat.st_size}:{stat.st_mtime_ns}:{digest}"


def load_json_cache(path: Path) -> dict[str, Any]:
    with _FILE_LOCK:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def write_json_cache(path: Path, payload: dict[str, Any]) -> bool:
    with _FILE_LOCK:
        tmp_name = ""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
                tmp.write(data)
                tmp_name = tmp.name
            os.replace(tmp_name, path)
            return True
        except Exception:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except Exception:
                    pass
            return False


class LruMemoryCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max(1, int(max_items))
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


def prune_files(
    directory: Path,
    pattern: str,
    *,
    max_files: int,
    max_age_seconds: int,
    keep_names: set[str] | None = None,
) -> int:
    keep = keep_names or set()
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    files = sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for idx, path in enumerate(files):
        if path.name in keep:
            continue
        age_sec = now - float(path.stat().st_mtime)
        if idx >= max(1, int(max_files)) or age_sec >= max(1, int(max_age_seconds)):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
    return removed
