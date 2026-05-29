from __future__ import annotations

import concurrent.futures
from typing import Any


class BackgroundRenderScheduler:
    """Tiny background scheduler so export and preview jobs do not freeze UI."""

    def __init__(self, max_workers: int = 1):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._jobs: dict[str, concurrent.futures.Future[tuple[bool, Any]]] = {}

    def submit(self, job_id: str, fn: Any, *args: Any, **kwargs: Any) -> str:
        self._jobs[job_id] = self._executor.submit(fn, *args, **kwargs)
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        future = self._jobs.get(job_id)
        if future is None:
            return {"state": "missing"}
        if future.running():
            return {"state": "running"}
        if not future.done():
            return {"state": "queued"}
        try:
            return {"state": "done", "result": future.result()}
        except Exception as exc:
            return {"state": "failed", "error": str(exc)}
