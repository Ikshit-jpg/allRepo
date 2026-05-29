from __future__ import annotations

from dataclasses import dataclass

from .operations import RenderOperation


@dataclass(frozen=True)
class RenderNode:
    node_id: str
    op: RenderOperation
    inputs: tuple[str, ...] = ()
    cacheable: bool = True
    stage: str = "video"
