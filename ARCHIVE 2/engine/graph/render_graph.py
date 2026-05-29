from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .nodes import RenderNode
from .operations import stable_value


@dataclass
class RenderGraph:
    nodes: list[RenderNode] = field(default_factory=list)
    output_node: str = ""
    notes: list[str] = field(default_factory=list)

    def cache_key(self, source_key: str = "") -> str:
        payload = {
            "source": source_key,
            "nodes": [
                {
                    "id": node.node_id,
                    "op": node.op.stable_dict(),
                    "inputs": node.inputs,
                    "cacheable": node.cacheable,
                    "stage": node.stage,
                }
                for node in self.nodes
            ],
            "output": self.output_node,
        }
        return hash_payload(payload)


def hash_payload(payload: dict[str, Any]) -> str:
    stable = json.dumps(stable_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
