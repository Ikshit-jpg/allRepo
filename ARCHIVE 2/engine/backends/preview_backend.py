from __future__ import annotations

from engine.graph.render_graph import RenderGraph


class CPUFallbackBackend:
    name = "cpu_fallback"
    available = True

    def can_render(self, graph: RenderGraph, target: str) -> bool:
        return target in {"preview", "proxy"}

    def render_graph(self, graph: RenderGraph, target: str, output_path: str) -> tuple[bool, str]:
        return False, "CPU frame fallback is a contract placeholder; FFmpeg fallback handles temp renders for now."
