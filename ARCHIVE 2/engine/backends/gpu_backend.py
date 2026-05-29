from __future__ import annotations

from engine.graph.render_graph import RenderGraph


class GPUCompositorBackend:
    """
    Placeholder contract for the future GPU renderer.

    It keeps GPU planning separate from FFmpeg export while the real compositor
    is still being built.
    """

    name = "gpu_compositor"
    available = False

    def can_render(self, graph: RenderGraph, target: str) -> bool:
        return self.available and target in {"preview", "export"}

    def render_graph(self, graph: RenderGraph, target: str, output_path: str) -> tuple[bool, str]:
        return False, "GPU compositor backend is planned but not enabled yet."

    def render(self, job: object, plan: object) -> tuple[bool, str]:
        return False, "GPU compositor backend is planned but not enabled yet."
