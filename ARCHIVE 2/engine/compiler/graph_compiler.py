from __future__ import annotations

from engine.graph.nodes import RenderNode
from engine.graph.operations import RenderOperation, operation_stage
from engine.graph.render_graph import RenderGraph
from engine.pipeline.render_pipeline import ClipRenderJob


class RenderGraphCompiler:
    def compile(self, job: ClipRenderJob, operations: list[RenderOperation]) -> RenderGraph:
        nodes: list[RenderNode] = []
        prev = "source"
        notes: list[str] = []

        source_op = RenderOperation(
            "source_clip",
            {
                "path": job.input_path,
                "trim_start": job.trim_start,
                "trim_dur": job.trim_dur,
                "width": job.width,
                "height": job.height,
                "fps": job.fps,
            },
        )
        nodes.append(RenderNode("source", source_op, inputs=(), cacheable=True, stage="decode"))

        if abs(job.speed_mult - 1.0) > 0.04:
            node_id = "speed_001"
            op = RenderOperation("speed", {"multiplier": job.speed_mult})
            nodes.append(RenderNode(node_id, op, inputs=(prev,), cacheable=True, stage="temporal"))
            prev = node_id

        for index, op in enumerate(operations, start=1):
            node_id = f"{op.op}_{index:03d}"
            cacheable = op.sync not in {"realtime"} and op.op not in {"preview_marker"}
            nodes.append(RenderNode(node_id, op, inputs=(prev,), cacheable=cacheable, stage=operation_stage(op.op)))
            prev = node_id

        graph = RenderGraph(nodes=nodes, output_node=prev, notes=notes)
        graph.notes.append("Compiled operation graph; legacy effect name is not used past preset expansion.")
        return graph
