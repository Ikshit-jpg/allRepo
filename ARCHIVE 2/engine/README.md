# ARCHIVE Render Engine

This package is the permanent render engine. Keep AI planning, graph modeling,
compilation, backend execution, caching, and scheduling in separate modules.

## Ownership

- `graph/`: render graph data, nodes, and operation primitives.
- `compiler/`: translators from operations/graphs into backend-specific plans.
- `backends/`: execution engines such as FFmpeg, GPU compositor, and preview.
- `cache/`: render/proxy/cache key ownership.
- `scheduler/`: background job execution.
- `presets/`: legacy effect-name compatibility and preset expansion.
- `pipeline/`: render jobs, timeline jobs, settings, and shared contracts.

## Rule

New visual ideas should become operations or planner choices first. Do not grow
new giant effect switchboards in the backend.
