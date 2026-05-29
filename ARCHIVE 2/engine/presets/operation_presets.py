from __future__ import annotations

from engine.graph.operations import RenderOperation


class OperationPresetMapper:
    """
    Compatibility bridge from old effect names to modular operations.

    New code should fill ClipRenderJob.operations directly. The string effect
    remains for old callers, but it is no longer the engine's core model.
    """

    @classmethod
    def operations_for(cls, job: object) -> list[RenderOperation]:
        operations = getattr(job, "operations", None)
        if operations:
            return list(operations)
        if getattr(job, "effect", "") == "__clean__":
            return []

        i = max(0.0, min(1.0, float(getattr(job, "intensity", 0.75))))
        ops: list[RenderOperation] = [
            RenderOperation("color_grade", {"mood": getattr(job, "mood", "hype"), "intensity": i}),
        ]

        effect = getattr(job, "effect", "cinematic_grade")
        if effect in {"impact_flash_combo", "full_impact", "zoom_burst_shake"}:
            ops.extend(
                [
                    RenderOperation("rgb_split", {"amount": i * 0.85}, backend_hint="gpu"),
                    RenderOperation("contrast_punch", {"amount": i * 0.25}),
                ]
            )
        elif effect in {"energy_aura", "dark_surge"}:
            ops.extend(
                [
                    RenderOperation("glow", {"radius": 5, "amount": i * 0.45}, backend_hint="gpu"),
                    RenderOperation("saturation_pulse", {"amount": i * 0.35}, sync="section"),
                ]
            )
        elif effect in {"emotional_peak", "portrait_enhance", "emotional_glow"}:
            ops.extend(
                [
                    RenderOperation("glow", {"radius": 9, "amount": i * 0.25}, backend_hint="gpu"),
                    RenderOperation("warm_halation", {"amount": i * 0.06}, backend_hint="gpu"),
                ]
            )
        elif effect in {"speed_blur_lines", "chromatic_shake"}:
            ops.extend(
                [
                    RenderOperation("temporal_blend", {"frames": max(2, min(5, int(2 + i * 4)))}, backend_hint="gpu"),
                    RenderOperation("rgb_split", {"amount": i * 0.70}, backend_hint="gpu"),
                ]
            )
        elif effect in {"atmospheric_grade", "god_rays", "cinematic_grade"}:
            ops.append(RenderOperation("soft_sharpen", {"radius": 7, "amount": i * 0.22}))
        elif effect in {"vhs_glitch", "datamosh_corrupt"}:
            ops.extend(
                [
                    RenderOperation("rgb_split", {"amount": i * 0.90}, backend_hint="gpu"),
                    RenderOperation("grain", {"amount": int(i * 18), "style": "vhs"}),
                    RenderOperation("scanlines", {"opacity": i * 0.18}),
                ]
            )
        elif effect == "impact_frame":
            ops.extend(
                [
                    RenderOperation("desaturate", {"amount": i * 0.65}),
                    RenderOperation("contrast_punch", {"amount": i * 1.2}),
                    RenderOperation("sharpen", {"amount": i * 1.2}),
                ]
            )

        if abs(getattr(job, "zoom_start", 1.0) - getattr(job, "zoom_end", 1.0)) >= 0.005 or abs(getattr(job, "zoom_start", 1.0) - 1.0) >= 0.005:
            ops.append(
                RenderOperation(
                    "transform",
                    {
                        "zoom_start": getattr(job, "zoom_start", 1.0),
                        "zoom_end": getattr(job, "zoom_end", 1.05),
                        "curve": getattr(job, "zoom_curve", "ease"),
                        "duration": getattr(job, "trim_dur", 5.0),
                    },
                    sync="time",
                    backend_hint="gpu",
                )
            )

        if getattr(job, "anim_intensity", 0.0) >= 0.01 and getattr(job, "animation", "") in {"handheld_shake", "camera_slam", "subtle_drift"}:
            ops.append(
                RenderOperation(
                    "shake",
                    {"mode": getattr(job, "animation", "subtle_drift"), "intensity": getattr(job, "anim_intensity", 0.4)},
                    sync="time",
                    backend_hint="gpu",
                )
            )

        if getattr(job, "light_leak", 0.0) > 0.01:
            ops.append(RenderOperation("overlay", {"asset": "light_leak", "opacity": getattr(job, "light_leak", 0.0)}, backend_hint="gpu"))
        if getattr(job, "particle_type", ""):
            ops.append(RenderOperation("overlay", {"asset": getattr(job, "particle_type"), "seed": getattr(job, "seed", 42)}, backend_hint="gpu"))
        if getattr(job, "is_drop", False) and not getattr(job, "is_breath", False):
            ops.append(RenderOperation("drop_punch", {"amount": i * 0.18}, sync="beat"))

        ops.extend(
            [
                RenderOperation("vignette", {"intensity": i * (0.35 if getattr(job, "is_breath", False) else 1.0)}),
                RenderOperation("grain", {"amount": max(2, int(i * (8 if getattr(job, "is_breath", False) else 18))), "style": "film"}),
            ]
        )
        return ops
