from __future__ import annotations

import math

from engine.compiler.graph_compiler import RenderGraphCompiler
from engine.graph.operations import RenderOperation
from engine.pipeline.render_pipeline import ClipRenderJob, EffectPlan, atempo_chain
from engine.presets.operation_presets import OperationPresetMapper


class HybridFilterCompiler:
    """
    Builds the FFmpeg portion of an operation graph.

    Expensive compositor operations are listed in gpu_effects instead of being
    forced into giant FFmpeg expressions.
    """

    MOOD_EQ = {
        "hype": {"contrast": 1.28, "saturation": 1.35, "brightness": 0.02, "gamma": 0.92},
        "phonk": {"contrast": 1.35, "saturation": 1.18, "brightness": 0.01, "gamma": 0.88},
        "dark": {"contrast": 1.32, "saturation": 0.82, "brightness": -0.04, "gamma": 0.85},
        "villain": {"contrast": 1.38, "saturation": 0.90, "brightness": -0.03, "gamma": 0.88},
        "sad": {"contrast": 1.12, "saturation": 0.72, "brightness": 0.02, "gamma": 1.05},
        "emotional": {"contrast": 1.10, "saturation": 0.80, "brightness": 0.03, "gamma": 1.02},
        "dramatic": {"contrast": 1.25, "saturation": 1.15, "brightness": -0.01, "gamma": 0.95},
        "cinematic": {"contrast": 1.18, "saturation": 1.05, "brightness": -0.01, "gamma": 0.95},
        "cyberpunk": {"contrast": 1.32, "saturation": 1.45, "brightness": -0.02, "gamma": 0.90},
        "chill": {"contrast": 1.05, "saturation": 1.08, "brightness": 0.04, "gamma": 1.05},
        "warm": {"contrast": 1.15, "saturation": 1.20, "brightness": 0.03, "gamma": 0.98},
    }

    MOOD_CURVES = {
        "hype": {"r": "0/0 0.5/0.55 1/1", "g": "0/0 0.5/0.50 1/0.97", "b": "0/0 0.5/0.45 1/0.92"},
        "phonk": {"r": "0/0 0.4/0.45 1/1", "g": "0/0 0.5/0.48 1/0.95", "b": "0/0 0.5/0.42 1/0.88"},
        "dark": {"r": "0/0 0.3/0.28 1/0.95", "g": "0/0 0.5/0.50 1/0.98", "b": "0/0.02 0.5/0.53 1/1"},
        "villain": {"r": "0/0 0.35/0.32 1/0.96", "g": "0/0 0.5/0.49 1/0.96", "b": "0/0.02 0.5/0.54 1/1"},
        "sad": {"r": "0/0.03 0.5/0.50 1/0.95", "g": "0/0.02 0.5/0.51 1/0.97", "b": "0/0 0.5/0.55 1/1"},
        "emotional": {"r": "0/0.02 0.5/0.51 1/0.96", "g": "0/0.01 0.5/0.51 1/0.97", "b": "0/0 0.5/0.54 1/1"},
        "dramatic": {"r": "0/0 0.4/0.44 1/1", "g": "0/0 0.5/0.50 1/0.97", "b": "0/0 0.5/0.50 1/0.96"},
        "cyberpunk": {"r": "0/0 0.4/0.38 1/0.98", "g": "0/0 0.5/0.49 1/0.97", "b": "0/0.02 0.5/0.56 1/1"},
        "warm": {"r": "0/0 0.5/0.56 1/1", "g": "0/0 0.5/0.52 1/0.99", "b": "0/0 0.5/0.46 1/0.92"},
        "chill": {"r": "0/0.02 0.5/0.52 1/0.98", "g": "0/0.01 0.5/0.51 1/0.98", "b": "0/0 0.5/0.52 1/1"},
    }

    def __init__(self, job: ClipRenderJob):
        self.job = job
        self.w = max(16, int(job.width))
        self.h = max(16, int(job.height))
        self.fps = max(1.0, float(job.fps))
        self.i = max(0.0, min(1.0, float(job.intensity)))

    def compile(self) -> EffectPlan:
        plan = EffectPlan()
        operations = OperationPresetMapper.operations_for(self.job)
        render_graph = RenderGraphCompiler().compile(self.job, operations)
        plan.operations = operations
        plan.render_graph = render_graph
        plan.cache_key = render_graph.cache_key(
            source_key=f"{self.job.input_path}:{self.job.trim_start:.3f}:{self.job.trim_dur:.3f}"
        )

        if abs(self.job.speed_mult - 1.0) > 0.04:
            pts = 1.0 / max(0.05, self.job.speed_mult)
            plan.ffmpeg_filters.append(f"setpts={pts:.4f}*PTS")

        plan.ffmpeg_filters.append(f"scale={self.w}:{self.h}:flags=lanczos")

        for operation in operations:
            self._append_operation(plan, operation)
        return plan

    def _append_operation(self, plan: EffectPlan, operation: RenderOperation) -> None:
        op = operation.op
        params = operation.params

        if self._should_route_to_gpu(operation):
            plan.gpu_effects.append(operation.stable_dict())
            plan.notes.append(f"Operation {op} is a GPU candidate.")

        if op == "color_grade":
            plan.ffmpeg_filters.append(self._eq_filter())
            curves = self._curves_filter()
            if curves:
                plan.ffmpeg_filters.append(curves)
        elif op == "rgb_split":
            plan.ffmpeg_filters.append(self._chromatic(float(params.get("amount", self.i))))
        elif op == "contrast_punch":
            amount = float(params.get("amount", 0.2))
            plan.ffmpeg_filters.append(f"eq=contrast={1.0 + amount:.3f}")
        elif op == "drop_punch":
            amount = float(params.get("amount", self.i * 0.18))
            plan.ffmpeg_filters.append(f"eq=contrast={1.0 + amount:.3f}:brightness={-amount * 0.05:.4f}")
        elif op in {"glow", "soft_sharpen"}:
            radius = int(params.get("radius", 7))
            amount = float(params.get("amount", self.i * 0.22))
            radius = max(3, radius if radius % 2 == 1 else radius + 1)
            plan.ffmpeg_filters.append(f"unsharp=lx={radius}:ly={radius}:la={amount:.2f}:cx=0:cy=0:ca=0")
        elif op == "saturation_pulse":
            amount = float(params.get("amount", self.i * 0.35))
            plan.ffmpeg_filters.append(f"eq=saturation={1.0 + amount:.3f}")
        elif op == "warm_halation":
            amount = float(params.get("amount", self.i * 0.06))
            plan.ffmpeg_filters.append(f"curves=r='0/0 0.6/{0.6 + amount:.3f} 1/1'")
        elif op == "temporal_blend":
            frames = max(2, min(5, int(params.get("frames", 3))))
            plan.ffmpeg_filters.append(f"tmix=frames={frames}:weights='1'")
        elif op == "grain":
            amount = int(params.get("amount", max(2, int(self.i * 18))))
            plan.ffmpeg_filters.append(f"noise=alls={amount}:allf=t+u")
        elif op == "scanlines":
            opacity = float(params.get("opacity", self.i * 0.18))
            plan.ffmpeg_filters.append(f"drawgrid=width=0:height=4:thickness=1:color=black@{opacity:.2f}")
        elif op == "desaturate":
            amount = float(params.get("amount", self.i * 0.65))
            plan.ffmpeg_filters.append(f"eq=saturation={max(0.0, 1.0 - amount):.3f}")
        elif op == "sharpen":
            amount = float(params.get("amount", self.i))
            plan.ffmpeg_filters.append(f"unsharp=lx=5:ly=5:la={amount:.2f}:cx=5:cy=5:ca={amount * 0.35:.2f}")
        elif op == "transform":
            plan.ffmpeg_filters.extend(self._cheap_transform_filter(operation))
        elif op == "shake":
            plan.ffmpeg_filters.extend(self._cheap_shake_filter(operation))
        elif op == "vignette":
            intensity = float(params.get("intensity", self.i))
            angle = min(math.pi / 2, 0.15 + intensity * 0.50)
            plan.ffmpeg_filters.append(f"vignette=angle={angle:.4f}:mode=forward")
        elif op == "overlay":
            plan.notes.append(f"Overlay operation {params.get('asset', 'unknown')} requires compositor backend.")
        elif op == "raw_filter":
            raw = str(params.get("filter", "")).strip()
            if raw:
                plan.ffmpeg_filters.append(raw)

    def _should_route_to_gpu(self, operation: RenderOperation) -> bool:
        if not self.job.prefer_gpu_effects:
            return False
        if operation.backend_hint == "gpu":
            return True
        return operation.op in {"glow", "warm_halation", "temporal_blend", "transform", "shake", "overlay", "mask", "composite"}

    def _eq_filter(self) -> str:
        eq = self.MOOD_EQ.get(self.job.mood, self.MOOD_EQ["hype"])
        contrast = 1.0 + (eq["contrast"] - 1.0) * self.i
        saturation = 1.0 + (eq["saturation"] - 1.0) * self.i
        brightness = eq["brightness"] * self.i
        gamma = 1.0 + (eq["gamma"] - 1.0) * self.i

        if self.job.is_drop or self.job.is_climax:
            contrast = min(contrast * 1.12, 2.0)
            saturation = min(saturation * 1.10, 3.0)
        if self.job.is_breath:
            contrast = 1.0 + (contrast - 1.0) * 0.4
            saturation = 1.0 + (saturation - 1.0) * 0.4
            brightness *= 0.3

        return (
            f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:"
            f"brightness={brightness:.4f}:gamma={gamma:.3f}"
        )

    def _curves_filter(self) -> str:
        curves = self.MOOD_CURVES.get(self.job.mood)
        if not curves:
            return ""
        return f"curves=r='{curves['r']}':g='{curves['g']}':b='{curves['b']}'"

    def _chromatic(self, strength: float) -> str:
        if strength < 0.05:
            return ""
        shift = max(1, int(strength * 10))
        return f"rgbashift=rh={shift}:bh=-{shift}:rv=0:bv=0"

    def _cheap_transform_filter(self, operation: RenderOperation) -> list[str]:
        params = operation.params
        start = float(params.get("zoom_start", self.job.zoom_start))
        end = float(params.get("zoom_end", self.job.zoom_end))
        if abs(start - end) < 0.005 and abs(start - 1.0) < 0.005:
            return []

        z = max(start, end, 1.0)
        crop_w = int(self.w / z) // 2 * 2
        crop_h = int(self.h / z) // 2 * 2
        x = max(0, (self.w - crop_w) // 2)
        y = max(0, (self.h - crop_h) // 2)
        return [f"crop={crop_w}:{crop_h}:{x}:{y}", f"scale={self.w}:{self.h}:flags=lanczos"]

    def _cheap_shake_filter(self, operation: RenderOperation) -> list[str]:
        params = operation.params
        ai = max(0.0, min(1.0, float(params.get("intensity", self.job.anim_intensity))))
        if ai < 0.01:
            return []

        mode = str(params.get("mode", self.job.animation))
        amp = max(2, int(ai * (18 if mode != "subtle_drift" else 8)))
        crop_w = max(16, self.w - amp * 2) // 2 * 2
        crop_h = max(16, self.h - amp * 2) // 2 * 2
        if mode == "camera_slam":
            x = f"{amp}+{amp}*sin(n*0.9)*exp(-n/{max(1.0, self.fps * 0.15):.3f})"
            y = str(amp)
        else:
            x = f"{amp}+{max(1, amp // 2)}*sin(n/{self.fps:.3f}*6.283)"
            y = f"{amp}+{max(1, amp // 3)}*cos(n/{self.fps:.3f}*4.712)"
        return [f"crop={crop_w}:{crop_h}:{x}:{y}", f"scale={self.w}:{self.h}:flags=lanczos"]

    def audio_filter(self) -> str:
        filters = []
        if abs(self.job.speed_mult - 1.0) > 0.04:
            filters.extend(atempo_chain(self.job.speed_mult))
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        return ",".join(filters)
