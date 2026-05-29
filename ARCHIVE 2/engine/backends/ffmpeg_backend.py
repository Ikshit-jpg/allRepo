from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from engine.backends.gpu_backend import GPUCompositorBackend
from engine.backends.preview_backend import CPUFallbackBackend
from engine.cache.render_cache import RenderCache
from engine.compiler.ffmpeg_compiler import HybridFilterCompiler
from engine.graph.operations import RenderOperation
from engine.graph.render_graph import RenderGraph
from engine.pipeline.render_pipeline import (
    ClipRenderJob,
    PreviewSettings,
    ProxySettings,
    RenderBackend,
    TimelineGraph,
    TransitionJob,
    copy_job,
    parse_fps,
)
from engine.scheduler.background_scheduler import BackgroundRenderScheduler


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class XFadeBuilder:
    XFADE_MAP = {
        "dissolve": "dissolve",
        "fade_in": "fade",
        "fade_black": "fadeblack",
        "fade_white": "fadewhite",
        "wipe_right": "wiperight",
        "wipe_left": "wipeleft",
        "wipe_up": "wipeup",
        "slide_right": "slideright",
        "slide_left": "slideleft",
        "circle_open": "circleopen",
        "circle_close": "circleclose",
        "radial": "radial",
        "zoom_in": "zoomin",
        "pixelize": "pixelize",
        "hblur": "hblur",
        "squeeze_h": "squeezeh",
        "smooth_left": "smoothleft",
        "smooth_right": "smoothright",
        "zoom_punch": "zoomin",
        "glitch_cut": "pixelize",
        "rgb_split_sweep": "smoothleft",
        "impact_cut": "fade",
        "warp_cut": "hblur",
        "flash_cut": "fadewhite",
        "whip_pan_right": "wiperight",
        "whip_pan_up": "wipeup",
        "energy_slash": "diagtr",
        "smear_cut": "hblur",
        "luma_wipe": "fade",
        "cross_zoom": "zoomin",
        "tv_static": "pixelize",
        "spin": "radial",
        "ripple": "hblur",
        "shatter_cut": "pixelize",
        "dolly_zoom_trans": "zoomin",
    }

    @classmethod
    def get_xfade_name(cls, trans_type: str) -> str:
        return cls.XFADE_MAP.get(trans_type, "dissolve")


class FFmpegEngine:
    def __init__(
        self,
        ffmpeg_path: str = FFMPEG,
        ffprobe_path: str = FFPROBE,
        default_crf: int = 16,
        default_preset: str = "veryfast",
        threads: int = 0,
        enable_gpu_preview_plan: bool = True,
        cache_root: str | Path = "runtime/render_cache",
    ):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self.default_crf = default_crf
        self.default_preset = default_preset
        self.threads = threads or max(1, (os.cpu_count() or 2) - 1)
        self.cache = RenderCache(cache_root)
        self.gpu_backend = GPUCompositorBackend()
        self.cpu_backend = CPUFallbackBackend()
        self.backends: list[RenderBackend] = [self.gpu_backend, self.cpu_backend]
        self.scheduler = BackgroundRenderScheduler(max_workers=1)
        self.enable_gpu_preview_plan = enable_gpu_preview_plan

    def _run(self, args: list[str], timeout: int = 900) -> tuple[bool, str]:
        cmd = [self.ffmpeg, "-hide_banner", "-y"] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            if result.returncode != 0:
                return False, (result.stderr or "")[-2000:]
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except FileNotFoundError:
            return False, f"FFmpeg not found at: {self.ffmpeg}"
        except Exception as exc:
            return False, str(exc)

    def plan_clip(self, job: ClipRenderJob):
        return HybridFilterCompiler(job).compile()

    def build_render_graph(self, job: ClipRenderJob) -> RenderGraph:
        plan = self.plan_clip(job)
        return plan.render_graph or RenderGraph()

    def plan_preview(self, job: ClipRenderJob, settings: PreviewSettings = PreviewSettings()):
        preview_job = copy_job(
            job,
            output_path=job.output_path,
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            crf=settings.crf,
            preset=settings.preset,
        )
        plan = self.plan_clip(preview_job)
        plan.notes.append("Preview plan uses proxy resolution and keeps compositor operations as backend candidates.")
        return plan

    def render_proxy(self, job: ClipRenderJob, settings: ProxySettings = ProxySettings()) -> tuple[bool, str]:
        key = self.cache.make_key(
            "proxy",
            {
                "input": job.input_path,
                "trim_start": job.trim_start,
                "trim_dur": job.trim_dur,
                "settings": settings.__dict__,
            },
        )
        cached = self.cache.get(key)
        if cached:
            return True, str(cached.path)

        proxy_path = self.cache.path_for("proxy", key, ".mp4")
        proxy_job = copy_job(
            job,
            output_path=str(proxy_path),
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            crf=settings.crf,
            preset=settings.preset,
            operations=[
                RenderOperation("color_grade", {"mood": job.mood, "intensity": min(job.intensity, 0.5)}),
                RenderOperation("vignette", {"intensity": min(job.intensity, 0.35)}),
            ],
        )
        ok, _elapsed = self.render_clip(proxy_job)
        if ok:
            self.cache.put(key, proxy_path, "proxy", {"settings": settings.__dict__})
            return True, str(proxy_path)
        return False, "Proxy render failed"

    def schedule_render_clip(self, job: ClipRenderJob, job_id: Optional[str] = None) -> str:
        render_id = job_id or self.cache.make_key(
            "background_render",
            {"input": job.input_path, "output": job.output_path, "trim": [job.trim_start, job.trim_dur], "time": time.time()},
        )
        return self.scheduler.submit(render_id, self.render_clip, job)

    def render_clip(self, job: ClipRenderJob) -> tuple[bool, float]:
        t0 = time.time()
        valid, reason = self._validate_clip_job(job)
        if not valid:
            print(f"  [ENGINE] Clip rejected: {reason}")
            return False, 0.0

        compiler = HybridFilterCompiler(job)
        plan = compiler.compile()
        cache_key = self.cache.make_key(
            "effect",
            {
                "plan": plan.cache_key,
                "input": job.input_path,
                "trim_start": job.trim_start,
                "trim_dur": job.trim_dur,
                "size": [job.width, job.height],
                "fps": job.fps,
                "crf": job.crf,
            },
        )
        cached = self.cache.get(cache_key)
        if cached:
            try:
                Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached.path, job.output_path)
                return True, time.time() - t0
            except Exception:
                pass

        if self.gpu_backend.available and plan.gpu_effects:
            ok, err = self.gpu_backend.render(job, plan)
            if ok:
                self.cache.put(cache_key, job.output_path, "effect", {"backend": self.gpu_backend.name})
                return True, time.time() - t0
            print(f"  [ENGINE] GPU path unavailable, using FFmpeg fallback: {err[:160]}")

        vf = plan.vf or f"scale={job.width}:{job.height}:flags=lanczos"
        if job.hold_frames > 0:
            hold_dur = job.hold_frames / max(1.0, job.fps)
            vf = f"tpad=start_duration={hold_dur:.3f}:start_mode=clone,{vf}"

        has_audio = job.include_audio and self._probe_has_audio(job.input_path)
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-ss",
            f"{job.trim_start:.3f}",
            "-t",
            f"{job.trim_dur:.3f}",
            "-i",
            job.input_path,
            "-vf",
            vf,
        ]
        if has_audio:
            args += ["-af", compiler.audio_filter()]
        elif not job.include_audio:
            args += ["-an"]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            job.preset or self.default_preset,
            "-crf",
            str(job.crf or self.default_crf),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.threads),
            "-movflags",
            "+faststart",
            job.output_path,
        ]

        ok, err = self._run(args, timeout=600)
        elapsed = time.time() - t0
        if not ok:
            print(f"  [ENGINE] Clip render failed: {err[:240]}")
            ok, err = self._fallback_render(job)
            if not ok:
                print(f"  [ENGINE] Fallback failed: {err[:240]}")
                return False, 0.0
        if os.path.exists(job.output_path):
            self.cache.put(cache_key, job.output_path, "effect", {"backend": "ffmpeg", "gpu_candidates": len(plan.gpu_effects)})
        return True, elapsed

    def render_transition(self, job: TransitionJob) -> tuple[bool, float]:
        t0 = time.time()
        valid, reason = self._validate_transition_job(job)
        if not valid:
            print(f"  [ENGINE] Transition rejected: {reason}")
            return False, 0.0
        ok, err = self._xfade_transition(job, XFadeBuilder.get_xfade_name(job.trans_type))
        if not ok:
            print(f"  [ENGINE] Transition failed: {err[:200]}")
        return ok, time.time() - t0

    def _xfade_transition(self, job: TransitionJob, xfade_name: str) -> tuple[bool, str]:
        dur = max(0.05, float(job.duration_sec))
        probe_a = self._probe_duration(job.clip_a)
        offset = max(0.0, probe_a - dur - 0.05)
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        vf = (
            f"[0:v]scale={job.width}:{job.height}:flags=lanczos,setsar=1[v0];"
            f"[1:v]scale={job.width}:{job.height}:flags=lanczos,setsar=1[v1];"
            f"[v0][v1]xfade=transition={xfade_name}:duration={dur:.3f}:offset={offset:.3f}[v]"
        )
        args = [
            "-i",
            job.clip_a,
            "-i",
            job.clip_b,
            "-filter_complex",
            vf,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            job.preset,
            "-crf",
            str(job.crf),
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{dur:.3f}",
            job.output_path,
        ]
        return self._run(args, timeout=120)

    def assemble_timeline(self, graph: TimelineGraph, crf: int = 16, preset: str = "veryfast") -> tuple[bool, str]:
        clips = [clip.render_job.output_path for clip in graph.clips]
        transitions = [transition.kind for transition in graph.transitions]
        durs = [transition.duration for transition in graph.transitions]
        return self.assemble_with_xfade(
            clips=clips,
            transitions=transitions,
            trans_durs=durs,
            output=graph.output_path,
            music=graph.music_path,
            total_dur=graph.total_duration(),
            crf=crf,
            preset=preset,
        )

    def assemble_with_xfade(
        self,
        clips: list[str],
        transitions: list[str],
        trans_durs: list[float],
        output: str,
        music: Optional[str],
        total_dur: float,
        crf: int,
        preset: str,
    ) -> tuple[bool, str]:
        n = len(clips)
        if n == 0:
            return False, "No clips"
        if n == 1:
            return self._single_clip_with_music(clips[0], music, output, total_dur, crf, preset)

        for clip in clips:
            if not os.path.exists(clip):
                return False, f"Clip not found: {clip}"

        if n > 20:
            return self._concat_assemble(clips, music, output, total_dur, crf, preset)

        input_args: list[str] = []
        for clip in clips:
            input_args += ["-i", clip]

        durations = [self._probe_duration(clip) for clip in clips]
        fc_parts: list[str] = []
        current_offset = 0.0
        prev_label = "[0:v]"

        for idx in range(n - 1):
            trans_dur = trans_durs[idx] if idx < len(trans_durs) else 0.3
            trans_type = transitions[idx] if idx < len(transitions) else "dissolve"
            xfade_name = XFadeBuilder.get_xfade_name(trans_type)
            current_offset += max(0.0, durations[idx] - trans_dur)
            out_label = f"[v{idx}]" if idx < n - 2 else "[vout]"
            fc_parts.append(
                f"{prev_label}[{idx + 1}:v]xfade=transition={xfade_name}:"
                f"duration={trans_dur:.3f}:offset={current_offset:.3f}{out_label}"
            )
            prev_label = out_label

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        args = input_args + ["-filter_complex", ";".join(fc_parts), "-map", "[vout]"]
        if music and os.path.exists(music):
            args += ["-i", music, "-map", f"{n}:a", "-t", f"{total_dur:.3f}", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            args += ["-t", f"{total_dur:.3f}"]

        args += [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            str(self.threads),
            output,
        ]
        return self._run(args, timeout=900)

    def _single_clip_with_music(
        self,
        clip: str,
        music: Optional[str],
        output: str,
        total_dur: float,
        crf: int,
        preset: str,
    ) -> tuple[bool, str]:
        if not os.path.exists(clip):
            return False, f"Clip not found: {clip}"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        args = ["-i", clip]
        if music and os.path.exists(music):
            args += ["-i", music, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        args += [
            "-t",
            f"{total_dur:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output,
        ]
        return self._run(args, timeout=600)

    def _concat_assemble(
        self,
        clips: list[str],
        music: Optional[str],
        output: str,
        total_dur: float,
        crf: int,
        preset: str,
    ) -> tuple[bool, str]:
        tmp = Path(tempfile.mkdtemp(prefix="archive_temp_concat_"))
        try:
            list_file = tmp / "list.txt"
            with open(list_file, "w", encoding="utf-8") as handle:
                for clip in clips:
                    escaped = str(clip).replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            args = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
            if music and os.path.exists(music):
                args += ["-i", music, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
            args += [
                "-t",
                f"{total_dur:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-threads",
                str(self.threads),
                output,
            ]
            return self._run(args, timeout=900)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _fallback_render(self, job: ClipRenderJob) -> tuple[bool, str]:
        i = max(0.0, min(1.0, job.intensity))
        vf = (
            f"scale={job.width}:{job.height}:flags=lanczos,"
            f"eq=contrast={1 + i * 0.2:.2f}:saturation={1 + i * 0.2:.2f},"
            f"vignette=angle={0.2 + i * 0.3:.3f}"
        )
        args = [
            "-ss",
            f"{job.trim_start:.3f}",
            "-t",
            f"{job.trim_dur:.3f}",
            "-i",
            job.input_path,
            "-vf",
            vf,
        ]
        if not job.include_audio:
            args += ["-an"]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(job.crf + 4),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.threads),
            job.output_path,
        ]
        return self._run(args, timeout=120)

    def post_process(self, input_path: str, output_path: str, mood: str = "hype") -> tuple[bool, str]:
        if not os.path.exists(input_path):
            return False, f"Input not found: {input_path}"
        mood_curves = {
            "hype": "curves=r='0/0 0.5/0.54 1/1':b='0/0 0.5/0.46 1/0.94'",
            "dark": "curves=r='0/0 0.3/0.28 1/0.95':b='0/0.02 0.5/0.53 1/1'",
            "sad": "curves=all='0/0.03 0.5/0.52 1/0.97':b='0/0 0.5/0.54 1/1'",
            "cyberpunk": "curves=r='0/0 0.4/0.38 1/1':b='0/0.02 0.5/0.55 1/1'",
            "dramatic": "curves=r='0/0 0.4/0.44 1/1':b='0/0 0.5/0.47 1/0.95'",
        }
        curve = mood_curves.get(mood, "curves=all='0/0 0.5/0.51 1/1'")
        vf = f"eq=contrast=1.05:saturation=1.06,{curve},unsharp=lx=3:ly=3:la=0.3,vignette=angle=0.15"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-i",
            input_path,
            "-vf",
            vf,
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "14",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        return self._run(args, timeout=600)

    def _validate_clip_job(self, job: ClipRenderJob) -> tuple[bool, str]:
        if not job.input_path or not os.path.exists(job.input_path):
            return False, f"Input not found: {job.input_path}"
        if job.trim_dur <= 0:
            return False, f"Invalid trim duration: {job.trim_dur}"
        if job.trim_start < 0:
            return False, f"Invalid trim start: {job.trim_start}"
        if job.width < 16 or job.height < 16:
            return False, f"Invalid output size: {job.width}x{job.height}"
        if job.fps <= 0 or job.fps > 240:
            return False, f"Invalid FPS: {job.fps}"
        probe = self.probe_video(job.input_path)
        if probe.get("duration", 0.0) <= 0:
            return False, "Input appears unreadable or has no duration"
        return True, ""

    def _validate_transition_job(self, job: TransitionJob) -> tuple[bool, str]:
        if not os.path.exists(job.clip_a):
            return False, f"First clip not found: {job.clip_a}"
        if not os.path.exists(job.clip_b):
            return False, f"Second clip not found: {job.clip_b}"
        if job.duration_sec <= 0:
            return False, f"Invalid transition duration: {job.duration_sec}"
        if job.fps <= 0 or job.fps > 240:
            return False, f"Invalid FPS: {job.fps}"
        return True, ""

    def _probe_duration(self, path: str) -> float:
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return float((result.stdout or "").strip())
        except Exception:
            return 5.0

    def probe_video(self, path: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,duration",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                return {"width": 1280, "height": 720, "fps": 30.0, "duration": 0.0}
            data = json.loads(result.stdout or "{}")
            stream = data.get("streams", [{}])[0]
            fps = parse_fps(stream.get("r_frame_rate", "30/1"))
            return {
                "width": int(stream.get("width", 1280) or 1280),
                "height": int(stream.get("height", 720) or 720),
                "fps": fps,
                "duration": float(stream.get("duration", 5.0) or 5.0),
            }
        except Exception:
            return {"width": 1280, "height": 720, "fps": 30.0, "duration": 0.0}

    def _probe_has_audio(self, path: str) -> bool:
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0 and "audio" in (result.stdout or "").lower()
        except Exception:
            return False
