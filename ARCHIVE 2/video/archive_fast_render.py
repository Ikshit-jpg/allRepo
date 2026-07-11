from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
CORES = max(2, min(16, os.cpu_count() or 4))

UNSUPPORTED_STYLE_KEYS = {
    "overlap_transitions",
    "mask_transition",
    "loop_clips",
    "loop_ending",
    "particles",
    "energy_aura",
    "turbulent",
    "blend_overlay",
    "mask_roto",
    "chroma_key",
    "difference_matte",
    "object_inpaint",
    "fire_lines",
    "boomerang",
    "pro_edit_pack",
}

GLOBAL_EFFECT_STYLE_KEYS = {
    "black_white",
    "chroma_pulse",
    "color_grade",
    "edge_glow",
    "fast_motion",
    "flash",
    "glitch",
    "glow",
    "grain",
    "motion_blur",
    "reverse",
    "shake",
    "slow_motion",
    "speed_lines",
    "speed_ramp",
    "vignette",
    "zoom",
}


def _log(msg: str) -> None:
    print(f"[FAST] {msg}", flush=True)


def _run(cmd: list[str], timeout: int = 900) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        err = (proc.stderr or "").strip()
        return proc.returncode == 0, err[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)


def _probe(path: str | Path) -> dict[str, Any]:
    cmd = [
        FFPROBE,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        data = json.loads(proc.stdout or "{}")
        video_stream = next(
            (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
            {},
        )
        fps_raw = str(video_stream.get("r_frame_rate") or "30/1")
        num, den = fps_raw.split("/")
        fps = float(num) / max(1.0, float(den))
        duration = float(data.get("format", {}).get("duration", 0) or video_stream.get("duration", 0) or 0)
        return {
            "w": int(video_stream.get("width", 1280)),
            "h": int(video_stream.get("height", 720)),
            "fps": round(fps or 30.0, 3),
            "dur": round(duration or 30.0, 3),
            "has_audio": any(stream.get("codec_type") == "audio" for stream in data.get("streams", [])),
        }
    except Exception:
        return {"w": 1280, "h": 720, "fps": 30.0, "dur": 30.0, "has_audio": True}


def is_fast_render_compatible(style: dict[str, Any] | None, has_music: bool) -> bool:
    if not has_music:
        return False
    profile = style or {}
    explicit_effects = [str(key) for key in profile.get("explicit_effects", []) or []]
    if explicit_effects:
        return False
    if int(profile.get("intensity", 1) or 1) >= 2 and not bool(profile.get("simple_mode", False)):
        return False
    blocked_keys = UNSUPPORTED_STYLE_KEYS | GLOBAL_EFFECT_STYLE_KEYS
    return not any(bool(profile.get(key, False)) for key in blocked_keys)


def _mood_grade(mood: str, intensity: int) -> dict[str, Any]:
    grades = {
        "hype": {
            "contrast": 1.22,
            "sat": 1.42,
            "bright": 0.025,
            "gamma": 0.95,
            "curves": "curves=r='0/0 0.5/0.56 1/1':b='0/0 0.5/0.44 1/0.92'",
        },
        "dark": {
            "contrast": 1.35,
            "sat": 0.72,
            "bright": -0.04,
            "gamma": 1.05,
            "curves": "curves=r='0/0 0.4/0.32 1/0.88':b='0/0 0.4/0.48 1/1.0'",
        },
        "sad": {
            "contrast": 1.12,
            "sat": 0.68,
            "bright": -0.02,
            "gamma": 1.02,
            "curves": "curves=b='0/0.05 0.5/0.55 1/1.0':r='0/0 0.5/0.46 1/0.92'",
        },
        "dramatic": {
            "contrast": 1.28,
            "sat": 1.32,
            "bright": 0.01,
            "gamma": 0.97,
            "curves": "curves=r='0/0 0.5/0.54 1/1.0':b='0/0 0.5/0.52 1/1.0'",
        },
        "chill": {
            "contrast": 1.10,
            "sat": 1.18,
            "bright": 0.03,
            "gamma": 0.98,
            "curves": "curves=r='0/0.02 0.5/0.52 1/1.0'",
        },
    }
    result = dict(grades.get(mood, grades["hype"]))
    scale = {1: 0.55, 2: 0.80, 3: 1.0}.get(intensity, 1.0)
    result["contrast"] = 1.0 + (result["contrast"] - 1.0) * scale
    result["sat"] = 1.0 + (result["sat"] - 1.0) * scale
    result["bright"] = result["bright"] * scale
    return result


def build_ffmpeg_effect_chain(style: dict[str, Any], mood: str, intensity: int, info: dict[str, Any]) -> str:
    filters: list[str] = []
    w = int(info["w"])
    h = int(info["h"])
    fps = float(info["fps"])
    no_color = bool(style.get("no_color_grade", False))
    bw = bool(style.get("black_white", False))

    if not no_color and not bw:
        grade = _mood_grade(mood, intensity)
        filters.append(
            f"eq=contrast={grade['contrast']:.3f}:saturation={grade['sat']:.3f}:"
            f"brightness={grade['bright']:.4f}:gamma={grade['gamma']:.3f}"
        )
        curves = str(grade.get("curves") or "")
        if curves:
            filters.append(curves)

    if bw:
        filters.extend(["hue=s=0", "eq=contrast=1.18:brightness=-0.02"])
    if bool(style.get("vignette", False)) and not no_color:
        filters.append("vignette=PI/5")
    if bool(style.get("grain", False)):
        strength = int(max(8, min(28, 12 + intensity * 5)))
        filters.append(f"noise=alls={strength}:allf=t+u")
    if bool(style.get("chroma_pulse", False) or style.get("glitch", False)):
        px = int(max(2, min(10, 2 + intensity * 2)))
        filters.append(f"rgbashift=rh={px}:bh=-{px}")
    if bool(style.get("motion_blur", False)):
        frames = max(3, min(5, 2 + intensity))
        weights = " ".join(f"{(i + 1) / (frames * (frames + 1) / 2):.3f}" for i in range(frames))
        filters.append(f"tmix=frames={frames}:weights='{weights}'")
    if bool(style.get("glow", False)) and not no_color:
        amount = 0.45 + intensity * 0.18
        filters.append(
            "unsharp="
            f"luma_msize_x=7:luma_msize_y=7:luma_amount={amount:.2f}:"
            f"chroma_msize_x=7:chroma_msize_y=7:chroma_amount={amount * 0.5:.2f}"
        )
    if bool(style.get("shake", False)):
        shake_px = int(max(4, min(14, 4 + intensity * 3)))
        freq = 8 + intensity * 2
        filters.append(
            f"crop=iw-{shake_px * 2}:ih-{shake_px * 2}:"
            f"{shake_px}+{max(1, shake_px // 2)}*sin(n/{fps:.3f}*{freq}*2*PI):{shake_px}"
        )
        filters.append(f"scale={w}:{h}")
    if bool(style.get("zoom", False)):
        zoom_factor = 1.0 + intensity * 0.03
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        cx = max(0, (w - crop_w) // 2)
        cy = max(0, (h - crop_h) // 2)
        filters.extend([f"crop={crop_w}:{crop_h}:{cx}:{cy}", f"scale={w}:{h}"])
    if bool(style.get("edge_glow", False)):
        filters.append("unsharp=5:5:0.8:3:3:0.35")
    if bool(style.get("posterize_time", False)):
        filters.extend(["fps=15", f"fps={max(24, round(fps))}"])
    if bool(style.get("vhs", False)):
        filters.extend(
            [
                "noise=alls=10:allf=t",
                "rgbashift=rh=3:bh=-3",
                "eq=contrast=1.04:saturation=0.88",
                "drawgrid=width=0:height=4:thickness=1:color=black@0.16",
            ]
        )
    return ",".join(filters)


def _build_cut_points(
    beats: list[float],
    duration: float,
    mood: str,
    intensity: int,
    selective: bool = True,
    all_beats: bool = False,
) -> list[float]:
    if not beats:
        interval = {"hype": 0.8, "dark": 1.2, "sad": 2.0, "dramatic": 1.5, "chill": 2.0}.get(mood, 1.2)
        points = [0.0]
        t = interval
        while t < duration:
            points.append(round(t, 3))
            t += interval
        points.append(duration)
        return points

    step = {"hype": 2, "dark": 2, "dramatic": 2, "sad": 3, "chill": 3}.get(mood, 2)
    step = max(1, step - (1 if intensity >= 2 else 0))
    if intensity >= 3 and duration <= 20:
        step = 1
    if selective and not all_beats:
        step += 1

    points = [0.0]
    for idx, beat in enumerate(beats):
        if 0 < beat < duration and idx % step == 0:
            points.append(float(beat))
    points.append(duration)
    points = sorted(set(points))

    cleaned = [points[0]]
    for point in points[1:]:
        if point - cleaned[-1] >= 0.15:
            cleaned.append(point)
    if cleaned[-1] < duration:
        cleaned.append(duration)
    return cleaned


def fast_cut_segments(
    input_path: str | Path,
    cut_points: list[float],
    output_dir: Path,
    style: dict[str, Any],
    info: dict[str, Any],
) -> list[Path]:
    segments: list[Path] = []
    for idx in range(max(0, len(cut_points) - 1)):
        start = float(cut_points[idx])
        end = float(cut_points[idx + 1])
        duration = end - start
        if duration < 0.08:
            continue

        speed = 1.0
        if bool(style.get("slow_motion", False)) and duration > 0.8:
            speed = 0.5
        elif bool(style.get("fast_motion", False)):
            speed = 1.5
        reverse = bool(style.get("reverse", False))
        segment_path = output_dir / f"seg_{idx:04d}.mp4"

        if speed == 1.0 and not reverse:
            cmd = [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.4f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.4f}",
                "-an",
                "-c:v",
                "copy",
                str(segment_path),
            ]
        else:
            video_filters: list[str] = []
            if reverse:
                video_filters.append("reverse")
            if speed != 1.0:
                video_filters.append(f"setpts={1.0 / speed:.4f}*PTS")
            cmd = [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.4f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.4f}",
                "-an",
                "-vf",
                ",".join(video_filters),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(segment_path),
            ]

        ok, err = _run(cmd, timeout=1200)
        if ok and segment_path.exists() and segment_path.stat().st_size > 0:
            segments.append(segment_path)
        else:
            _log(f"segment {idx} failed: {err[:120]}")
    return segments


def style_wants_flash(mood: str, intensity: int) -> bool:
    return mood in {"hype", "dramatic"} and intensity >= 2


def inject_flash_frames(
    segment_paths: list[Path],
    output_dir: Path,
    mood: str,
    intensity: int,
    info: dict[str, Any],
) -> list[Path]:
    if not style_wants_flash(mood, intensity):
        return segment_paths

    fps = float(info["fps"])
    w = int(info["w"])
    h = int(info["h"])
    flash_duration = max(0.04, 0.08 - intensity * 0.01)
    colors = {
        "hype": "white",
        "dark": "0x1a0030",
        "sad": "0xd0e0ff",
        "dramatic": "white",
        "chill": "0xe8f0f0",
    }
    flash_path = output_dir / f"flash_{uuid.uuid4().hex[:8]}.mp4"
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={colors.get(mood, 'white')}:size={w}x{h}:rate={fps:.3f}:duration={flash_duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(flash_path),
    ]
    ok, _ = _run(cmd, timeout=300)
    if not ok or not flash_path.exists():
        return segment_paths

    interval = max(2, 5 - intensity)
    result: list[Path] = []
    for idx, seg in enumerate(segment_paths):
        result.append(seg)
        if idx < len(segment_paths) - 1 and idx % interval == 0:
            result.append(flash_path)
    return result


def concat_with_effects(
    segment_paths: list[Path],
    output_path: str | Path,
    music_path: str | Path | None,
    music_start: float,
    style: dict[str, Any],
    mood: str,
    intensity: int,
    info: dict[str, Any],
    crf: int = 18,
) -> bool:
    if not segment_paths:
        return False

    output = Path(output_path)
    concat_list = output.parent / f"_concat_{uuid.uuid4().hex[:8]}.txt"
    with open(concat_list, "w", encoding="utf-8") as handle:
        for seg in segment_paths:
            escaped = str(seg).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    effect_chain = build_ffmpeg_effect_chain(style, mood, intensity, info)
    cmd: list[str] = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
    ]

    has_music = bool(music_path) and Path(str(music_path)).exists()
    if has_music:
        cmd += ["-ss", f"{music_start:.4f}", "-i", str(music_path)]
    if effect_chain:
        cmd += ["-vf", effect_chain]

    cmd += ["-map", "0:v:0"]
    if has_music:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        str(CORES),
        str(output),
    ]

    ok, err = _run(cmd, timeout=1800)
    try:
        concat_list.unlink(missing_ok=True)
    except Exception:
        pass
    if not ok:
        _log(f"concat/effects failed: {err[:200]}")
    return ok and output.exists() and output.stat().st_size > 0


def _minimal_fallback(
    video_path: Path,
    output_path: Path,
    music_path: str | Path | None,
    music_start: float,
) -> bool:
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
    if music_path and Path(str(music_path)).exists():
        cmd += [
            "-ss",
            f"{music_start:.4f}",
            "-i",
            str(music_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]
    else:
        cmd += ["-c:v", "copy"]
    cmd.append(str(output_path))
    ok, _ = _run(cmd, timeout=300)
    return ok and output_path.exists()


def fast_render(
    video_path: str | Path,
    output_path: str | Path,
    beat_markers: list[float],
    mood: str,
    music_path: str | Path | None = None,
    music_start: float = 0.0,
    prompt_text: str = "",
    style: dict[str, Any] | None = None,
    intensity: int = 3,
    crf: int = 18,
    drop_times: list[float] | None = None,
) -> float:
    del prompt_text, drop_times
    video_path = Path(video_path)
    output_path = Path(output_path)
    style = style or {}

    if not is_fast_render_compatible(style, bool(music_path)):
        return 0.0

    t0 = time.time()
    info = _probe(video_path)
    cut_points = _build_cut_points(
        beat_markers,
        float(info["dur"]),
        mood,
        intensity,
        selective=bool(style.get("selective_beats", True)),
        all_beats=bool(style.get("all_beats", False)),
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="archive_fast_"))
    try:
        segments = fast_cut_segments(video_path, cut_points, tmp_dir, style, info)
        if not segments:
            return 0.0
        if bool(style.get("flash", False)) or mood in {"hype", "dramatic"}:
            segments = inject_flash_frames(segments, tmp_dir, mood, intensity, info)
        ok = concat_with_effects(
            segments,
            output_path,
            music_path,
            music_start,
            style,
            mood,
            intensity,
            info,
            crf=crf,
        )
        if not ok:
            ok = _minimal_fallback(video_path, output_path, music_path, music_start)
        if not ok:
            return 0.0
        result = _probe(output_path)
        _log(f"fast render finished in {time.time() - t0:.1f}s")
        return float(result.get("dur") or info["dur"] or 0.0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def post_process_with_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    style: dict[str, Any],
    mood: str,
    intensity: int = 3,
    crf: int = 18,
) -> bool:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        return False

    info = _probe(input_path)
    effect_chain = build_ffmpeg_effect_chain(style, mood, intensity, info)
    if not effect_chain:
        shutil.copy2(input_path, output_path)
        return True

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        effect_chain,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-threads",
        str(CORES),
        str(output_path),
    ]
    ok, _ = _run(cmd, timeout=1800)
    return ok and output_path.exists() and output_path.stat().st_size > 0
