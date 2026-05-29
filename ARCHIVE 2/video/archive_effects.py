# =============================================================================
# ARCHIVE HEAVY EFFECTS ENGINE  —  archive_effects.py
# Focused FFmpeg-based visual effects for strong final polish.
# =============================================================================

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def _log(msg: str) -> None:
    print(f"[EFFECTS] {msg}", flush=True)


# ===================== PRO COMPOSITOR SYSTEM =====================

class CharacterIsolator:
    """Isolates characters from backgrounds with temporal smoothing and light wrap effects."""
    
    def __init__(self):
        self.prev_mask = None

    def get_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Smooth mask over time
        if self.prev_mask is not None:
            mask = cv2.addWeighted(mask, 0.6, self.prev_mask, 0.4, 0)

        # Clean edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (9, 9), 2)

        self.prev_mask = mask.copy()
        return mask

    def light_wrap(self, char, bg, mask):
        blur_bg = cv2.GaussianBlur(bg, (0, 0), 15)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
        edge = edge.astype(np.float32) / 255.0
        wrap = blur_bg.astype(np.float32) * edge[:, :, None] * 0.25
        return np.clip(char.astype(np.float32) + wrap, 0, 255).astype(np.uint8)

    def composite(self, char, bg, mask):
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, None]

        result = char * alpha + bg * (1 - alpha)
        result = self.light_wrap(result.astype(np.uint8), bg, mask)
        return result.astype(np.uint8)


class BackgroundSystem:
    """Generates procedural gradient backgrounds with subtle variations."""
    
    def __init__(self, bg_count: int = 6):
        self.bgs = self.generate_bgs(bg_count)

    def generate_bgs(self, n: int):
        bgs = []
        h, w = 720, 1280
        for _ in range(n):
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)

            grad = (np.sin(xx * 3) + np.cos(yy * 2)) * 0.5 + 0.5

            c1 = np.random.randint(0, 80, 3)
            c2 = np.random.randint(120, 255, 3)

            bg = c1 * (1 - grad[..., None]) + c2 * grad[..., None]
            bg = bg.astype(np.uint8)

            noise = np.random.normal(0, 6, (h, w, 3)).astype(np.uint8)
            bg = cv2.add(bg, noise)

            bg = cv2.GaussianBlur(bg, (21, 21), 8)
            bgs.append(bg)
        return bgs

    def get_bg(self, frame_idx: int, fps: int):
        return self.bgs[(frame_idx // (fps * 2)) % len(self.bgs)]


class Camera:
    """Applies dynamic camera transforms: zoom, jitter, and panning."""
    
    def transform(self, frame, t: float):
        h, w = frame.shape[:2]

        zoom = 1 + 0.05 * math.sin(t * 2)
        jitter = math.sin(t * 12) * 0.002
        zoom += jitter

        nw, nh = int(w * zoom), int(h * zoom)
        frame = cv2.resize(frame, (nw, nh))

        x = (nw - w) // 2
        y = (nh - h) // 2

        return frame[y : y + h, x : x + w]


def add_glow(frame):
    """Adds a bloom/glow effect to enhance brightness."""
    blur = cv2.GaussianBlur(frame, (0, 0), 8)
    return cv2.addWeighted(frame, 0.8, blur, 0.2, 0)


class ProCompositor:
    """Professional video compositor with character isolation, backgrounds, and effects."""
    
    def __init__(self, bg_count: int = 6):
        self.iso = CharacterIsolator()
        self.bg = BackgroundSystem(bg_count)
        self.cam = Camera()

    def process(self, frame, frame_idx: int, fps: int):
        """Process a single frame with all compositor effects."""
        t = frame_idx / fps

        mask = self.iso.get_mask(frame)
        bg = self.bg.get_bg(frame_idx, fps)

        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        bg = self.cam.transform(bg, t)
        char = self.cam.transform(frame, t)

        comp = self.iso.composite(char, bg, mask)

        comp = add_glow(comp)

        # final color grade
        comp = np.clip((comp - 128) * 1.2 + 128, 0, 255).astype(np.uint8)

        return comp

    def compose_video(self, input_path: str, output_path: str) -> dict[str, Any]:
        """Compose entire video with professional effects."""
        t0 = time.perf_counter()
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                return {"ok": False, "error": f"Input file not found: {input_path}"}

            cap = cv2.VideoCapture(str(input_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            tmp = Path(tempfile.mkdtemp())

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                out = self.process(frame, frame_idx, fps)
                cv2.imwrite(str(tmp / f"{frame_idx:06d}.png"), out)
                frame_idx += 1

            cap.release()
            total_frames = frame_idx

            # Reconstruct video with ffmpeg
            cmd = [
                FFMPEG, "-y", "-framerate", str(fps),
                "-i", str(tmp / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                str(output_path)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
            ok = r.returncode == 0 and output_path.exists()

            # Cleanup
            shutil.rmtree(tmp)

            return {
                "ok": ok,
                "output": str(output_path),
                "frames_processed": total_frames,
                "fps": fps,
                "render_time_sec": round(time.perf_counter() - t0, 2),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "render_time_sec": round(time.perf_counter() - t0, 2)}


# ===================== COMPOSITOR HELPER =====================

def composite_video(input_path: str, output_path: str, bg_count: int = 6) -> dict[str, Any]:
    """Convenience function to composite a video with professional effects."""
    compositor = ProCompositor(bg_count=bg_count)
    return compositor.compose_video(input_path, output_path)


# ===================== END PRO COMPOSITOR =====================


def probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path)
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=12, check=False)
        data = json.loads(r.stdout)
        vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        fps_raw = vs.get("r_frame_rate", "30/1")
        try:
            n, d = fps_raw.split("/")
            fps = float(n) / max(1.0, float(d))
        except Exception:
            fps = 30.0
        dur = float(data.get("format", {}).get("duration", 0) or vs.get("duration", 30.0))
        return {
            "width": int(vs.get("width", 1280)),
            "height": int(vs.get("height", 720)),
            "fps": round(fps, 3),
            "duration": round(dur, 3),
            "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        }
    except Exception:
        return {"width": 1280, "height": 720, "fps": 30.0, "duration": 30.0, "has_audio": True}


@dataclass
class EffectParams:
    intensity: float = 0.8


class FilterBuilder:
    @staticmethod
    def teal_orange(intensity: float) -> str:
        sat = 1.15 + (0.2 * intensity)
        cont = 1.08 + (0.15 * intensity)
        return f"eq=contrast={cont:.3f}:saturation={sat:.3f}:gamma=0.98,colorbalance=rs=0.04:bs=-0.03"

    @staticmethod
    def cyberpunk(intensity: float) -> str:
        sat = 1.2 + (0.25 * intensity)
        return f"eq=saturation={sat:.3f}:contrast=1.08,colorbalance=rs=0.04:bs=0.06:rh=-0.03:bh=0.05"

    @staticmethod
    def anime_vibrant(intensity: float) -> str:
        sat = 1.18 + (0.2 * intensity)
        return f"eq=saturation={sat:.3f}:contrast=1.06:gamma=0.98"

    @staticmethod
    def bloom(intensity: float) -> str:
        blur = 10 + int(10 * intensity)
        amt = 0.35 + (0.25 * intensity)
        return (
            "split[base][glow];"
            f"[glow]curves=all='0/0 0.7/0.9 1/1',gblur=sigma={blur}[gb];"
            f"[base][gb]blend=all_mode=screen:all_opacity={amt:.3f}"
        )

    @staticmethod
    def rgb_split(intensity: float) -> str:
        px = 3 + int(4 * intensity)
        return (
            "split=3[r][g][b];"
            f"[r]chromashift=cbh={px}:crh=-{px}[rs];"
            f"[b]chromashift=cbh=-{px}:crh={px}[bs];"
            "[rs][g][bs]mix=inputs=3"
        )

    @staticmethod
    def vignette(intensity: float) -> str:
        angle = math.pi / max(0.8, (1.0 - intensity * 0.7) * 6 + 1.2)
        return f"vignette=angle={angle:.4f}"

    @staticmethod
    def film_grain(intensity: float) -> str:
        n = int(6 + 10 * intensity)
        return f"noise=alls={n}:allf=t"


class EffectEngine:
    PRESETS: dict[str, list[str]] = {
        "hype_phonk": ["teal_orange", "bloom", "rgb_split", "vignette", "film_grain"],
        "dark_villain": ["cyberpunk", "rgb_split", "vignette", "film_grain"],
        "anime_cinematic": ["anime_vibrant", "bloom", "vignette", "film_grain"],
    }

    ALL_EFFECTS = [
        "teal_orange",
        "cyberpunk",
        "anime_vibrant",
        "bloom",
        "rgb_split",
        "vignette",
        "film_grain",
    ]

    def apply(
        self,
        input_path: Path,
        output_path: Path,
        effects: list[str],
        intensity: float = 0.8,
        preserve_audio: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not input_path.exists():
            return {"ok": False, "error": f"Input not found: {input_path}"}

        effects = [e for e in effects if e in self.ALL_EFFECTS]
        if not effects:
            shutil.copy2(input_path, output_path)
            return {"ok": True, "effect_count": 0, "render_time_sec": 0.0}

        filters: list[str] = []
        for eff in effects:
            if eff == "teal_orange":
                filters.append(FilterBuilder.teal_orange(intensity))
            elif eff == "cyberpunk":
                filters.append(FilterBuilder.cyberpunk(intensity))
            elif eff == "anime_vibrant":
                filters.append(FilterBuilder.anime_vibrant(intensity))
            elif eff == "bloom":
                filters.append(FilterBuilder.bloom(intensity))
            elif eff == "rgb_split":
                filters.append(FilterBuilder.rgb_split(intensity))
            elif eff == "vignette":
                filters.append(FilterBuilder.vignette(intensity))
            elif eff == "film_grain":
                filters.append(FilterBuilder.film_grain(intensity))

        vf = ",".join(filters)
        info = probe_video(input_path)
        audio_flags = ["-c:a", "copy"] if preserve_audio and info["has_audio"] else ["-an"]

        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            *audio_flags,
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
            ok = r.returncode == 0 and output_path.exists()
        except Exception:
            ok = False

        return {
            "ok": ok,
            "effect_count": len(effects),
            "effects_applied": effects,
            "render_time_sec": round(time.perf_counter() - t0, 2),
        }


def get_effects_for_style(style_profile: dict, mood: str, intensity_level: int = 3) -> list[str]:
    effects: list[str] = []
    if style_profile.get("color_grade", False):
        effects.append("teal_orange" if mood in {"dramatic", "hype"} else "anime_vibrant")
    if style_profile.get("glow", False) or intensity_level >= 2:
        effects.append("bloom")
    if style_profile.get("glitch", False) or intensity_level >= 3:
        effects.append("rgb_split")
    if style_profile.get("vignette", False) or intensity_level >= 2:
        effects.append("vignette")
    if style_profile.get("grain", False) or intensity_level >= 2:
        effects.append("film_grain")
    return list(dict.fromkeys(effects))


def apply_effects_to_clip(
    input_path: Path,
    output_path: Path,
    effects: list[str],
    intensity: float = 0.8,
    style_profile: dict | None = None,
) -> dict[str, Any]:
    engine = EffectEngine()
    return engine.apply(input_path, output_path, effects, intensity=intensity, preserve_audio=True)
