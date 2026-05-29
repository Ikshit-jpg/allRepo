from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from engine import ClipRenderJob, FFmpegEngine
from engine.graph.operations import RenderOperation

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
CORES = max(2, min(8, os.cpu_count() or 4))


def _log(msg: str, tag: str = "BRAIN") -> None:
    print(f"[{tag}] {msg}", flush=True)


def _run(cmd: list[str], timeout: int = 600) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode == 0, (proc.stderr or "")[-500:]
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        data = json.loads(proc.stdout or "{}")
        vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        fps_raw = str(vs.get("r_frame_rate") or "30/1")
        num, den = fps_raw.split("/")
        fps = float(num) / max(1.0, float(den))
        duration = float(data.get("format", {}).get("duration", 0) or vs.get("duration", 30) or 30)
        has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
        return {
            "w": int(vs.get("width", 1280)),
            "h": int(vs.get("height", 720)),
            "fps": round(fps or 30.0, 3),
            "dur": round(duration or 30.0, 3),
            "has_audio": has_audio,
        }
    except Exception:
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        dur = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 30.0) / max(1.0, fps)
        cap.release()
        return {"w": w, "h": h, "fps": fps, "dur": dur, "has_audio": False}


class AnalysisCache:
    """Disk cache for scene analysis and beat map results."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir or Path(tempfile.gettempdir()) / "archive_cache"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _file_sig(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return "missing"
        return f"{p.stat().st_size}_{int(p.stat().st_mtime)}"

    def _scene_key(self, video_path: str, start: float, end: float) -> str:
        raw = f"scene:{video_path}:{self._file_sig(video_path)}:{start:.4f}:{end:.4f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _beat_key(self, audio_path: str) -> str:
        raw = f"beat:{audio_path}:{self._file_sig(audio_path)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_scene(self, video_path: str, start: float, end: float) -> dict[str, Any] | None:
        key = self._scene_key(video_path, start, end)
        cache_file = self._dir / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def put_scene(self, video_path: str, start: float, end: float, data: dict[str, Any]) -> None:
        key = self._scene_key(video_path, start, end)
        cache_file = self._dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def get_beat_map(self, audio_path: str) -> dict[str, Any] | None:
        key = self._beat_key(audio_path)
        cache_file = self._dir / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def put_beat_map(self, audio_path: str, data: dict[str, Any]) -> None:
        key = self._beat_key(audio_path)
        cache_file = self._dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass


class EditorMemory:
    """Tracks global edit state to prevent effect/transition repetition."""

    EFFECT_FAMILIES = {
        "shake": {"shake", "camera_slam", "handheld_shake"},
        "glow": {"deep_glow", "glow", "bloom", "emotion_glow"},
        "chromatic": {"chromatic", "rgbashift", "chromatic_shake"},
        "flash": {"flash", "impact_flash_combo", "flash_cut"},
        "zoom": {"zoom", "zoom_pulse", "cross_zoom", "zoom_burst_shake"},
        "speed": {"speed_ramp", "slow_motion", "speed_lines", "speed_blur_lines"},
        "grade": {"color_grade", "cinematic_grade", "atmospheric_grade"},
        "grain": {"grain", "vhs", "noise"},
    }

    def __init__(self) -> None:
        self._effect_history: deque[str] = deque(maxlen=12)
        self._transition_history: deque[str] = deque(maxlen=8)
        self._family_counts: dict[str, int] = defaultdict(int)

    def _family_of(self, effect: str) -> str:
        for family, members in self.EFFECT_FAMILIES.items():
            if effect in members:
                return family
        return effect

    def filter_effects(self, candidates: list[str]) -> list[str]:
        """Remove effects whose family was used in the last 2 segments."""
        recent_families = {self._family_of(e) for e in list(self._effect_history)[-2:]}
        filtered = [c for c in candidates if self._family_of(c) not in recent_families]
        return filtered if filtered else candidates

    def filter_transitions(self, candidates: list[str]) -> list[str]:
        """Remove transitions used in the last 2 segments."""
        recent = set(list(self._transition_history)[-2:])
        filtered = [c for c in candidates if c not in recent]
        return filtered if filtered else candidates

    def record_effect(self, effect: str) -> None:
        self._effect_history.append(effect)
        self._family_counts[self._family_of(effect)] += 1

    def record_transition(self, transition: str) -> None:
        self._transition_history.append(transition)


@dataclass
class SceneRead:
    start: float = 0.0
    end: float = 1.0
    duration: float = 1.0
    scene_type: str = "action"
    motion: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    sharpness: float = 0.5
    has_face: bool = False
    face_size: float = 0.0
    face_pos_x: float = 0.5
    face_pos_y: float = 0.42
    subject_motion: float = 0.5
    emotional_intensity: float = 0.5
    color_mood: str = "neutral"
    is_climax: bool = False
    is_quiet: bool = False
    cinematic_quality: float = 0.5
    edit_value: float = 0.5
    impact_rank: float = 0.5
    motion_direction: str = "mixed"
    internal_impacts: list[float] = field(default_factory=list)
    needs_effects: list[str] = field(default_factory=list)
    rejects_effects: list[str] = field(default_factory=list)


class SceneUnderstandingEngine:
    def __init__(self) -> None:
        self._face_cascade = self._load_face_cascade()

    def _load_face_cascade(self) -> Any:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            return None if cascade.empty() else cascade
        return None

    def analyze_segment(
        self,
        video_path: str | Path,
        start: float,
        end: float,
        n_samples: int = 6,
    ) -> SceneRead:
        scene = SceneRead(start=start, end=end, duration=max(0.01, end - start))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return scene

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_times = np.linspace(start, max(start, end - 0.05), n_samples)
        frames: list[tuple[float, np.ndarray]] = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, t * fps)))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((float(t), frame))
        cap.release()
        if not frames:
            return scene

        scene.motion, scene.motion_direction = self._analyze_motion(frames)
        scene.subject_motion = scene.motion
        (
            scene.brightness,
            scene.saturation,
            scene.contrast,
            scene.sharpness,
            scene.color_mood,
        ) = self._analyze_color(frames)
        scene.has_face, scene.face_size, scene.face_pos_x, scene.face_pos_y = self._detect_face(frames)
        scene.scene_type = self._classify_scene(scene)
        scene.emotional_intensity = self._score_emotion(scene)
        scene.cinematic_quality = self._score_cinematic(scene)
        scene.internal_impacts = self._find_internal_impacts(frames)
        scene.is_climax = scene.emotional_intensity > 0.72 and scene.motion > 0.55
        scene.is_quiet = scene.motion < 0.12 and scene.emotional_intensity < 0.38
        scene.edit_value = self._score_edit_value(scene)
        scene.needs_effects, scene.rejects_effects = self._recommend_effects(scene)
        return scene

    def _analyze_motion(self, frames: list[tuple[float, np.ndarray]]) -> tuple[float, str]:
        if len(frames) < 2:
            return 0.2, "static"
        flows = []
        for idx in range(min(5, len(frames) - 1)):
            g1 = cv2.cvtColor(cv2.resize(frames[idx][1], (160, 90)), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(cv2.resize(frames[idx + 1][1], (160, 90)), cv2.COLOR_BGR2GRAY)
            try:
                flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 8, 2, 5, 1.1, 0)
                flows.append(flow)
            except Exception:
                continue
        if not flows:
            diff = cv2.absdiff(
                cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frames[-1][1], cv2.COLOR_BGR2GRAY),
            )
            return float(np.clip(diff.mean() / 40.0, 0, 1)), "mixed"

        mags = [np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean() for flow in flows]
        motion = float(np.clip(np.mean(mags) / 6.0, 0, 1))
        dx = float(np.mean([flow[..., 0].mean() for flow in flows]))
        dy = float(np.mean([flow[..., 1].mean() for flow in flows]))
        adx = abs(dx)
        ady = abs(dy)
        if motion < 0.05:
            direction = "static"
        elif adx > ady * 1.8:
            direction = "right" if dx > 0 else "left"
        elif ady > adx * 1.8:
            direction = "down" if dy > 0 else "up"
        else:
            direction = "mixed"
        return motion, direction

    def _analyze_color(self, frames: list[tuple[float, np.ndarray]]) -> tuple[float, float, float, float, str]:
        brightness_values = []
        saturation_values = []
        contrast_values = []
        sharpness_values = []
        for _, frame in frames[:5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            brightness_values.append(float(gray.mean()) / 255.0)
            saturation_values.append(float(hsv[:, :, 1].mean()) / 255.0)
            contrast_values.append(float(gray.std()) / 128.0)
            sharpness_values.append(float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 0, 1)))
        bri = float(np.mean(brightness_values))
        sat = float(np.mean(saturation_values))
        con = float(np.mean(contrast_values))
        sha = float(np.mean(sharpness_values))
        sample = frames[0][1].astype(np.float32)
        temperature = (float(sample[:, :, 2].mean()) - float(sample[:, :, 0].mean()) + 128.0) / 256.0
        if bri < 0.28:
            mood = "dark"
        elif sat > 0.55 and bri > 0.40:
            mood = "vibrant"
        elif temperature > 0.62 and sat > 0.25:
            mood = "warm"
        elif temperature < 0.38 and sat > 0.20:
            mood = "cold"
        elif bri > 0.72:
            mood = "bright"
        else:
            mood = "neutral"
        return bri, sat, con, sha, mood

    def _detect_face(self, frames: list[tuple[float, np.ndarray]]) -> tuple[bool, float, float, float]:
        if not self._face_cascade:
            return False, 0.0, 0.5, 0.42
        for _, frame in frames[:3]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180))
            try:
                faces = self._face_cascade.detectMultiScale(small, 1.1, 4, minSize=(18, 18))
            except Exception:
                faces = []
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                area = (w * h) / float(320 * 180)
                return True, float(area), float((x + w / 2) / 320.0), float((y + h / 2) / 180.0)
        return False, 0.0, 0.5, 0.42

    def _classify_scene(self, scene: SceneRead) -> str:
        if scene.motion > 0.70:
            return "impact"
        if scene.motion > 0.42 and scene.has_face and scene.face_size < 0.15:
            return "action"
        if scene.has_face and scene.motion < 0.35 and scene.face_size > 0.08:
            return "reaction" if scene.contrast > 0.35 or scene.saturation > 0.30 else "dialogue"
        if scene.motion < 0.10:
            return "scenery" if scene.sharpness > 0.5 else "quiet"
        if scene.motion > 0.25 and scene.contrast > 0.45:
            return "transition"
        return "action"

    def _score_emotion(self, scene: SceneRead) -> float:
        score = 0.0
        score += scene.motion * 0.25
        score += scene.contrast * 0.20
        score += scene.saturation * 0.15
        score += (1.0 - abs(scene.brightness - 0.45) * 1.5) * 0.15
        score += (scene.face_size * 3.0) * 0.15 if scene.has_face else 0.0
        if scene.color_mood in {"vibrant", "warm"}:
            score += 0.10
        if scene.scene_type in {"impact", "reaction"}:
            score += 0.12
        return float(np.clip(score, 0, 1))

    def _score_cinematic(self, scene: SceneRead) -> float:
        score = scene.sharpness * 0.35
        score += (1.0 - abs(scene.brightness - 0.48) * 1.8) * 0.25
        score += scene.contrast * 0.20
        if scene.has_face and 0.15 < scene.face_pos_x < 0.85:
            score += 0.20 * min(1.0, scene.face_size * 5.0)
        return float(np.clip(score, 0, 1))

    def _find_internal_impacts(self, frames: list[tuple[float, np.ndarray]]) -> list[float]:
        impacts: list[float] = []
        for idx in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[idx - 1][1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[idx][1], cv2.COLOR_BGR2GRAY)
            diff = float(cv2.absdiff(curr_gray, prev_gray).mean())
            if diff > 20:
                impacts.append(round(frames[idx][0], 3))
        return impacts

    def _score_edit_value(self, scene: SceneRead) -> float:
        score = scene.cinematic_quality * 0.30
        score += scene.emotional_intensity * 0.28
        if scene.scene_type in {"impact", "reaction"}:
            score += 0.22
        elif scene.scene_type == "action":
            score += 0.14
        elif scene.scene_type in {"dialogue", "quiet"}:
            score -= 0.10
        if scene.has_face and scene.face_size > 0.05:
            score += 0.12
        if scene.motion_direction in {"left", "right"}:
            score += 0.06
        if scene.is_climax:
            score += 0.15
        return float(np.clip(score, 0, 1))

    def _recommend_effects(self, scene: SceneRead) -> tuple[list[str], list[str]]:
        needs: list[str] = []
        rejects: list[str] = []
        if scene.scene_type == "impact":
            needs += ["flash", "shake", "zoom", "chromatic", "speed_lines"]
            rejects += ["slow_motion", "ken_burns"]
        elif scene.scene_type == "action":
            needs += ["motion_blur", "shake", "zoom", "speed_ramp"]
            rejects += ["slow_motion", "ken_burns", "vhs"]
        elif scene.scene_type == "reaction":
            needs += ["deep_glow", "slow_motion"]
            rejects += ["shake", "glitch", "chromatic", "speed_lines", "flash"]
        elif scene.scene_type == "scenery":
            needs += ["ken_burns", "deep_glow", "color_grade"]
            rejects += ["shake", "flash", "glitch", "speed_lines"]
        elif scene.scene_type == "dialogue":
            needs += ["color_grade", "vignette"]
            rejects += ["shake", "flash", "glitch", "chromatic", "speed_lines"]
        else:
            needs += ["color_grade", "grain"]
            rejects += ["flash", "shake", "glitch", "speed_lines"]
        needs += ["color_grade", "vignette"]
        if scene.color_mood == "dark":
            needs.append("grain")
        if scene.brightness < 0.25:
            rejects.append("vignette")
        return sorted(set(needs)), sorted(set(rejects))


class RestraintEngine:
    HEAVY_RATE = {"hype": 0.55, "dark": 0.45, "dramatic": 0.40, "sad": 0.25, "chill": 0.20}
    MAX_DRY_STREAK = {"hype": 3, "dark": 4, "dramatic": 4, "sad": 6, "chill": 8}

    def __init__(self, mood: str, total_segments: int) -> None:
        self.mood = mood
        self.total = total_segments
        self._since_last_big = 0

    def should_fire_heavy(
        self,
        scene: SceneRead,
        beat_energy: float,
        section: str,
        index: int,
        is_peak: bool,
    ) -> bool:
        rate = self.HEAVY_RATE.get(self.mood, 0.4)
        max_dry = self.MAX_DRY_STREAK.get(self.mood, 4)
        self._since_last_big += 1
        if is_peak and beat_energy > 0.80:
            return self._fire()
        if section == "drop" and beat_energy > 0.70:
            return self._fire()
        if self._since_last_big >= max_dry:
            return self._fire()
        # Keep openings alive: allow one signature opener instead of muting intro.
        if section == "intro" and index < 2:
            if self.mood in {"hype", "dark", "dramatic"} and (scene.edit_value > 0.58 or beat_energy > 0.62):
                return self._fire()
            if scene.scene_type in {"impact", "action"} and scene.edit_value > 0.50:
                return self._fire()
        if scene.is_quiet and section != "drop":
            return False
        if scene.scene_type == "dialogue":
            return False
        scene_prob = scene.edit_value * 0.40 + scene.emotional_intensity * 0.30 + beat_energy * 0.30
        return self._fire() if scene_prob >= (1.0 - rate) else False

    def _fire(self) -> bool:
        self._since_last_big = 0
        return True

    def _narrative_curve(self, position: float) -> float:
        """Continuous narrative arc curve: intro→build→drop→chorus→break→outro."""
        # Smooth tension curve based on position (0..1) in the edit
        control_points = [
            (0.00, 0.15), (0.10, 0.25), (0.20, 0.45), (0.30, 0.65),
            (0.38, 0.90), (0.45, 0.78), (0.55, 0.72), (0.62, 0.30),
            (0.70, 0.25), (0.78, 0.55), (0.85, 0.88), (0.92, 0.95),
            (0.97, 0.60), (1.00, 0.30),
        ]
        for i in range(len(control_points) - 1):
            x0, y0 = control_points[i]
            x1, y1 = control_points[i + 1]
            if x0 <= position <= x1:
                t = (position - x0) / max(1e-9, x1 - x0)
                t = t * t * (3 - 2 * t)  # smoothstep
                return y0 + (y1 - y0) * t
        return 0.5

    def get_continuous_intensity(
        self, scene: SceneRead, beat_energy: float, section: str, index: int, base_intensity: int,
    ) -> float:
        """Returns smooth 0.0-1.0 intensity driven by narrative arc, beat energy, and scene."""
        position = index / max(1, self.total - 1)
        arc_base = self._narrative_curve(position)
        # Beat energy modulation
        beat_mod = beat_energy * 0.25
        # Scene type modifier
        scene_mods = {
            "impact": 0.15, "action": 0.08, "reaction": -0.05,
            "scenery": -0.12, "dialogue": -0.20, "quiet": -0.25,
        }
        scene_mod = scene_mods.get(scene.scene_type, 0.0)
        # Section override for drops
        section_boost = {"drop": 0.18, "chorus": 0.08, "buildup": 0.04}.get(section, 0.0)
        # Mood scaling
        mood_scale = {"hype": 1.15, "dark": 1.05, "dramatic": 1.10, "sad": 0.80, "chill": 0.75}.get(self.mood, 1.0)
        # Base intensity tier
        tier_mod = {1: -0.12, 2: 0.0, 3: 0.12}.get(base_intensity, 0.0)
        raw = (arc_base + beat_mod + scene_mod + section_boost + tier_mod) * mood_scale
        return float(np.clip(raw, 0.08, 1.0))

    def get_effect_intensity(self, scene: SceneRead, base_intensity: int, section: str, heavy: bool) -> float:
        """Legacy interface kept for compatibility — still used in some paths."""
        if not heavy:
            return 0.15 + scene.edit_value * 0.25
        base = {1: 0.45, 2: 0.70, 3: 0.90}.get(base_intensity, 0.70)
        multipliers = {
            "impact": 1.20,
            "action": 1.00,
            "reaction": 0.75,
            "scenery": 0.65,
            "dialogue": 0.40,
            "quiet": 0.35,
        }
        section_mult = {
            "drop": 1.20,
            "chorus": 1.00,
            "buildup": 0.80,
            "break": 0.50,
            "intro": 0.68,
            "outro": 0.55,
        }
        return float(
            np.clip(
                base
                * multipliers.get(scene.scene_type, 0.85)
                * section_mult.get(section, 0.85),
                0.0,
                1.0,
            )
        )


@dataclass
class NarrativeSection:
    type: str
    start: float
    end: float
    energy: float
    intensity: float


class NarrativeArcEngine:
    def __init__(self, beat_map: dict[str, Any], mood: str) -> None:
        self.beat_map = beat_map
        self.mood = mood
        self.sections = self._build_sections()

    def _build_sections(self) -> list[NarrativeSection]:
        duration = float(self.beat_map.get("duration", 30.0))
        beats = list(self.beat_map.get("beats", []))
        energy = list(self.beat_map.get("energy_curve", []))
        drops = list(self.beat_map.get("drop_times", [duration * 0.4]))
        if not beats:
            return [
                NarrativeSection("intro", 0.0, duration * 0.12, 0.30, 0.30),
                NarrativeSection("buildup", duration * 0.12, duration * 0.38, 0.60, 0.55),
                NarrativeSection("drop", duration * 0.38, duration * 0.55, 1.00, 1.00),
                NarrativeSection("chorus", duration * 0.55, duration * 0.75, 0.85, 0.80),
                NarrativeSection("break", duration * 0.75, duration * 0.85, 0.35, 0.30),
                NarrativeSection("outro", duration * 0.85, duration, 0.40, 0.35),
            ]
        values = energy[: len(beats)] if len(energy) >= len(beats) else [0.5] * len(beats)
        hi = float(np.quantile(values, 0.75)) if values else 0.7
        lo = float(np.quantile(values, 0.30)) if values else 0.3
        sections: list[NarrativeSection] = []
        current_type = "intro"
        section_start = 0.0
        last_energy = 0.5
        bar = 4
        idx = 0
        while idx < len(beats):
            beat_time = beats[idx]
            avg_energy = float(np.mean(values[idx : min(idx + bar, len(values))]))
            at_drop = any(abs(beat_time - drop) < 1.0 for drop in drops)
            past_drop = any(0 < beat_time - drop < 6.0 for drop in drops)
            if beat_time / duration < 0.12:
                section_type = "intro"
            elif at_drop and avg_energy > hi * 0.85:
                section_type = "drop"
            elif past_drop and avg_energy > lo * 1.1:
                section_type = "chorus"
            elif avg_energy < lo and beat_time / duration > 0.15:
                section_type = "break"
            elif avg_energy > last_energy and not past_drop and beat_time / duration > 0.12:
                section_type = "buildup"
            elif beat_time / duration > 0.85:
                section_type = "outro"
            else:
                section_type = current_type
            if section_type != current_type:
                sections.append(NarrativeSection(current_type, section_start, beat_time, last_energy, last_energy))
                current_type = section_type
                section_start = beat_time
            last_energy = avg_energy
            idx += bar
        sections.append(NarrativeSection(current_type, section_start, duration, last_energy, last_energy))
        return sections

    def section_at(self, t: float) -> NarrativeSection:
        for section in self.sections:
            if section.start <= t <= section.end:
                return section
        return self.sections[-1] if self.sections else NarrativeSection("unknown", 0.0, 30.0, 0.5, 0.5)

    def plan_clip_assignment(self, scenes: list[SceneRead]) -> list[tuple[SceneRead, NarrativeSection]]:
        scored = sorted(scenes, key=lambda item: item.edit_value, reverse=True)
        if not scored:
            return []
        buckets: dict[str, list[SceneRead]] = defaultdict(list)
        n = len(scored)
        top = scored[: max(1, n // 4)]
        mid = scored[n // 4 : n // 2]
        low = scored[n // 2 :]
        for scene in top:
            buckets["drop"].append(scene)
            buckets["chorus"].append(scene)
        for scene in mid:
            buckets["buildup"].append(scene)
            buckets["chorus"].append(scene)
        for scene in low:
            buckets["intro"].append(scene)
            buckets["break"].append(scene)
            buckets["outro"].append(scene)
        usage_count: dict[int, int] = defaultdict(int)
        last_used_at: dict[int, int] = defaultdict(lambda: -99)
        last_scene_key: int | None = None
        result: list[tuple[SceneRead, NarrativeSection]] = []
        MAX_REUSE = 2  # reduced from 3 — less visible repetition
        MIN_REUSE_GAP = 3  # at least 3 segments between reuses
        for section in self.sections:
            pool = buckets.get(section.type) or scored
            section_duration = section.end - section.start
            used = 0.0
            while used < section_duration and pool:
                seg_idx = len(result)
                ranked_pool = sorted(pool, key=lambda sc: (usage_count[id(sc)], -float(sc.edit_value)))
                scene = ranked_pool[0]
                # Avoid back-to-back same scene
                if last_scene_key is not None and id(scene) == last_scene_key and len(ranked_pool) > 1:
                    scene = ranked_pool[1]
                # Enforce minimum gap between reuses of same scene
                if usage_count[id(scene)] > 0 and (seg_idx - last_used_at[id(scene)]) < MIN_REUSE_GAP and len(ranked_pool) > 1:
                    for alt in ranked_pool[1:]:
                        if (seg_idx - last_used_at[id(alt)]) >= MIN_REUSE_GAP:
                            scene = alt
                            break
                used += max(0.3, min(scene.duration, section_duration - used, 3.0))
                result.append((scene, section))
                scene_key = id(scene)
                usage_count[scene_key] += 1
                last_used_at[scene_key] = seg_idx
                last_scene_key = scene_key
                if usage_count[scene_key] >= MAX_REUSE and len(pool) > 1:
                    pool = [candidate for candidate in pool if id(candidate) != scene_key]
                if len(result) >= max(8, len(scenes) * 3):
                    break
        return result


class EffectTasteEngine:
    def __init__(self, mood: str) -> None:
        self.mood = mood

    def build_filter_chain(
        self,
        scene: SceneRead,
        section: NarrativeSection,
        intensity: float,
        style: dict[str, Any],
        info: dict[str, Any],
    ) -> str:
        filters: list[str] = []
        width = int(info["w"])
        height = int(info["h"])
        grade_strength = 0.5 + intensity * 0.4 + section.energy * 0.1
        grade = self._color_grade(self.mood, grade_strength, scene.scene_type)
        if grade:
            filters.append(grade)
        if scene.brightness > 0.2:
            filters.append("vignette=PI/5")
        if self.mood in {"dark", "sad", "dramatic"} or intensity > 0.6:
            grain = int(8 + intensity * 16)
            if scene.scene_type in {"quiet", "dialogue", "reaction"}:
                grain = int(grain * 0.5)
            filters.append(f"noise=alls={grain}:allf=t+u")
        if (
            scene.scene_type in {"impact", "action"}
            and section.type in {"drop", "chorus"}
            and intensity > 0.55
            and (bool(style.get("glitch", False)) or self.mood in {"hype", "dark", "dramatic"})
        ):
            px = int(4 + intensity * 12)
            filters.append(f"rgbashift=rh={px}:bh=-{px}")
        if scene.scene_type in {"impact", "action"} and scene.motion > 0.45 and intensity > 0.5:
            amount = 0.4 + intensity * 0.6
            filters.append(
                "unsharp="
                f"luma_msize_x=9:luma_msize_y=3:luma_amount={amount:.2f}:"
                f"chroma_msize_x=9:chroma_msize_y=3:chroma_amount={amount * 0.3:.2f}"
            )
        if scene.brightness > 0.35 and scene.scene_type != "impact" and intensity > 0.4:
            glow = min(0.55, 0.25 + intensity * 0.45) if scene.scene_type == "reaction" else 0.25 + intensity * 0.45
            filters.append(
                "unsharp="
                f"luma_msize_x=7:luma_msize_y=7:luma_amount={glow:.2f}:"
                f"chroma_msize_x=5:chroma_msize_y=5:chroma_amount={glow * 0.4:.2f}"
            )
        if (
            scene.scene_type in {"impact", "action"}
            and section.type in {"drop", "chorus"}
            and intensity > 0.60
            and (bool(style.get("shake", False)) or self.mood in {"hype", "dark", "dramatic"})
        ):
            px = int(6 + intensity * 12)
            freq = 10 + int(intensity * 4)
            fps = float(info.get("fps", 30.0))
            filters.append(
                f"crop=iw-{px * 2}:ih-{px * 2}:{px}+{max(1, px // 2)}*sin(n/{fps:.3f}*{freq}*2*PI):{px}"
            )
            filters.append(f"scale={width}:{height}")
        return ",".join([f for f in filters if f])

    def _color_grade(self, mood: str, strength: float, scene_type: str) -> str:
        if scene_type in {"reaction", "dialogue"}:
            strength = min(strength, 0.65)
        s = strength
        grades = {
            "hype": (
                f"eq=contrast={1.0 + s * 0.28:.3f}:saturation={1.0 + s * 0.48:.3f}:brightness={s * 0.025:.4f},"
                f"curves=r='0/0 0.5/{0.5 + s * 0.06:.3f} 1/1':b='0/0 0.5/{0.5 - s * 0.04:.3f} 1/{1 - s * 0.08:.3f}'"
            ),
            "dark": (
                f"eq=contrast={1.0 + s * 0.38:.3f}:saturation={1.0 - s * 0.30:.3f}:brightness={-s * 0.04:.4f},"
                f"curves=r='0/0 0.4/{0.32 + s * 0.05:.3f} 1/{0.88 + s * 0.05:.3f}':b='0/0 0.4/{0.48 - s * 0.04:.3f} 1/1'"
            ),
            "sad": (
                f"eq=contrast={1.0 + s * 0.14:.3f}:saturation={1.0 - s * 0.32:.3f}:brightness={-s * 0.02:.4f},"
                f"curves=b='0/{s * 0.04:.3f} 0.5/{0.5 + s * 0.05:.3f} 1/1':r='0/0 0.5/{0.5 - s * 0.04:.3f} 1/{1 - s * 0.08:.3f}'"
            ),
            "dramatic": (
                f"eq=contrast={1.0 + s * 0.30:.3f}:saturation={1.0 + s * 0.34:.3f}:brightness={s * 0.01:.4f},"
                f"curves=r='0/0 0.5/{0.5 + s * 0.04:.3f} 1/1':b='0/0 0.5/{0.5 + s * 0.04:.3f} 1/1':g='0/0 0.5/{0.5 - s * 0.02:.3f} 1/{1 - s * 0.04:.3f}'"
            ),
            "chill": f"eq=contrast={1.0 + s * 0.12:.3f}:saturation={1.0 + s * 0.20:.3f}:brightness={s * 0.03:.4f}",
        }
        return grades.get(mood, grades["hype"])


class BeatMusicAnalyzer:
    def analyze(self, music_path: str | Path) -> dict[str, Any]:
        if not HAS_LIBROSA:
            return self._fallback_beats(music_path)
        try:
            return self._analyze_librosa(str(music_path))
        except Exception as exc:
            _log(f"librosa analysis failed: {exc}", "WARN")
            return self._fallback_beats(music_path)

    def _analyze_librosa(self, path: str) -> dict[str, Any]:
        y, sr = librosa.load(path, sr=22050, mono=True)
        duration = float(len(y) / sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
        beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        energy_curve = []
        for beat in beats:
            idx = int(np.argmin(np.abs(rms_times - beat)))
            energy_curve.append(float(rms[idx]))
        if energy_curve:
            hi = max(energy_curve)
            lo = min(energy_curve)
            rng = hi - lo if hi > lo else 1.0
            energy_curve = [(value - lo) / rng for value in energy_curve]
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_density = []
        for beat in beats:
            count = sum(1 for onset in onset_times if beat - 0.25 <= onset <= beat + 0.25)
            onset_density.append(float(count) / (0.5 * bpm / 60.0 + 0.1))
        return {
            "bpm": bpm,
            "beats": beats,
            "duration": duration,
            "energy_curve": energy_curve,
            "onset_density": onset_density,
            "drop_times": self._detect_drops(energy_curve, beats),
            "downbeats": beats[::4],
        }

    def _detect_drops(self, energy: list[float], beats: list[float]) -> list[float]:
        if len(energy) < 8:
            return [beats[len(beats) // 3]] if beats else [0.0]
        drops: list[float] = []
        window = 8
        for idx in range(window, len(energy) - 2):
            before = sum(energy[idx - window : idx]) / float(window)
            delta = energy[idx] - before
            if delta > 0.32 and idx < len(beats):
                drops.append(beats[idx])
        if not drops:
            return [beats[len(beats) * 4 // 10]] if beats else [0.0]
        clean = [drops[0]]
        for drop in drops[1:]:
            if drop - clean[-1] > 2.0:
                clean.append(drop)
        return sorted(clean)[:3]

    def _fallback_beats(self, path: str | Path) -> dict[str, Any]:
        info = _probe(str(path)) if Path(str(path)).exists() else {"dur": 30.0}
        duration = float(info.get("dur", 30.0))
        bpm = 130.0
        interval = 60.0 / bpm
        beats = [round(i * interval, 4) for i in range(int(duration / interval) + 1)]
        # Generate musical energy curve instead of flat 0.5
        energy_curve = self._generate_musical_energy(beats, duration)
        onset_density = [max(0.1, e * 0.8 + 0.2) for e in energy_curve]
        drop_times = self._estimate_drop_positions(energy_curve, beats, duration)
        return {
            "bpm": bpm,
            "beats": beats,
            "duration": duration,
            "energy_curve": energy_curve,
            "onset_density": onset_density,
            "drop_times": drop_times,
            "downbeats": beats[::4],
        }

    @staticmethod
    def _generate_musical_energy(beats: list[float], duration: float) -> list[float]:
        """Generate a synthetic energy curve with musical structure."""
        if not beats:
            return []
        curve: list[float] = []
        for beat in beats:
            p = beat / max(0.01, duration)  # position 0..1
            # Intro: low energy ramp up
            if p < 0.12:
                e = 0.15 + p * 2.5
            # Buildup: rising
            elif p < 0.35:
                e = 0.35 + (p - 0.12) * 2.2
            # Drop 1: high energy
            elif p < 0.50:
                e = 0.85 + 0.15 * math.sin((p - 0.35) * 20)
            # Chorus: sustained high
            elif p < 0.65:
                e = 0.70 + 0.12 * math.sin((p - 0.50) * 15)
            # Break: energy dip
            elif p < 0.75:
                e = 0.25 + 0.10 * math.sin((p - 0.65) * 10)
            # Buildup 2: rising again
            elif p < 0.85:
                e = 0.35 + (p - 0.75) * 5.0
            # Drop 2 / Climax: peak
            elif p < 0.95:
                e = 0.90 + 0.10 * math.sin((p - 0.85) * 25)
            # Outro: fade
            else:
                e = 0.90 - (p - 0.95) * 12.0
            curve.append(float(np.clip(e, 0.05, 1.0)))
        return curve

    @staticmethod
    def _estimate_drop_positions(
        energy: list[float], beats: list[float], duration: float,
    ) -> list[float]:
        """Estimate drop positions from synthetic energy curve."""
        drops = []
        if not energy or not beats:
            return [duration * 0.38]
        for i in range(4, len(energy)):
            before = sum(energy[max(0, i - 4) : i]) / 4.0
            if energy[i] - before > 0.25 and i < len(beats):
                if not drops or beats[i] - drops[-1] > 3.0:
                    drops.append(beats[i])
        if not drops:
            drops = [duration * 0.38, duration * 0.85]
        return drops[:3]


class UnifiedRenderEngine:
    def __init__(self) -> None:
        self.engine = FFmpegEngine()

    def render(
        self,
        assignments: list[tuple[SceneRead, NarrativeSection]],
        video_path: str | Path,
        output_path: str | Path,
        music_path: str | Path | None,
        music_start: float,
        mood: str,
        style: dict[str, Any],
        intensity: int,
        info: dict[str, Any],
        beat_map: dict[str, Any],
        restraint: RestraintEngine,
        taste: EffectTasteEngine,
    ) -> bool:
        job_id = uuid.uuid4().hex[:8]
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"arc_{job_id}_"))
        memory = EditorMemory()
        try:
            segments = self._cut_and_process_segments(
                assignments,
                Path(video_path),
                tmp_dir,
                mood,
                style,
                intensity,
                info,
                beat_map,
                restraint,
                taste,
                memory,
            )
            if not segments:
                return False
            return self._concat_and_mix(segments, Path(output_path), music_path, music_start)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cut_and_process_segments(
        self,
        assignments: list[tuple[SceneRead, NarrativeSection]],
        src: Path,
        tmp: Path,
        mood: str,
        style: dict[str, Any],
        intensity: int,
        info: dict[str, Any],
        beat_map: dict[str, Any],
        restraint: RestraintEngine,
        taste: EffectTasteEngine,
        memory: EditorMemory | None = None,
    ) -> list[Path]:
        if memory is None:
            memory = EditorMemory()
        beats = list(beat_map.get("beats", []))
        energy = list(beat_map.get("energy_curve", []))
        segments: list[Path] = []
        for idx, (scene, section) in enumerate(assignments):
            midpoint = (scene.start + scene.end) / 2.0
            beat_energy = 0.5
            if beats:
                beat_idx = min(range(len(beats)), key=lambda i: abs(beats[i] - midpoint))
                beat_energy = float(energy[beat_idx]) if beat_idx < len(energy) else 0.5
            is_peak = beat_energy > 0.72 and section.type in {"drop", "chorus"}
            heavy = restraint.should_fire_heavy(scene, beat_energy, section.type, idx, is_peak)
            # Use continuous intensity curve instead of binary heavy/light
            effect_intensity = restraint.get_continuous_intensity(
                scene, beat_energy, section.type, idx, intensity,
            )
            vf = taste.build_filter_chain(scene, section, effect_intensity, style, info)
            speed = 1.0
            if scene.scene_type == "impact" and heavy and bool(style.get("speed_ramp", False)):
                speed = 1.25
            elif scene.scene_type == "reaction" and heavy and bool(style.get("slow_motion", False)):
                speed = 0.70
            segment_path = tmp / f"seg_{idx:04d}.mp4"
            transition_sec = 0.08 if heavy else 0.045
            ok = self._render_segment(
                src,
                segment_path,
                scene.start,
                scene.end,
                vf,
                speed,
                info,
                transition_sec=transition_sec,
            )
            if ok and segment_path.exists() and segment_path.stat().st_size > 0:
                # Record effect used for memory tracking
                memory.record_effect(vf[:20] if vf else "clean")
                segments.append(segment_path)
            if idx < len(assignments) - 1:
                transition_path = tmp / f"trn_{idx:04d}.mp4"
                # Build transition candidates based on section + mood
                if section.type in {"drop", "chorus"} and (heavy or idx % 2 == 0):
                    candidates = ["flash", "whip"]
                elif mood == "dark" and idx % 4 == 1:
                    candidates = ["glitch", "whip", "flash"]
                else:
                    candidates = ["whip", "flash", "glitch"]
                # Use EditorMemory to avoid repeating the same transition type
                candidates = memory.filter_transitions(candidates)
                transition_type = candidates[0] if candidates else "whip"
                transition_duration = {"flash": 0.085, "whip": 0.095, "glitch": 0.075}.get(transition_type, 0.080)
                # Try real xfade overlap for non-impact transitions
                xfade_done = False
                if transition_type == "whip" and len(segments) >= 1 and segment_path.exists():
                    xfade_done = self._render_xfade_transition(
                        segments[-1], segment_path, transition_path,
                        transition_type="fade", duration=0.12, info=info,
                    )
                if not xfade_done:
                    if self._make_transition_clip(
                        transition_path,
                        mood,
                        info,
                        transition_type=transition_type,
                        duration=transition_duration,
                    ):
                        segments.append(transition_path)
                memory.record_transition(transition_type)
            if heavy and is_peak and bool(style.get("flash", True)):
                flash_path = tmp / f"flash_{idx:04d}.mp4"
                if self._make_flash(flash_path, mood, info):
                    segments.append(flash_path)
        return segments

    def _render_segment(
        self,
        src: Path,
        out: Path,
        start: float,
        end: float,
        vf: str,
        speed: float,
        info: dict[str, Any],
        transition_sec: float = 0.045,
    ) -> bool:
        duration = max(0.05, end - start)
        filters: list[str] = []
        if speed != 1.0:
            filters.append(f"setpts={1.0 / speed:.4f}*PTS")
        if vf:
            filters.append(vf)
        # Always add subtle in/out fades so transitions are not hard raw cuts.
        if duration > max(0.12, transition_sec * 2.0):
            filters.append(f"fade=t=in:st=0:d={transition_sec:.3f}")
            filters.append(f"fade=t=out:st={max(0.0, duration - transition_sec):.3f}:d={transition_sec:.3f}")
        filter_str = ",".join(filters)
        operations = [RenderOperation("raw_filter", {"filter": filter_str})] if filter_str else []
        job = ClipRenderJob(
            input_path=str(src),
            output_path=str(out),
            trim_start=start,
            trim_dur=duration,
            mood="hype",
            effect="__clean__",
            width=max(16, int(info.get("w", 1280))),
            height=max(16, int(info.get("h", 720))),
            fps=max(1.0, float(info.get("fps", 30.0))),
            crf=16,
            preset="faster",
            speed_mult=1.0,
            include_audio=False,
            operations=operations,
            prefer_gpu_effects=False,
        )
        ok, _elapsed = self.engine.render_clip(job)
        if not ok:
            _log("segment render failed through permanent engine", "WARN")
        return ok

    def _make_flash(self, out: Path, mood: str, info: dict[str, Any]) -> bool:
        colors = {
            "hype": "white",
            "dark": "0x100020",
            "dramatic": "white",
            "sad": "0xd0d8ff",
            "chill": "0xe8f0f0",
        }
        cmd = [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colors.get(mood, 'white')}:size={int(info['w'])}x{int(info['h'])}:rate={float(info['fps']):.3f}:duration=0.067",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        ok, _ = _run(cmd, timeout=300)
        return ok and out.exists()

    def _make_transition_clip(
        self,
        out: Path,
        mood: str,
        info: dict[str, Any],
        *,
        transition_type: str,
        duration: float,
    ) -> bool:
        width = int(info["w"])
        height = int(info["h"])
        fps = float(info["fps"])
        colors = {
            "hype": "white",
            "dark": "0x1a0030",
            "dramatic": "white",
            "sad": "0xd0d8ff",
            "chill": "0xe8f0f0",
        }
        base_color = colors.get(mood, "white")
        if transition_type == "flash":
            return self._make_flash(out, mood, info)
        if transition_type == "whip":
            cmd = [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={base_color}:size={width}x{height}:rate={fps:.3f}:duration={duration:.3f}",
                "-vf",
                "drawbox=x='mod(n*48,iw)-iw/4':y=0:w=iw/3:h=ih:color=white@0.40:t=fill,boxblur=10:2,eq=contrast=1.2:saturation=1.1",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ]
            ok, _ = _run(cmd, timeout=300)
            return ok and out.exists()
        cmd = [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps:.3f}:duration={duration:.3f}",
            "-vf",
            "hue=s=2.0,noise=alls=22:allf=t+u,eq=contrast=1.35:brightness=0.03,tblend=all_mode=average",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        ok, _ = _run(cmd, timeout=300)
        return ok and out.exists()

    def _render_xfade_transition(
        self,
        clip_a: Path,
        clip_b: Path,
        out: Path,
        transition_type: str = "fade",
        duration: float = 0.12,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Real overlapping transition using FFmpeg xfade filter."""
        # Probe clip_a duration to set the xfade offset
        try:
            probe_cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", str(clip_a)]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            clip_a_dur = float(result.stdout.strip())
        except Exception:
            return False
        if clip_a_dur <= duration:
            return False
        offset = max(0.0, clip_a_dur - duration)
        xfade_map = {
            "fade": "fade", "dissolve": "fade", "wiperight": "wiperight",
            "wipeleft": "wipeleft", "zoomin": "zoomin",
            "smoothleft": "smoothleft", "smoothright": "smoothright",
        }
        xfade_type = xfade_map.get(transition_type, "fade")
        filter_str = f"xfade=transition={xfade_type}:duration={duration:.3f}:offset={offset:.3f}"
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip_a), "-i", str(clip_b),
            "-filter_complex", filter_str,
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "16", "-pix_fmt", "yuv420p", str(out),
        ]
        ok, err = _run(cmd, timeout=600)
        if not ok:
            _log(f"xfade transition failed: {err[:120]}", "WARN")
        return ok and out.exists() and out.stat().st_size > 0

    def _concat_and_mix(
        self,
        segments: list[Path],
        output: Path,
        music_path: str | Path | None,
        music_start: float,
    ) -> bool:
        list_file = output.parent / f"_elite_concat_{int(time.time())}.txt"
        with open(list_file, "w", encoding="utf-8") as handle:
            for seg in segments:
                escaped = str(seg).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        cmd = [
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
            str(list_file),
        ]
        has_music = bool(music_path) and Path(str(music_path)).exists()
        if has_music:
            cmd += [
                "-ss",
                f"{music_start:.4f}",
                "-i",
                str(music_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
            ]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += [
            "-c:v",
            "copy",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            str(CORES),
            str(output),
        ]
        ok, err = _run(cmd, timeout=900)
        # Fallback: re-encode if stream copy fails (codec mismatch)
        if not ok or not output.exists() or output.stat().st_size == 0:
            _log("stream copy concat failed, falling back to re-encode", "WARN")
            cmd_fallback = [
                FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
            ]
            if has_music:
                cmd_fallback += [
                    "-ss", f"{music_start:.4f}", "-i", str(music_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                ]
            else:
                cmd_fallback += ["-map", "0:v:0"]
            cmd_fallback += [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-threads", str(CORES), str(output),
            ]
            ok, err = _run(cmd_fallback, timeout=900)
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass
        if not ok:
            _log(f"concat failed: {err[:200]}", "WARN")
        return ok and output.exists() and output.stat().st_size > 0


class ArchiveEliteBrain:
    def __init__(self) -> None:
        self.scene_engine = SceneUnderstandingEngine()
        self.music_analyzer = BeatMusicAnalyzer()
        self.cache = AnalysisCache()

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path,
        mood: str,
        prompt: str = "",
        beat_markers: list[float] | None = None,
        music_path: str | Path | None = None,
        music_start: float = 0.0,
        style: dict[str, Any] | None = None,
        intensity: int = 3,
        crf: int = 17,
    ) -> dict[str, Any]:
        del prompt, crf
        started = time.time()
        video_path = Path(video_path)
        output_path = Path(output_path)
        style = style or {}
        info = _probe(video_path)
        duration = float(info["dur"])

        # Beat analysis with caching
        if music_path and Path(str(music_path)).exists():
            cached_beats = self.cache.get_beat_map(str(music_path))
            if cached_beats:
                _log("Using cached beat analysis")
                beat_map = cached_beats
            else:
                beat_map = self.music_analyzer.analyze(music_path)
                self.cache.put_beat_map(str(music_path), beat_map)
        elif beat_markers:
            beat_map = self._beats_to_map(beat_markers, duration)
        else:
            beat_map = self.music_analyzer._fallback_beats(video_path)

        cut_points = self._build_cut_points(
            list(beat_map.get("beats", [])),
            duration,
            mood,
            intensity,
            style,
        )
        # Scene analysis with caching
        scenes: list[SceneRead] = []
        cache_hits = 0
        for idx in range(len(cut_points) - 1):
            start = float(cut_points[idx])
            end = float(cut_points[idx + 1])
            if end - start < 0.1:
                continue
            cached = self.cache.get_scene(str(video_path), start, end)
            if cached:
                scene = SceneRead(**{k: v for k, v in cached.items() if k in SceneRead.__dataclass_fields__})
                cache_hits += 1
            else:
                scene = self.scene_engine.analyze_segment(video_path, start, end)
                try:
                    self.cache.put_scene(str(video_path), start, end, asdict(scene))
                except Exception:
                    pass
            scenes.append(scene)
        if cache_hits:
            _log(f"Cache: {cache_hits}/{len(scenes)} scenes loaded from cache")
        if not scenes:
            return {"ok": False, "duration": 0.0, "reason": "no_scenes"}
        max_edit_value = max(scene.edit_value for scene in scenes) or 1.0
        for scene in scenes:
            scene.impact_rank = scene.edit_value / max_edit_value
        arc = NarrativeArcEngine(beat_map, mood)
        assignments = arc.plan_clip_assignment(scenes)
        if not assignments:
            return {"ok": False, "duration": 0.0, "reason": "no_assignments"}
        renderer = UnifiedRenderEngine()
        restraint = RestraintEngine(mood, len(assignments))
        taste = EffectTasteEngine(mood)
        ok = renderer.render(
            assignments=assignments,
            video_path=video_path,
            output_path=output_path,
            music_path=music_path,
            music_start=music_start,
            mood=mood,
            style=style,
            intensity=intensity,
            info=info,
            beat_map=beat_map,
            restraint=restraint,
            taste=taste,
        )
        elapsed = time.time() - started
        return {
            "ok": ok,
            "duration": duration,
            "segments": len(assignments),
            "elapsed_sec": round(elapsed, 2),
            "elapsed_min": round(elapsed / 60.0, 2),
            "bpm": beat_map.get("bpm", 130.0),
            "sections": [section.type for section in arc.sections],
            "scene_types": [scene.scene_type for scene in scenes[:10]],
            "drops": beat_map.get("drop_times", []),
        }

    def _build_cut_points(
        self,
        beats: list[float],
        duration: float,
        mood: str,
        intensity: int,
        style: dict[str, Any],
    ) -> list[float]:
        if not beats:
            interval = {"hype": 0.8, "dark": 1.2, "sad": 2.0, "dramatic": 1.5, "chill": 2.0}.get(mood, 1.2)
            points = [0.0]
            t = interval
            while t < duration:
                points.append(round(t, 3))
                t += interval
            return points + [duration]
        step = {"hype": 2, "dark": 2, "dramatic": 2, "sad": 3, "chill": 3}.get(mood, 2)
        step = max(1, step - (1 if intensity >= 2 else 0))
        if intensity >= 3 and duration <= 20:
            step = 1
        if bool(style.get("selective_beats", True)) and not bool(style.get("all_beats", False)):
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

    def _beats_to_map(self, beats: list[float], duration: float) -> dict[str, Any]:
        if len(beats) >= 2:
            intervals = [beats[idx + 1] - beats[idx] for idx in range(min(8, len(beats) - 1))]
            bpm = 60.0 / max(0.1, sum(intervals) / len(intervals))
        else:
            bpm = 130.0
        # Generate musical energy instead of flat 0.5
        energy_curve = BeatMusicAnalyzer._generate_musical_energy(beats, duration)
        if not energy_curve:
            energy_curve = [0.5] * len(beats)
        onset_density = [max(0.1, e * 0.8 + 0.2) for e in energy_curve]
        drop_times = BeatMusicAnalyzer._estimate_drop_positions(energy_curve, beats, duration)
        return {
            "bpm": bpm,
            "beats": beats,
            "duration": duration,
            "energy_curve": energy_curve,
            "onset_density": onset_density,
            "drop_times": drop_times,
            "downbeats": beats[::4],
        }


def run_brain_render(
    video_path: Any,
    output_path: Any,
    mood: str,
    prompt: str = "",
    intensity: int = 3,
    music_path: Any = None,
    beat_markers: list[float] | None = None,
    style_profile: dict[str, Any] | None = None,
    variation_seed: int | None = None,
    target_fps: int = 30,
    tmp_dir: Any = None,
    cleanup_tmp: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    del variation_seed, target_fps, tmp_dir, cleanup_tmp, kwargs
    # No longer hard-fail without music — use video audio or synthetic beats
    effective_music = music_path
    if not effective_music or not Path(str(effective_music)).exists():
        # Try using video's own audio for beat analysis
        video_info = _probe(str(video_path))
        if video_info.get("has_audio"):
            effective_music = str(video_path)
            _log("No music track provided — using video audio for beat analysis")
        else:
            effective_music = None
            _log("No music or audio — using synthetic beat fallback")
    brain = ArchiveEliteBrain()
    music_start = 0.0
    if style_profile and "music_start_sec" in style_profile:
        music_start = float(style_profile["music_start_sec"])
    result = brain.run(
        video_path=video_path,
        output_path=output_path,
        mood=mood,
        prompt=prompt,
        beat_markers=beat_markers,
        music_path=effective_music,
        music_start=music_start,
        style=style_profile or {},
        intensity=intensity,
    )
    result["ok"] = bool(result.get("ok", False))
    result["total_time_sec"] = result.get("elapsed_sec", 0.0)
    result["total_time_min"] = result.get("elapsed_min", 0.0)
    result["quality_score"] = 0.85 if result.get("ok") else 0.0
    drops = result.get("drops", [])
    result["music_drop_time"] = drops[0] if drops else None
    result["arc_phases"] = result.get("sections", [])
    return result
