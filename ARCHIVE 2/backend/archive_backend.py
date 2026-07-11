from __future__ import annotations

import csv
import builtins
import hmac
import io
import json
import logging
import logging.handlers
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from collections import deque
from threading import Lock
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import cv2
from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    Limiter = None  # type: ignore[assignment,misc]
    get_remote_address = None  # type: ignore[assignment]
    LIMITER_AVAILABLE = False

import librosa
import numpy as np
from video.cache import file_cache_signature, load_json_cache, prune_files, write_json_cache
from video.audio_features import analyze_audio_features
from brain.editor_personality import apply_personality_to_style, infer_editor_personality
from brain.memory import load_editor_memory, save_editor_memory
from video.render_engine import ffmpeg_hwaccel_input_args, prune_runtime_temp, run_subprocess
from brain.scene_analysis import scene_analysis_config
from video.smart_cutting import annotate_cut_scores
from video.transitions import transition_name_for_mood
from brain.visual_understanding import enrich_segment_rows, transition_hint_from_row
try:
    from archive_pyav_renderer import render_video_with_pyav

    PYAV_RENDERER_AVAILABLE = True
except Exception:
    render_video_with_pyav = None
    PYAV_RENDERER_AVAILABLE = False
try:
    from archive_engine import FFmpegEngine as PermanentFFmpegEngine

    PERMANENT_ENGINE_AVAILABLE = True
except Exception:
    PermanentFFmpegEngine = None  # type: ignore[assignment]
    PERMANENT_ENGINE_AVAILABLE = False
from brain.nlp import (
    build_style_profile_from_prompt,
    infer_mood_from_prompt,
    infer_music_profile_from_prompt,
)
try:
    from video.archive_music_upgrade import (
        init_music_upgrade,
        pick_local_phonk_track_v2,
        select_music_excerpt_for_video_v2,
        build_emotional_arc,
        build_human_cut_points,
        build_cinematic_grade_ffmpeg,
    )

    MUSIC_UPGRADE_AVAILABLE = True
except Exception:
    init_music_upgrade = None
    pick_local_phonk_track_v2 = None
    select_music_excerpt_for_video_v2 = None
    build_emotional_arc = None
    build_human_cut_points = None
    build_cinematic_grade_ffmpeg = None
    MUSIC_UPGRADE_AVAILABLE = False

try:
    from video.archive_effects import apply_effects_to_clip, get_effects_for_style  # type: ignore

    EFFECTS_ENGINE_AVAILABLE = True
except Exception:
    apply_effects_to_clip = None
    get_effects_for_style = None
    EFFECTS_ENGINE_AVAILABLE = False

try:
    from video.archive_fast_render import (
        fast_render as run_archive_fast_render,
        is_fast_render_compatible as archive_fast_render_is_compatible,
    )

    FAST_RENDERER_AVAILABLE = True
except Exception:
    run_archive_fast_render = None
    archive_fast_render_is_compatible = None
    FAST_RENDERER_AVAILABLE = False

try:
    from brain.archive_decision_brain import run_single_effect as run_decision_single_effect  # type: ignore

    DECISION_BRAIN_AVAILABLE = True
except Exception:
    run_decision_single_effect = None
    DECISION_BRAIN_AVAILABLE = False

CPP_EFFECTS_EXE = os.getenv("ARCHIVE_CPP_EFFECTS_EXE")
if not CPP_EFFECTS_EXE:
    maybe_exe = Path(__file__).resolve().parent.parent / "cpp" / "archive_effects.exe"
    if maybe_exe.exists():
        CPP_EFFECTS_EXE = str(maybe_exe)
    if CPP_EFFECTS_EXE:
        print(f"[BOOT] Master Brain Engine detected at: {CPP_EFFECTS_EXE}")
    else:
        print("[BOOT] WARNING: Master Brain Engine NOT found! Elite VFX will be skipped.")
CPP_FRAME_ENGINE_EXE = os.getenv("ARCHIVE_FRAME_ENGINE_EXE")
if not CPP_FRAME_ENGINE_EXE:
    maybe_frame_exe = Path(__file__).resolve().parent.parent / "cpp" / "archive_frame_engine.exe"
    if maybe_frame_exe.exists():
        CPP_FRAME_ENGINE_EXE = str(maybe_frame_exe)
    if CPP_FRAME_ENGINE_EXE:
        print(f"[BOOT] Frame Engine detected at: {CPP_FRAME_ENGINE_EXE}")
AI_EDITOR_EXE = os.getenv("ARCHIVE_AI_EDITOR_EXE")
if not AI_EDITOR_EXE:
    for maybe_ai_editor in (
        Path(__file__).resolve().parent.parent / "ai_editor" / "build" / "Release" / "ai_editor.exe",
        Path(__file__).resolve().parent.parent / "ai_editor" / "build" / "Debug" / "ai_editor.exe",
    ):
        if maybe_ai_editor.exists():
            AI_EDITOR_EXE = str(maybe_ai_editor)
            break
    if AI_EDITOR_EXE:
        print(f"[BOOT] AI Editor cortex detected at: {AI_EDITOR_EXE}")
try:
    from brain.archive_brain import run_brain_render  # type: ignore

    BRAIN_RENDER_AVAILABLE = True
except Exception:
    run_brain_render = None
    BRAIN_RENDER_AVAILABLE = False

try:
    import joblib  # type: ignore

    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    from skimage import exposure  # type: ignore

    SKIMAGE_AVAILABLE = True
except Exception:
    exposure = None
    SKIMAGE_AVAILABLE = False

try:
    import mediapipe as mp  # type: ignore

    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore

    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

MOVIEPY_DISABLED = True
_moviepy_audio_loop_fx = None

if MOVIEPY_DISABLED and not getattr(builtins, "_archive_moviepy_disabled", False):
    _ARCHIVE_ORIGINAL_IMPORT = builtins.__import__

    def _archive_block_moviepy_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "moviepy" or name.startswith("moviepy."):
            raise ImportError("MoviePy is disabled in ARCHIVE; use PyAV/FFmpeg renderers.")
        return _ARCHIVE_ORIGINAL_IMPORT(name, *args, **kwargs)

    builtins.__import__ = _archive_block_moviepy_import
    builtins._archive_moviepy_disabled = True

sys.dont_write_bytecode = True

# =============================================================================
# APP FACTORY & CORE CONFIG
# =============================================================================

app = Flask(__name__)
# CORS: restrict to explicit origins (default = local dev only).
# In production set ARCHIVE_CORS_ORIGINS="https://yourapp.com,https://other.com".
_cors_origins = [
    o.strip()
    for o in os.getenv(
        "ARCHIVE_CORS_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000,"
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]
CORS(app, origins=_cors_origins, supports_credentials=False)
BACKEND_BUILD = "2026-03-04-runtime-variation7-hardmusicfix"
API_DEBUG = os.getenv("ARCHIVE_API_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
ARCHIVE_VARIATION_MODE = os.getenv("ARCHIVE_VARIATION_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
ARCHIVE_ANIME_ONLY_MODE = os.getenv("ARCHIVE_ANIME_ONLY_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
ARCHIVE_CONTENT_SAFETY_MODE = os.getenv("ARCHIVE_CONTENT_SAFETY_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(tempfile.gettempdir()) / "archive_runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"


def _bootstrap_bundled_ffmpeg() -> str:
    """Expose the venv-bundled imageio ffmpeg as ffmpeg.exe for subprocess calls."""
    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    try:
        import imageio_ffmpeg  # type: ignore

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return ""

    if not bundled.exists():
        return ""

    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        runtime_ffmpeg = RUNTIME_DIR / "ffmpeg.exe"
        if (
            not runtime_ffmpeg.exists()
            or runtime_ffmpeg.stat().st_size != bundled.stat().st_size
        ):
            shutil.copy2(bundled, runtime_ffmpeg)
        os.environ["IMAGEIO_FFMPEG_EXE"] = str(runtime_ffmpeg)
        os.environ["FFMPEG_BINARY"] = str(runtime_ffmpeg)
        runtime_dir = str(RUNTIME_DIR)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if runtime_dir not in path_parts:
            os.environ["PATH"] = runtime_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[BOOT] Bundled ffmpeg exposed at: {runtime_ffmpeg}")
        return str(runtime_ffmpeg)
    except Exception as exc:
        print(f"[BOOT] WARNING: Could not expose bundled ffmpeg: {exc}")
        return ""


BOOTSTRAPPED_FFMPEG_EXE = _bootstrap_bundled_ffmpeg()
ARCHIVE_BRAIN_TIMEOUT_SEC = int(float(os.getenv("ARCHIVE_BRAIN_TIMEOUT_SEC", "1740")))
ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC = int(float(os.getenv("ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC", "90")))
ARCHIVE_SPEED_MODE = os.getenv("ARCHIVE_SPEED_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
ARCHIVE_DEFAULT_FAST_RENDER = os.getenv("ARCHIVE_DEFAULT_FAST_RENDER", "0").strip().lower() in {"1", "true", "yes", "on"}
ARCHIVE_DEFAULT_BRAIN_RENDER = os.getenv("ARCHIVE_DEFAULT_BRAIN_RENDER", "1").strip().lower() in {"1", "true", "yes", "on"}

# --- Upload validation: 500 MB hard ceiling ---
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("ARCHIVE_MAX_UPLOAD_MB", "500")) * 1024 * 1024

# --- API key authentication (opt-in) ---
# Set ARCHIVE_API_KEY env var to enable. Leave unset to allow open access.
ARCHIVE_API_KEY = os.getenv("ARCHIVE_API_KEY", "").strip()

# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

def _setup_logging() -> None:
    """Configure structured logging with console + rotating file output."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RUNTIME_DIR / "archive.log"
    root_logger = logging.getLogger()
    # Only add handlers once (avoid duplicates on module reimport).
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers):
        root_logger.setLevel(logging.DEBUG if API_DEBUG else logging.INFO)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT)
        # File handler — rotates at 10 MB, keeps 3 backups.
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG if API_DEBUG else logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            pass  # Don't crash if log directory is not writable.
        # Console handler.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    # Quiet noisy libraries.
    for lib in ("werkzeug", "imageio", "urllib3", "numba"):
        logging.getLogger(lib).setLevel(logging.WARNING)

_setup_logging()
logger = logging.getLogger("archive")

# =============================================================================
# RATE LIMITING
# =============================================================================

if LIMITER_AVAILABLE and Limiter is not None:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["60 per minute"],
        storage_uri="memory://",
    )
else:
    limiter = None  # type: ignore[assignment]
    logger.warning("flask-limiter not installed — rate limiting is DISABLED.")


def _rate_limit(limit_string: str):
    """Decorator that applies rate limiting when flask-limiter is available."""
    if limiter is not None:
        return limiter.limit(limit_string)
    # No-op decorator fallback.
    def _noop(fn):
        return fn
    return _noop

# =============================================================================
# REQUEST LIFECYCLE HOOKS
# =============================================================================

@app.before_request
def _before_request_hook():
    """Attach request ID, enforce API key, log incoming requests."""
    # --- Request ID ---
    g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    g.request_start = time.perf_counter()

    # --- API key authentication (skip for static assets) ---
    if ARCHIVE_API_KEY:
        path = request.path
        # Allow unauthenticated access to static files and the health endpoint.
        if path.startswith("/api/") and path != "/api/health":
            provided_key = request.headers.get("X-API-Key", "").strip()
            # Constant-time comparison prevents timing side-channel attacks.
            if not hmac.compare_digest(provided_key, ARCHIVE_API_KEY):
                logger.warning(
                    "[%s] AUTH DENIED %s %s from %s",
                    g.request_id, request.method, path, request.remote_addr,
                )
                return jsonify({"error": "unauthorized", "hint": "Set the X-API-Key header."}), 401

    # --- Log incoming request ---
    if request.path.startswith("/api/"):
        logger.info(
            "[%s] --> %s %s from %s",
            g.request_id, request.method, request.path, request.remote_addr,
        )


@app.after_request
def _after_request_hook(response):
    """Log response status and inject request-id header."""
    req_id = getattr(g, "request_id", "?")
    response.headers["X-Request-ID"] = req_id
    if request.path.startswith("/api/"):
        elapsed = time.perf_counter() - getattr(g, "request_start", time.perf_counter())
        logger.info(
            "[%s] <-- %s %s %d (%.2fs)",
            req_id, request.method, request.path, response.status_code, elapsed,
        )
    return response


if MUSIC_UPGRADE_AVAILABLE and init_music_upgrade:
    try:
        init_music_upgrade(RUNTIME_DIR)
    except Exception:
        pass
PHONK_DIR = BASE_DIR / "phonks"
ARTIFACT_DIR = BASE_DIR / "artifacts"

STATIC_EXTENSIONS = {".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico"}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PHONK_DIR.mkdir(exist_ok=True)
prune_runtime_temp(RUNTIME_DIR, max_age_seconds=1800)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _can_write_to_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".archive_write_probe_{uuid.uuid4().hex}.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _collect_startup_diagnostics() -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg") or ""
    return {
        "build": BACKEND_BUILD,
        "python": sys.version.split()[0],
        "api_debug": API_DEBUG,
        "dependencies": {
            "flask": True,
            "werkzeug": True,
            "numpy": True,
            "opencv": True,
            "moviepy": False,
            "moviepy_disabled": MOVIEPY_DISABLED,
            "pyav_renderer": PYAV_RENDERER_AVAILABLE,
            "librosa": True,
            "music_upgrade": MUSIC_UPGRADE_AVAILABLE,
            "effects_engine": EFFECTS_ENGINE_AVAILABLE,
            "fast_renderer": FAST_RENDERER_AVAILABLE,
            "permanent_render_engine": PERMANENT_ENGINE_AVAILABLE,
            "brain_renderer": BRAIN_RENDER_AVAILABLE,
            "decision_brain": DECISION_BRAIN_AVAILABLE,
            "mediapipe_optional": MEDIAPIPE_AVAILABLE,
            "onnxruntime_optional": ONNXRUNTIME_AVAILABLE,
        },
        "external_engines": {
            "cpp_effects_exe": bool(CPP_EFFECTS_EXE and Path(CPP_EFFECTS_EXE).exists()),
            "cpp_frame_engine_exe": bool(CPP_FRAME_ENGINE_EXE and Path(CPP_FRAME_ENGINE_EXE).exists()),
            "ai_editor_exe": bool(AI_EDITOR_EXE and Path(AI_EDITOR_EXE).exists()),
        },
        "feature_flags": {
            "anime_only_mode": ARCHIVE_ANIME_ONLY_MODE,
            "content_safety_mode": ARCHIVE_CONTENT_SAFETY_MODE,
        },
        "runtime_paths": {
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
            "phonk_dir": str(PHONK_DIR),
            "upload_dir_writable": _can_write_to_dir(UPLOAD_DIR),
            "output_dir_writable": _can_write_to_dir(OUTPUT_DIR),
            "phonk_dir_writable": _can_write_to_dir(PHONK_DIR),
        },
        "ffmpeg": {
            "available": bool(ffmpeg_path),
            "path": ffmpeg_path,
        },
    }


STARTUP_DIAGNOSTICS = _collect_startup_diagnostics()
_RENDER_STATUS_LOCK = Lock()
_RENDER_STATUS: dict[str, dict[str, Any]] = {}
_RENDER_STATUS_TTL_SEC = 60 * 60


def _prune_render_status_cache(now_ts: float | None = None) -> None:
    now = float(now_ts or time.time())
    stale_ids: list[str] = []
    for rid, payload in _RENDER_STATUS.items():
        updated_at = float(payload.get("updated_at", 0.0) or 0.0)
        done = bool(payload.get("done", False))
        if done and (now - updated_at) > _RENDER_STATUS_TTL_SEC:
            stale_ids.append(rid)
    for rid in stale_ids:
        _RENDER_STATUS.pop(rid, None)


def _set_render_status(
    render_id: str,
    *,
    stage: str,
    progress_pct: float,
    message: str,
    done: bool = False,
    ok: bool | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    now = time.time()
    with _RENDER_STATUS_LOCK:
        _prune_render_status_cache(now)
        prev = _RENDER_STATUS.get(render_id, {})
        started_at = float(prev.get("started_at", now) or now)
        payload: dict[str, Any] = {
            "render_id": render_id,
            "stage": stage,
            "message": message,
            "progress_pct": float(np.clip(progress_pct, 0.0, 100.0)),
            "done": bool(done),
            "ok": ok if ok is not None else prev.get("ok"),
            "started_at": started_at,
            "updated_at": now,
        }
        if extras:
            payload.update(extras)
        _RENDER_STATUS[render_id] = payload


def _get_render_status(render_id: str) -> dict[str, Any] | None:
    with _RENDER_STATUS_LOCK:
        _prune_render_status_cache()
        payload = _RENDER_STATUS.get(render_id)
        if not payload:
            return None
        return dict(payload)


# ─────────────────────────────────────────────────────────────────────────────
# CANCELLATION TRACKING
# ─────────────────────────────────────────────────────────────────────────────
_CANCELLED_RENDERS_LOCK = Lock()
_CANCELLED_RENDERS: set[str] = set()


def _is_render_cancelled(render_id: str) -> bool:
    """Check if a render has been cancelled by the user."""
    with _CANCELLED_RENDERS_LOCK:
        return render_id in _CANCELLED_RENDERS


def _mark_render_cancelled(render_id: str) -> None:
    """Mark a render as cancelled."""
    with _CANCELLED_RENDERS_LOCK:
        _CANCELLED_RENDERS.add(render_id)


def _clear_cancelled_render(render_id: str) -> None:
    """Remove a render from cancelled set (cleanup after completion)."""
    with _CANCELLED_RENDERS_LOCK:
        _CANCELLED_RENDERS.discard(render_id)


try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, min(8, os.cpu_count() or 4)))
except Exception:
    pass

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _audio_loop_clip(clip: Any, duration: float) -> Any:
    target = float(max(0.0, duration or 0.0))
    if clip is None or target <= 0:
        return clip
    raise RuntimeError("MoviePy audio looping is disabled; use FFmpeg/PyAV audio handling.")

MOOD_TO_QUERY = {
    "hype": "aggressive phonk drift",
    "dark": "dark phonk bass",
    "sad": "emotional phonk",
    "dramatic": "cinematic phonk",
    "chill": "lofi phonk",
}

MOOD_TO_BPM = {
    "hype": 154,
    "dark": 142,
    "sad": 118,
    "dramatic": 132,
    "chill": 108,
}

FALLBACK_TRACKS_BY_MOOD = {
    "hype": [
        {"title": "Night Drift Phonk", "duration_sec": 42},
        {"title": "Street Battle Phonk", "duration_sec": 38},
    ],
    "dark": [
        {"title": "Shadow Bass Phonk", "duration_sec": 46},
        {"title": "Noir Engine Phonk", "duration_sec": 40},
    ],
    "sad": [
        {"title": "Broken Memory Phonk", "duration_sec": 48},
        {"title": "Last Frame Phonk", "duration_sec": 44},
    ],
    "dramatic": [
        {"title": "Cinematic Impact Phonk", "duration_sec": 50},
        {"title": "Final Arc Phonk", "duration_sec": 45},
    ],
    "chill": [
        {"title": "Low Key Phonk", "duration_sec": 52},
        {"title": "Neon Calm Phonk", "duration_sec": 47},
    ],
}

MOOD_KEYWORDS = {
    "hype": ["hype", "energy", "drift", "hard", "sport", "aggressive", "fight", "happy", "party", "uplift"],
    "dark": ["dark", "trap", "noir", "shadow", "villain", "horror"],
    "sad": ["sad", "slow", "slowed", "emotional", "melancholy", "heartbreak", "cry", "tears", "heart"],
    "dramatic": ["dramatic", "cinematic", "impact", "story", "epic", "romance"],
    "chill": ["chill", "lofi", "calm", "soft", "ambient", "romantic", "pleasant", "love", "sweet", "happy", "relax"],
}

INTENSITY_KEYWORDS = ["max", "ultra", "insane", "crazy", "professional", "cinematic", "hardcore", "extreme"]

STYLE_KEYWORDS = {
    "shake": ["shake", "camera shake"],
    "zoom": ["zoom", "punch", "punch in"],
    "flash": ["flash", "strobe", "blink"],
    "glow": ["glow", "bloom", "neon"],
    "glitch": ["glitch", "rgb split", "chromatic"],
    "motion_blur": ["motion blur", "blur"],
    "grain": ["grain", "film grain", "noise"],
    "vignette": ["vignette", "dark corners"],
    "edge_glow": ["edge", "outline", "edge glow"],
    "speed_ramp": ["speed ramp", "ramp", "velocity"],
    "slow_motion": ["slow motion", "slowmo", "slow-mo", "slow motion effect", "slo mo", "slomo", "slomotion"],
    "fast_motion": ["fast motion", "fastmo", "fast-mo", "speed up", "speedup", "fast forward", "fastforward"],
    "reverse": ["reverse", "rewind", "backwards", "reverse clip", "reverse video", "time reverse"],
    "boomerang": ["boomerang", "forward reverse", "reverse forward", "ping pong", "pingpong", "yo yo", "yoyo"],
    "anime_tone": ["anime tone", "toon", "cel shade", "stylized"],
    "ken_burns": ["ken burns", "cinematic pan", "parallax", "camera move"],
    "chroma_pulse": ["chroma pulse", "color pulse", "chromatic pulse"],
    "color_grade": [
        "color grade",
        "colour grade",
        "color grading",
        "colour grading",
        "cinematic color",
        "cinematic colours",
        "cinematic colors",
        "teal orange",
        "lut",
        "vibrant colors",
        "boost colors",
        "boost colour",
        "increase saturation",
    ],
    "character_focus": [
        "character",
        "charecter",
        "face",
        "hero",
        "main character",
        "main charecter",
        "highlight character",
        "highlight charecter",
    ],
    "black_white": ["black and white", "black white", "balck and white", "monochrome", "grayscale", "greyscale", "b and w", "bw"],
    "no_color_grade": [
        "dont change colors",
        "don't change colors",
        "do not change colors",
        "do not change colour",
        "do not change colours",
        "not change colors",
        "not change colour",
        "not change colours",
        "not to change colors",
        "not to change colour",
        "not to change colours",
        "dont change colour",
        "dont change colours",
        "don't change colour",
        "don't change colours",
        "no color change",
        "no colour change",
        "no colours change",
        "no color grade",
        "no colour grade",
        "no color grading",
        "no colour grading",
        "keep original colors",
        "keep original colours",
        "keep original colour",
        "keep colors same",
        "keep colours same",
        "no grading",
    ],
    "fire_lines": [
        "fire lines",
        "fire boundary",
        "boundary fire",
        "fire outline",
        "flame lines",
        "fire highlight",
        "ae fire",
        "burn lines",
        "lightning lines",
        "electric outline",
        "character boundary",
        "charecter boundary",
        "highlight boundary",
        "outline character",
        "character boundaries",
        "boundary highlight",
        "highlight character boundary",
        "highlight character boundaries",
        "highlighting the character boundary",
        "highlighting character boundary",
    ],
    "overlap_transitions": [
        "overlap",
        "overlapping",
        "crossfade",
        "cross fade",
        "seamless transition",
        "smooth transition",
        "cool transition",
        "cool transitions",
        "transition heavy",
        "more transitions",
        "better transitions",
    ],
    "particles": [
        "particles",
        "particle effects",
        "snow",
        "rain",
        "sparks",
        "smoke",
        "dust",
        "embers",
    ],
    "energy_aura": [
        "energy aura",
        "aura",
        "saber",
        "neon lines",
        "magic shield",
        "magical shield",
        "electric aura",
        "chakra aura",
    ],
    "turbulent_displace": [
        "turbulent displace",
        "turbulence",
        "warp scene",
        "scene warp",
        "distortion",
        "channel effect",
        "surreal look",
        "speed lines",
    ],
    "posterize_time": [
        "posterize time",
        "12 fps look",
        "15 fps look",
        "stop motion look",
        "impact fps",
    ],
    "mask_transition": [
        "mask transition",
        "masking transition",
        "mask slide",
        "luma wipe",
        "matte transition",
        "wipe transition",
        "character cut transition",
        "custom transition",
    ],
    "loop_clips": [
        "repeat clip",
        "repeat clips",
        "repeat some clips",
        "clip repeat",
        "loop clip",
        "loop clips",
        "replay clip",
        "replay clips",
        "stutter loop",
        "stutter repeat",
    ],
    "blend_overlay": [
        "blend mode",
        "screen blend",
        "add blend",
        "overlay blend",
        "light leaks",
        "texture overlay",
        "compositing",
        "layering",
    ],
    "mask_roto": [
        "masking",
        "mask rotoscoping",
        "rotoscope",
        "rotoscoping",
        "rotobrush",
        "frame by frame mask",
        "object mask",
        "subject mask",
        "mask around character",
    ],
    "chroma_key": [
        "green screen",
        "greenscreen",
        "chroma key",
        "background remove",
        "remove green background",
        "replace green background",
        "replace background color",
    ],
    "difference_matte": [
        "difference matte",
        "difference mask",
        "background difference",
        "motion matte",
    ],
    "object_inpaint": [
        "inpaint",
        "inpainting",
        "object inpaint",
        "content aware fill",
        "remove object",
        "erase object",
        "background inpaint",
        "replace background keeping subject",
    ],
    "pro_edit_pack": [
        "pro anime edit",
        "professional anime edit",
        "ae style",
        "after effects style",
        "ae toolset",
        "ae tools",
        "after effects tools",
        "all ae tools",
        "all after effects tools",
        "hardcore edit",
        "top level edit",
        "advanced vfx",
        "all vfx",
        "all effects",
    ],
    "selective_beats": [
        "selective beats",
        "smart beats",
        "few beats",
        "some beats",
        "not all beats",
        "only some beats",
        "where required",
        "where needed",
        "wisely",
        "human like",
        "intelligent beats",
    ],
    "all_beats": [
        "all beats",
        "every beat",
        "each beat",
        "on all beats",
        "on every beat",
    ],
    "fast_render": [
        "fast render",
        "minimum time",
        "quick render",
        "faster",
        "asap",
        "less time",
    ],
    "quality_priority": [
        "quality priority",
        "best quality",
        "highest quality",
        "quality first",
        "no quality loss",
        "no quality compromise",
    ],
    "smart_music": [
        "smart music",
        "best phonk",
        "best music",
        "music according to vibe",
        "phonk according to vibe",
        "match music to video",
        "suitable phonk",
        "music auto choose",
    ],
    "no_phonk": [
        "no phonk",
        "without phonk",
        "dont add phonk",
        "dont add any phonk",
        "don't add phonk",
        "don't add any phonk",
        "do not add phonk",
        "do not add any phonk",
        "no phonk please",
        "not phonk",
        "avoid phonk",
    ],
    "no_music": [
        "no music",
        "without music",
        "dont add music",
        "dont add any music",
        "don't add music",
        "don't add any music",
        "do not add music",
        "do not add any music",
        "no bgm",
        "no background music",
        "keep original audio",
        "use original audio",
        "keep video audio",
        "original sound only",
    ],
}


def _is_style_negated_fallback(text: str, terms: list[str]) -> bool:
    prefixes = ("no", "not", "without", "avoid", "skip", "remove", "dont", "do not", "don t")
    for term in terms:
        t = re.sub(r"\s+", " ", str(term).strip().lower())
        if not t:
            continue
        escaped = re.escape(t)
        for pref in prefixes:
            pref_esc = re.escape(pref)
            if re.search(rf"\b{pref_esc}\s+(?:any\s+)?{escaped}\b", text):
                return True
    return False

TRACK_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
TRACK_ANALYSIS_CACHE_PATH = RUNTIME_DIR / "track_analysis_cache.json"
TRACK_ANALYSIS_CACHE_DIRTY = 0
RECENT_TRACKS_PATH = RUNTIME_DIR / "recent_tracks.json"
EDITOR_MEMORY_PATH = RUNTIME_DIR / "editor_memory.json"
EDITOR_MEMORY = load_editor_memory(EDITOR_MEMORY_PATH)
_EDIT_BRAIN_CACHE: dict[str, Any] = {"loaded": False, "artifact": None}
AUDIO_RESAMPLE_CANDIDATES = ("soxr_mq", "kaiser_fast", None)
_ENCODER_CACHE: dict[str, Any] = {"checked": False, "codec": "libx264"}
_AUDIO_BEAT_CACHE: dict[str, tuple[int, list[float]]] = {}
_MUSIC_EXCERPT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
LAST_SELECTED_TRACK_KEY: str = ""
RECENT_TRACKS: list[str] = []
_MEDIAPIPE_SEG_CACHE: dict[str, Any] = {"loaded": False, "segmenter": None}
_MATTE_MODEL_CACHE: dict[str, Any] = {
    "loaded": False,
    "session": None,
    "input_name": "",
    "input_hw": (320, 320),
    "path": "",
}
BEAT_ANALYSIS_MAX_SEC = 12.0
HEAVY_EFFECT_MIN_SEG_SEC = 0.24
FIRE_BOUNDARY_REFRESH_HZ = 1.8
EDGE_GLOW_REFRESH_HZ = 2.0
FFMPEG_OFFLOAD_EFFECT_KEYS = ("grain", "vignette", "edge_glow", "color_grade")
USE_FAST_ANIME_TONE = os.getenv("ARCHIVE_USE_SKIMAGE_TONE", "0").lower() not in {"1", "true", "yes"}
SUBJECT_MASK_TEMPORAL_ALPHA = float(np.clip(float(os.getenv("ARCHIVE_MASK_TEMPORAL_ALPHA", "0.72")), 0.0, 0.94))
RENDER_RETENTION_SECONDS = max(60, int(float(os.getenv("ARCHIVE_RENDER_RETENTION_SECONDS", "900"))))
MAX_RETAINED_RENDERS = max(1, int(os.getenv("ARCHIVE_MAX_RETAINED_RENDERS", "1")))

ANIME_EDIT_ARCHETYPES = {
    "action": {"auto_zoom": True, "auto_shake": True, "auto_flash": True},
    "hard_cut": {"auto_zoom": True, "auto_shake": False, "auto_flash": True},
    "character_focus": {"auto_zoom": False, "auto_shake": False, "auto_flash": False},
    "dark_tension": {"auto_zoom": False, "auto_shake": False, "auto_flash": False},
    "buildup": {"auto_zoom": False, "auto_shake": False, "auto_flash": False},
    "neutral": {"auto_zoom": False, "auto_shake": False, "auto_flash": False},
}


@app.errorhandler(413)
def handle_too_large(exc: Exception) -> Any:
    """Reject uploads exceeding MAX_CONTENT_LENGTH."""
    max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
    logger.warning(
        "[%s] Upload rejected: file exceeds %d MB limit.",
        getattr(g, "request_id", "?"), max_mb,
    )
    return jsonify({
        "error": f"File too large. Maximum upload size is {max_mb} MB.",
        "max_upload_mb": max_mb,
    }), 413


@app.errorhandler(429)
def handle_rate_limit(exc: Exception) -> Any:
    """Return a clear message when rate limit is exceeded."""
    logger.warning(
        "[%s] Rate limit exceeded: %s %s from %s",
        getattr(g, "request_id", "?"), request.method, request.path, request.remote_addr,
    )
    return jsonify({
        "error": "Rate limit exceeded. Please slow down.",
        "retry_after_seconds": 60,
    }), 429


@app.errorhandler(Exception)
def handle_global_error(exc: Exception) -> Any:
    # Keep API failures machine-readable to debug client-side 500s quickly.
    req_id = getattr(g, "request_id", "?")
    if request.path.startswith("/api/"):
        if isinstance(exc, HTTPException):
            return jsonify({
                "error": f"{exc.name}: {exc.description}",
                "request_id": req_id,
            }), int(exc.code or 500)
        trace = traceback.format_exc()
        logger.error("[%s] Unhandled API exception:\n%s", req_id, trace)
        payload: dict[str, Any] = {
            "error": f"unhandled error: {type(exc).__name__}: {exc}",
            "build": BACKEND_BUILD,
            "request_id": req_id,
        }
        if API_DEBUG:
            payload["debug_traceback"] = trace[-3000:]
        return jsonify(payload), 500
    raise exc


@app.get("/")
def serve_index() -> Any:
    """Serve the main index.html file"""
    index_path = Path(__file__).resolve().parent / "index.html"
    if not index_path.exists():
        index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        logger.error(f"[ERROR] index.html not found at {index_path}")
        return jsonify({"error": "index.html not found", "base_dir": str(BASE_DIR)}), 500
    logger.info(f"[ROOT] Serving index.html from {index_path.parent}")
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        from flask import Response
        return Response(content, mimetype='text/html')
    except Exception as e:
        logger.error(f"[ERROR] Failed to serve index.html: {e}")
        return jsonify({"error": str(e)}), 500


@app.get("/rating.html")
def serve_rating() -> Any:
    rating_path = Path(__file__).resolve().parent / "rating.html"
    if not rating_path.exists():
        rating_path = BASE_DIR / "rating.html"
    if rating_path.exists():
        try:
            with open(rating_path, 'r', encoding='utf-8') as f:
                content = f.read()
            from flask import Response
            return Response(content, mimetype='text/html')
        except Exception as e:
            logger.error(f"[ERROR] Failed to serve rating.html: {e}")
            return jsonify({"error": str(e)}), 500
    return ("Not Found", 404)


@app.get("/ARCHIVE 2/")
@app.get("/ARCHIVE 2")
def serve_index_archive2() -> Any:
    """Redirect ARCHIVE 2/ requests to root"""
    logger.info("[REDIRECT] Serving index.html for /ARCHIVE 2/ request")
    return serve_index()


@app.get("/<path:filename>")
def serve_static(filename: str) -> Any:
    """Serve static files"""
    if filename.startswith("api/"):
        return ("Not Found", 404)
    
    # Handle ARCHIVE 2 prefix
    if filename.startswith("ARCHIVE 2/"):
        filename = filename.replace("ARCHIVE 2/", "", 1)
    
    target = Path(__file__).resolve().parent / filename
    if not target.exists():
        target = BASE_DIR / filename
    
    # Block non-static files
    if target.suffix.lower() not in STATIC_EXTENSIONS:
        return ("Not Found", 404)
    
    # Check if file exists
    if not target.exists():
        logger.debug(f"[STATIC] File not found: {target}")
        return ("Not Found", 404)
    
    logger.debug(f"[STATIC] Serving {filename} from {target.parent}")
    return send_from_directory(target.parent, target.name)

_face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(_face_cascade_path)


@dataclass
class Track:
    id: int | None
    title: str
    tags: str
    duration_sec: int | None
    preview_url: str
    audio_url: str
    user: str
    source: str


def infer_mood(prompt: str) -> str:
    try:
        return infer_mood_from_prompt(prompt)
    except Exception:
        text = (prompt or "").lower()
        if any(k in text for k in ["hype", "epic", "energetic", "fight", "fast"]):
            return "hype"
        if any(k in text for k in ["dark", "intense", "villain", "shadow"]):
            return "dark"
        if any(k in text for k in ["sad", "emotional", "heartbreak", "cry"]):
            return "sad"
        if any(k in text for k in ["dramatic", "cinematic", "story"]):
            return "dramatic"
        return "chill"


def build_style_profile(prompt: str, mood: str) -> dict[str, Any]:
    try:
        profile = build_style_profile_from_prompt(prompt, mood)
        return apply_personality_to_style(profile, infer_editor_personality(prompt, mood))
    except Exception:
        text = (prompt or "").lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", text)
        words = [w for w in normalized.split() if w]
        word_set = set(words)

        # Default to strong edits unless the user clearly asks for softer/romantic.
        intensity = 3
        intensity_hits = sum(1 for word in INTENSITY_KEYWORDS if word in text)
        if intensity_hits >= 2:
            intensity = 3
        elif intensity_hits == 1:
            intensity = 2
        if "max effects" in text or "all effects" in text:
            intensity = 3
        romantic_intent = any(t in text for t in ("romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"))
        soft_intent = any(t in text for t in ("soft", "simple", "clean", "minimal", "smooth", "subtle"))
        sad_dark_intent = any(t in text for t in ("sad", "heartbreak", "dark", "villain", "shadow"))
        if intensity_hits == 0 and romantic_intent and not sad_dark_intent:
            intensity = 1 if soft_intent else 2

        profile = {
            "intensity": intensity,
            "shake": False,
            "zoom": False,
            "flash": False,
            "glow": False,
            "glitch": False,
            "motion_blur": False,
            "grain": False,
            "vignette": False,
            "edge_glow": False,
            "speed_ramp": False,
            "slow_motion": False,
            "fast_motion": False,
            "reverse": False,
            "boomerang": False,
            "anime_tone": False,
            "ken_burns": mood in {"dramatic", "sad", "chill"},
            "chroma_pulse": False,
            "color_grade": False,
            "character_focus": mood in {"dramatic", "sad"},
            "black_white": False,
            "no_color_grade": False,
            "fire_lines": False,
            "overlap_transitions": False,
            "particles": False,
            "energy_aura": False,
            "turbulent_displace": False,
            "posterize_time": False,
            "mask_transition": False,
            "blend_overlay": False,
            "mask_roto": False,
            "chroma_key": False,
            "difference_matte": False,
            "object_inpaint": False,
            "pro_edit_pack": False,
            "auto_timefx": False,
            "explicit_effects": [],
            "loop_clips": False,
            "loop_count": 2,
            "selective_beats": True,
            "all_beats": False,
            "fast_render": False,
            "quality_priority": False,
            "smart_music": True,
            "no_phonk": False,
            "no_music": False,
            "simple_mode": False,
        }

        for key, terms in STYLE_KEYWORDS.items():
            negated = _is_style_negated_fallback(normalized, terms)
            if negated:
                profile[key] = False
                continue
            if any(w in text for w in terms):
                profile[key] = True
                profile["explicit_effects"].append(key)

        has_black_word = any(w in word_set for w in {"black", "balck", "blk"})
        has_white_word = any(w in word_set for w in {"white", "wht"})
        if has_black_word and has_white_word:
            profile["black_white"] = True

        color_words = {"color", "colors", "colour", "colours", "grade", "grading", "saturation", "saturated", "lut"}
        neg_words = {"no", "not", "dont", "don", "t", "without", "avoid"}
        change_words = {"change", "alter", "grade", "grading", "modify", "touch"}
        grade_words = {"grade", "grading", "cinematic", "vibrant", "saturation", "saturated", "lut", "boost"}
        if (
            any(w in word_set for w in color_words)
            and any(w in word_set for w in neg_words)
            and any(w in word_set for w in change_words)
        ):
            profile["no_color_grade"] = True
        if (
            any(w in word_set for w in color_words)
            and any(w in word_set for w in grade_words)
            and (not profile["no_color_grade"])
        ):
            profile["color_grade"] = True
        if re.search(r"\b(?:no|without|dont|don't|do not|avoid)\s+(?:any\s+)?phonk\b", normalized):
            profile["no_phonk"] = True
        if re.search(r"\b(?:no|without|dont|don't|do not|avoid)\s+(?:any\s+)?music\b", normalized):
            profile["no_music"] = True

        all_fx_request = (
            "all effects" in normalized
            or "everything" in normalized
            or "use everything" in normalized
            or "add everything" in normalized
            or "use all" in normalized
            or "all effects and transitions" in normalized
        )
        if all_fx_request:
            for k in (
                "glow",
                "glitch",
                "motion_blur",
                "grain",
                "vignette",
                "edge_glow",
                "speed_ramp",
                "slow_motion",
                "fast_motion",
                "ken_burns",
                "chroma_pulse",
                "color_grade",
                "fire_lines",
                "overlap_transitions",
                "particles",
                "energy_aura",
                "turbulent_displace",
                "posterize_time",
                "mask_transition",
                "blend_overlay",
                "mask_roto",
                "chroma_key",
                "difference_matte",
                "object_inpaint",
                "pro_edit_pack",
                "auto_timefx",
            ):
                profile[k] = True
                if k not in profile["explicit_effects"]:
                    profile["explicit_effects"].append(k)
            profile["heavy_effects_engine"] = True
        if (
            any(w in text for w in ("boundary", "boundaries", "outline"))
            and any(w in text for w in ("character", "charecter"))
            and any(w in text for w in ("highlight", "highlighting"))
        ):
            profile["fire_lines"] = True

        if "clean" in text or "minimal" in text or "simple" in text:
            profile["simple_mode"] = True
            for k in STYLE_KEYWORDS:
                profile[k] = False
            profile["intensity"] = 1

        if profile["intensity"] >= 3:
            for k in STYLE_KEYWORDS:
                if k in {
                    "black_white",
                    "no_color_grade",
                    "color_grade",
                    "particles",
                    "energy_aura",
                    "turbulent_displace",
                    "posterize_time",
                    "mask_transition",
                    "blend_overlay",
                    "mask_roto",
                    "chroma_key",
                    "difference_matte",
                    "object_inpaint",
                    "pro_edit_pack",
                    "selective_beats",
                    "all_beats",
                    "fast_render",
                    "quality_priority",
                    "smart_music",
                    "no_phonk",
                    "no_music",
                }:
                    continue
                profile[k] = True
        elif profile["intensity"] == 2:
            profile["motion_blur"] = profile["motion_blur"] or mood in {"hype", "dramatic"}
            profile["grain"] = profile["grain"] or mood in {"dark"}
            profile["vignette"] = profile["vignette"] or mood in {"dark", "sad"}
            profile["anime_tone"] = profile["anime_tone"] or mood in {"dramatic"}
            profile["character_focus"] = profile["character_focus"] or mood in {"dramatic", "sad"}
        if profile.get("black_white", False):
            profile["no_color_grade"] = True
            profile["color_grade"] = False
        if (not profile.get("black_white", False)) and (not profile.get("color_grade", False)):
            profile["no_color_grade"] = True
        if profile.get("no_color_grade", False):
            profile["color_grade"] = False
            profile["anime_tone"] = False
            profile["chroma_pulse"] = False
            profile["vignette"] = False
            profile["glow"] = False

        if profile.get("pro_edit_pack", False):
            for k in (
                "shake",
                "zoom",
                "flash",
                "glow",
                "glitch",
                "motion_blur",
                "grain",
                "edge_glow",
                "speed_ramp",
                "slow_motion",
                "fast_motion",
                "reverse",
                "boomerang",
                "ken_burns",
                "overlap_transitions",
                "particles",
                "energy_aura",
                "turbulent_displace",
                "posterize_time",
                "mask_transition",
                "blend_overlay",
                "mask_roto",
                "chroma_key",
                "fire_lines",
                "character_focus",
                "difference_matte",
                "object_inpaint",
            ):
                profile[k] = True
            profile["intensity"] = max(3, int(profile.get("intensity", 1)))
            profile["auto_timefx"] = True
            if not profile.get("all_beats", False):
                profile["selective_beats"] = True
                profile["all_beats"] = False

        if profile.get("all_beats", False):
            profile["selective_beats"] = False
        repeat_count = 2
        m = re.search(r"\b([2-4])\s*x\b", normalized) or re.search(r"\b([2-4])\s*times\b", normalized)
        if m:
            try:
                repeat_count = int(m.group(1))
            except Exception:
                repeat_count = 2
        elif any(k in normalized for k in ("triple", "thrice", "3x", "3 times")):
            repeat_count = 3
        elif any(k in normalized for k in ("quad", "4x", "4 times")):
            repeat_count = 4
        profile["loop_count"] = int(np.clip(repeat_count, 2, 4))
        if profile.get("no_music", False):
            profile["no_phonk"] = True
            profile["smart_music"] = False
        if profile.get("no_phonk", False):
            profile["smart_music"] = False
        return apply_personality_to_style(profile, infer_editor_personality(prompt, mood))


def build_fallback_tracks(mood: str) -> list[Track]:
    fallback_seed = FALLBACK_TRACKS_BY_MOOD.get(mood, FALLBACK_TRACKS_BY_MOOD["chill"])
    tracks: list[Track] = []
    for idx, item in enumerate(fallback_seed, start=1):
        tracks.append(
            Track(
                id=-(idx),
                title=item["title"],
                tags=f"{mood}, phonk, fallback",
                duration_sec=item["duration_sec"],
                preview_url="",
                audio_url="",
                user="archive-fallback",
                source="fallback",
            )
        )
    return tracks


def _is_phonk_like_track(track_path: Path) -> bool:
    rel = _relative_track_key(track_path).lower()
    name = track_path.name.lower()
    return ("phonk" in rel) or ("phonk" in name)


def list_local_phonk_files(*, allow_phonk: bool = True) -> list[Path]:
    tracks: list[Path] = []
    for p in PHONK_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
            if (not allow_phonk) and _is_phonk_like_track(p):
                continue
            tracks.append(p)
    return sorted(tracks, key=lambda x: str(x.relative_to(PHONK_DIR)).lower())


def _cleanup_old_render_outputs(keep_names: set[str] | None = None) -> None:
    prune_files(
        OUTPUT_DIR,
        "render_*.mp4",
        max_files=MAX_RETAINED_RENDERS,
        max_age_seconds=RENDER_RETENTION_SECONDS,
        keep_names=keep_names or set(),
    )


def _relative_track_key(track_path: Path) -> str:
    try:
        return track_path.relative_to(PHONK_DIR).as_posix()
    except Exception:
        return track_path.name


def infer_track_mood_from_name(filename: str) -> str:
    name = filename.replace("\\", "/").lower()
    scores = {m: 0 for m in MOOD_KEYWORDS}
    for mood, words in MOOD_KEYWORDS.items():
        for w in words:
            if w in name:
                scores[mood] += 1

    romantic_terms = ("romantic", "romance", "pleasant", "love", "sweet", "cute", "wholesome", "gentle", "tender")
    sad_terms = ("sad", "heartbreak", "cry", "tears", "melancholy", "tragic", "dukhi", "udaas", "dard")
    if any(t in name for t in romantic_terms) and not any(t in name for t in sad_terms):
        scores["chill"] += 2
        scores["sad"] = max(0, scores["sad"] - 1)

    max_score = max(scores.values()) if scores else 0
    if max_score <= 0:
        return "hype"

    tie_priority = ("chill", "dramatic", "hype", "dark", "sad")
    tied = [m for m, s in scores.items() if s == max_score]
    for mood in tie_priority:
        if mood in tied:
            return mood
    return tied[0]


def _infer_prompt_music_folder_hints(prompt: str) -> tuple[list[str], bool]:
    text = re.sub(r"\s+", " ", str(prompt or "").lower()).strip()
    if not text:
        return [], False

    hints: set[str] = set()
    romantic_terms = ("romantic", "romance", "love", "pleasant", "sweet", "cute", "gentle", "tender")
    happy_terms = ("happy", "joy", "cheerful", "uplifting", "smile", "positive", "fun")
    if any(t in text for t in romantic_terms):
        hints.add("romantic")
    if any(t in text for t in happy_terms):
        hints.add("happy")

    strict = any(
        phrase in text
        for phrase in (
            "from happy folder",
            "from romantic folder",
            "from happy music file",
            "from romantic music file",
            "happy folder",
            "romantic folder",
            "only happy",
            "only romantic",
            "happy music file",
            "romantic music file",
            "happy music",
            "romantic music",
            "happy song",
            "romantic song",
            "happy track",
            "romantic track",
            "use happy music",
            "use romantic music",
        )
    )
    return sorted(hints), strict


def _track_matches_folder_hints(track_path: Path, hints: list[str]) -> bool:
    rel = _relative_track_key(track_path).lower()
    if not hints:
        return True
    for hint in hints:
        if hint in rel:
            return True
        if hint == "romantic" and any(t in rel for t in ("romantic", "romance", "love", "pleasant", "sweet", "cute", "gentle", "tender")):
            return True
        if hint == "happy" and any(t in rel for t in ("happy", "joy", "cheerful", "uplift", "uplifting", "fun", "positive")):
            return True
    return False


def _track_signature(track_path: Path) -> str:
    try:
        return file_cache_signature(track_path)
    except Exception:
        # Keep cache key generation robust even if path metadata fails on Windows.
        return f"fallback:{track_path.name}:{len(str(track_path))}"


def _load_track_analysis_cache_from_disk() -> None:
    payload = load_json_cache(TRACK_ANALYSIS_CACHE_PATH)
    for key, value in payload.items():
        if isinstance(value, dict) and "sig" in value and "data" in value:
            TRACK_ANALYSIS_CACHE[key] = value


def _persist_track_analysis_cache(force: bool = False) -> None:
    global TRACK_ANALYSIS_CACHE_DIRTY
    if TRACK_ANALYSIS_CACHE_DIRTY <= 0:
        return
    if not force and TRACK_ANALYSIS_CACHE_DIRTY < 4:
        return
    while len(TRACK_ANALYSIS_CACHE) > 512:
        TRACK_ANALYSIS_CACHE.pop(next(iter(TRACK_ANALYSIS_CACHE)), None)
    if write_json_cache(TRACK_ANALYSIS_CACHE_PATH, TRACK_ANALYSIS_CACHE):
        TRACK_ANALYSIS_CACHE_DIRTY = 0


def _load_recent_tracks() -> None:
    global RECENT_TRACKS
    if RECENT_TRACKS:
        return
    try:
        if RECENT_TRACKS_PATH.exists():
            data = json.loads(RECENT_TRACKS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                RECENT_TRACKS = [str(x) for x in data if isinstance(x, str)]
    except Exception:
        RECENT_TRACKS = []


def _persist_recent_tracks() -> None:
    try:
        RECENT_TRACKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        keep_n = 40 if MUSIC_UPGRADE_AVAILABLE else 20
        RECENT_TRACKS_PATH.write_text(json.dumps(RECENT_TRACKS[:keep_n]), encoding="utf-8")
    except Exception:
        pass


def _fast_audio_load(track_path: Path, duration_sec: float) -> tuple[np.ndarray, int]:
    for res_type in AUDIO_RESAMPLE_CANDIDATES:
        try:
            kwargs = {
                "sr": 22050,
                "mono": True,
                "duration": duration_sec if duration_sec > 0 else None,
            }
            if res_type is not None:
                kwargs["res_type"] = res_type
            return librosa.load(str(track_path), **kwargs)
        except ModuleNotFoundError:
            continue
        except Exception:
            if res_type is None:
                raise
            continue
    return librosa.load(
        str(track_path),
        sr=22050,
        mono=True,
        duration=duration_sec if duration_sec > 0 else None,
    )


def _pick_video_codec() -> str:
    if _ENCODER_CACHE.get("checked", False):
        return str(_ENCODER_CACHE.get("codec", "libx264"))

    codec = "libx264"
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        try:
            probe = run_subprocess(
                [ffmpeg_cmd, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            encoders_text = f"{probe.stdout}\n{probe.stderr}".lower()
            if "h264_nvenc" in encoders_text:
                codec = "h264_nvenc"
            elif "h264_qsv" in encoders_text:
                codec = "h264_qsv"
            elif "h264_amf" in encoders_text:
                codec = "h264_amf"
        except Exception:
            codec = "libx264"

    _ENCODER_CACHE["checked"] = True
    _ENCODER_CACHE["codec"] = codec
    return codec


def _can_use_ffmpeg_encoder(codec: str) -> bool:
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        return False
    try:
        probe = run_subprocess(
            [
                ffmpeg_cmd,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=128x72:rate=24:duration=0.25",
                "-an",
                "-c:v",
                codec,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return probe.ok
    except Exception:
        return False


def _pick_best_working_video_codec() -> str:
    if _ENCODER_CACHE.get("working_checked", False):
        return str(_ENCODER_CACHE.get("working_codec", "libx264"))

    codec = _pick_video_codec()
    if codec == "libx264":
        _ENCODER_CACHE["working_checked"] = True
        _ENCODER_CACHE["working_codec"] = codec
        return codec
    if _can_use_ffmpeg_encoder(codec):
        _ENCODER_CACHE["working_checked"] = True
        _ENCODER_CACHE["working_codec"] = codec
        return codec
    _ENCODER_CACHE["working_checked"] = True
    _ENCODER_CACHE["working_codec"] = "libx264"
    return "libx264"


def _mood_grade_params_for_ffmpeg(mood: str, intensity: int) -> dict[str, float]:
    base = {
        "hype": {"sat": 1.12, "cont": 1.04, "gamma": 0.99},
        "dark": {"sat": 0.98, "cont": 1.05, "gamma": 1.01},
        "dramatic": {"sat": 1.1, "cont": 1.045, "gamma": 1.0},
        "sad": {"sat": 0.97, "cont": 0.995, "gamma": 1.01},
        "chill": {"sat": 1.05, "cont": 1.01, "gamma": 1.0},
    }.get(mood, {"sat": 1.04, "cont": 1.0, "gamma": 1.0})
    boost = 1.0 + (0.008 * max(0, int(intensity) - 1))
    return {
        "sat": float(np.clip(base["sat"] * boost, 0.96, 1.16)),
        "cont": float(np.clip(base["cont"] * boost, 0.96, 1.09)),
        "gamma": float(np.clip(base["gamma"], 0.97, 1.04)),
    }


def _build_ffmpeg_post_filter(style: dict[str, Any], mood: str, intensity: int) -> str:
    filters: list[str] = []
    no_color_mode = bool(style.get("no_color_grade", False))
    if bool(style.get("color_grade", False)):
        if MUSIC_UPGRADE_AVAILABLE and build_cinematic_grade_ffmpeg:
            filters.append(build_cinematic_grade_ffmpeg(mood, intensity))
        else:
            p = _mood_grade_params_for_ffmpeg(mood, intensity)
            eq_filter = f"eq=contrast={p['cont']:.4f}:saturation={p['sat']:.4f}:gamma={p['gamma']:.4f}"
            if mood == "hype":
                filters.append(f"{eq_filter},colorbalance=rs=0.012:bs=-0.008")
            elif mood == "dark":
                filters.append(f"{eq_filter},colorbalance=bs=0.012:rs=-0.006")
            elif mood == "sad":
                filters.append(f"{eq_filter},colorbalance=bs=0.01")
            else:
                filters.append(eq_filter)
    if bool(style.get("grain", False)):
        grain_strength = int(np.clip(5 + (2 * max(1, intensity)), 5, 10))
        filters.append(f"noise=alls={grain_strength}:allf=t")
    if bool(style.get("vignette", False)) and (not no_color_mode):
        filters.append("vignette=PI/6")
    if bool(style.get("edge_glow", False)):
        # Fast edge accent approximation in ffmpeg.
        filters.append("unsharp=5:5:0.75:3:3:0")

    if not filters:
        return ""
    return ",".join(filters)


def _run_with_timeout(stage_name: str, fn: Any, timeout_sec: int, logger_obj: logging.Logger) -> Any:
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        result = future.result(timeout=max(1, int(timeout_sec)))
        elapsed = time.perf_counter() - started
        logger_obj.info("[ARCHIVE] %s completed in %.2fs", stage_name, elapsed)
        return result
    except FutureTimeoutError:
        elapsed = time.perf_counter() - started
        future.cancel()
        logger_obj.warning("[ARCHIVE] %s exceeded %.0fs; falling back", stage_name, elapsed)
        return {"ok": False, "reason": f"{stage_name}_timeout", "elapsed_sec": round(elapsed, 2)}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_ffmpeg_postprocess(
    input_path: Path,
    output_path: Path,
    filter_chain: str,
    *,
    threads: int,
) -> bool:
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd or not filter_chain:
        return False
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(max(1, int(threads))),
        *ffmpeg_hwaccel_input_args(),
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        run = run_subprocess(cmd, capture_output=True, text=True, timeout=ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC)
        return run.ok and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _run_cpp_effects_preset(
    input_path: Path,
    output_path: Path,
    preset: str,
    *,
    timeout_sec: int = ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC,
) -> bool:
    if not CPP_EFFECTS_EXE:
        return False
    exe_path = Path(CPP_EFFECTS_EXE)
    if not exe_path.exists():
        return False
    try:
        cmd = [
            str(exe_path),
            "--preset",
            preset,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        # Bridge the C++ logs to the terminal so the user can see the "Master Brain" thinking
        result = run_subprocess(cmd, text=True, timeout=timeout_sec)
        return result.ok and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _run_cpp_frame_engine_render(
    input_path: Path,
    output_path: Path,
    mood: str,
    intensity: int,
    *,
    timeout_sec: int = ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC,
) -> bool:
    if not CPP_FRAME_ENGINE_EXE:
        return False
    exe_path = Path(CPP_FRAME_ENGINE_EXE)
    if not exe_path.exists():
        return False
    mood_to_effect = {
        "hype": "impact_flash_combo",
        "dark": "dark_surge",
        "dramatic": "energy_aura",
        "sad": "emotional_peak",
        "chill": "atmospheric_grade",
    }
    mood_to_anim = {
        "hype": "camera_slam",
        "dark": "handheld_shake",
        "dramatic": "zoom_pulse",
        "sad": "subtle_drift",
        "chill": "ken_burns_slow",
    }
    fx = mood_to_effect.get(mood, "cinematic_grade")
    anim = mood_to_anim.get(mood, "subtle_drift")
    intensity_norm = float(np.clip(intensity / 3.0, 0.4, 1.0))
    try:
        cmd = [
            str(exe_path),
            "render_clip",
            str(input_path),
            str(output_path),
            mood,
            f"{intensity_norm:.3f}",
            fx,
            anim,
            "0.75",
        ]
        result = run_subprocess(cmd, text=True, timeout=timeout_sec)
        return result.ok and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _decision_brain_effect_for_style(mood: str, style: dict[str, Any]) -> str:
    if bool(style.get("fire_lines", False)):
        return "fire_outline"
    if bool(style.get("energy_aura", False)):
        return "energy_aura"
    if bool(style.get("datamosh", False)):
        return "datamosh"
    if bool(style.get("glitch", False)):
        return "vhs"
    if bool(style.get("speed_lines", False)) or bool(style.get("fast_motion", False)):
        return "speed_lines"
    if bool(style.get("edge_glow", False)):
        return "glow"
    if bool(style.get("particles", False)):
        return "god_rays"
    if mood == "dark":
        return "vhs"
    if mood == "sad":
        return "halation"
    if mood == "dramatic":
        return "anamorphic_flare"
    if mood == "hype":
        return "impact_frame"
    return "contrast_sat"


def _ffmpeg_filter_for_decision_effect(effect: str, mood: str, intensity: float) -> str:
    i = float(np.clip(intensity, 0.25, 1.0))
    base = {
        "hype": "eq=contrast=1.18:saturation=1.28:gamma=0.98",
        "dark": "eq=contrast=1.22:saturation=0.95:gamma=0.92,colorbalance=bs=0.03:rs=-0.02",
        "sad": "eq=contrast=1.08:saturation=0.88:gamma=1.03,colorbalance=bs=0.02",
        "dramatic": "eq=contrast=1.16:saturation=1.12:gamma=0.98",
    }.get(mood, "eq=contrast=1.10:saturation=1.08:gamma=1.0")
    if effect in {"fire_outline", "energy_aura", "god_rays", "glow", "halation", "anamorphic_flare"}:
        return f"{base},split[v1][v2];[v2]boxblur={int(8 + i * 14)}:2[glow];[v1][glow]blend=all_mode=screen:all_opacity={0.18 + i * 0.22:.2f},unsharp=5:5:{0.4 + i * 0.35:.2f}"
    if effect in {"vhs", "datamosh"}:
        return f"{base},rgbashift=rh={int(2 + i * 5)}:bh=-{int(2 + i * 5)},noise=alls={int(6 + i * 10)}:allf=t"
    if effect in {"speed_lines", "impact_frame", "contrast_sat"}:
        return f"{base},unsharp=5:5:{0.45 + i * 0.45:.2f}:3:3:0.25,vignette=PI/6"
    return f"{base},unsharp=5:5:0.65"


def _run_decision_brain_single_effect(
    input_path: Path,
    output_path: Path,
    mood: str,
    style: dict[str, Any],
    intensity: int,
) -> bool:
    if not DECISION_BRAIN_AVAILABLE or not run_decision_single_effect or not CPP_FRAME_ENGINE_EXE:
        return False
    engine_path = Path(CPP_FRAME_ENGINE_EXE)
    if not engine_path.exists():
        return False
    try:
        effect = _decision_brain_effect_for_style(mood, style)
        intensity_norm = float(np.clip(max(float(intensity) / 3.0, 0.55), 0.35, 1.0))
        ok = run_decision_single_effect(
            str(input_path),
            str(output_path),
            effect,
            intensity_norm,
            mood,
            str(engine_path),
            crf=18,
            preset="veryfast",
        )
        if bool(ok) and output_path.exists() and output_path.stat().st_size > 0:
            return True
        output_path.unlink(missing_ok=True)
        fallback_filter = _ffmpeg_filter_for_decision_effect(effect, mood, intensity_norm)
        return _run_ffmpeg_postprocess(input_path, output_path, fallback_filter, threads=2)
    except Exception:
        return False


def _ai_editor_fallback_filter(mood: str) -> str:
    grade = "eq=contrast=1.15:saturation=1.35:brightness=0.03,curves=r='0/0 0.5/0.58 1/1':b='0/0 0.5/0.48 1/1':g='0/0 0.5/0.52 1/1'"
    if mood == "dark":
        grade = "eq=contrast=1.22:saturation=1.02:brightness=-0.02,curves=r='0/0 0.5/0.46 1/1':b='0/0 0.5/0.58 1/1'"
    elif mood == "sad":
        grade = "eq=contrast=1.08:saturation=0.9:brightness=0.02,curves=b='0/0 0.5/0.56 1/1'"
    return f"scale=1440:-2,{grade},tmix=frames=2:weights='1 1',unsharp=3:3:0.8,format=yuv420p"


def _run_ai_editor_polish(
    input_path: Path,
    output_path: Path,
    mood: str,
    *,
    timeout_sec: int = ARCHIVE_EXTERNAL_EFFECT_TIMEOUT_SEC,
) -> bool:
    if not AI_EDITOR_EXE:
        return False
    exe_path = Path(AI_EDITOR_EXE)
    if not exe_path.exists():
        return False
    preset = _cpp_preset_for_mood(mood)
    try:
        result = run_subprocess(
            [
                str(exe_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--preset",
                preset,
            ],
            text=True,
            timeout=timeout_sec,
        )
        if result.ok and output_path.exists() and output_path.stat().st_size > 0:
            return True
        output_path.unlink(missing_ok=True)
        return _run_ffmpeg_postprocess(input_path, output_path, _ai_editor_fallback_filter(mood), threads=2)
    except Exception:
        output_path.unlink(missing_ok=True)
        return _run_ffmpeg_postprocess(input_path, output_path, _ai_editor_fallback_filter(mood), threads=2)


def _replace_video_if_valid(current_path: Path, candidate_path: Path) -> bool:
    if not candidate_path.exists() or candidate_path.stat().st_size <= 0:
        candidate_path.unlink(missing_ok=True)
        return False
    current_path.unlink(missing_ok=True)
    candidate_path.replace(current_path)
    return True


def _apply_external_brain_polish(
    output_path: Path,
    mood: str,
    style: dict[str, Any],
    intensity: int,
    *,
    logger_obj: logging.Logger | None = None,
    request_id: str = "",
) -> dict[str, bool]:
    """
    Let every installed render brain contribute, but keep each pass optional.
    A failed polish layer should never throw away the render the user already has.
    """
    results = {
        "frame_engine": False,
        "master_effects": False,
        "decision_brain": False,
        "ai_editor": False,
        "python_effects": False,
    }
    rid = request_id or "-"

    if CPP_FRAME_ENGINE_EXE or CPP_EFFECTS_EXE:
        try:
            fx_out = output_path.with_name(f"{output_path.stem}_cppfx.mp4")
            if CPP_FRAME_ENGINE_EXE:
                if logger_obj:
                    logger_obj.info("[%s] Handing off to FRAME ENGINE for Elite VFX Pass...", rid)
                results["frame_engine"] = _run_cpp_frame_engine_render(output_path, fx_out, mood, intensity)
            if (not results["frame_engine"]) and CPP_EFFECTS_EXE:
                preset = _cpp_preset_for_mood(mood)
                if logger_obj:
                    logger_obj.info("[%s] Handing off to MASTER BRAIN for Elite VFX Pass...", rid)
                results["master_effects"] = _run_cpp_effects_preset(output_path, fx_out, preset)
            if (results["frame_engine"] or results["master_effects"]):
                _replace_video_if_valid(output_path, fx_out)
            else:
                fx_out.unlink(missing_ok=True)
        except Exception:
            pass

    wants_decision_pass = (
        DECISION_BRAIN_AVAILABLE
        and not bool(style.get("simple_mode", False))
        and (intensity >= 2 or bool(style.get("pro_edit_pack", False)) or bool(style.get("heavy_effects_engine", False)))
    )
    if wants_decision_pass:
        try:
            decision_out = output_path.with_name(f"{output_path.stem}_decisionfx.mp4")
            if logger_obj:
                logger_obj.info("[%s] Handing off to DECISION BRAIN single-effect pass...", rid)
            results["decision_brain"] = _run_decision_brain_single_effect(
                output_path,
                decision_out,
                mood,
                style,
                intensity,
            )
            if results["decision_brain"]:
                _replace_video_if_valid(output_path, decision_out)
            else:
                decision_out.unlink(missing_ok=True)
        except Exception:
            pass

    wants_ai_editor = (
        bool(AI_EDITOR_EXE)
        and not bool(style.get("simple_mode", False))
        and (bool(style.get("pro_edit_pack", False)) or int(style.get("intensity", intensity)) >= 3)
    )
    if wants_ai_editor:
        try:
            ai_out = output_path.with_name(f"{output_path.stem}_aieditor.mp4")
            if logger_obj:
                logger_obj.info("[%s] Handing off to AI_EDITOR cortex pass...", rid)
            results["ai_editor"] = _run_ai_editor_polish(output_path, ai_out, mood)
            if results["ai_editor"]:
                _replace_video_if_valid(output_path, ai_out)
            else:
                ai_out.unlink(missing_ok=True)
        except Exception:
            pass

    if EFFECTS_ENGINE_AVAILABLE and apply_effects_to_clip and get_effects_for_style:
        try:
            fx_list = get_effects_for_style(style, mood, intensity)
            if not fx_list:
                fx_list = _default_always_on_effects(mood)
            else:
                fx_list = list(dict.fromkeys(fx_list + _default_always_on_effects(mood)))
            fx_out = output_path.with_name(f"{output_path.stem}_fx.mp4")
            fx_result = apply_effects_to_clip(
                input_path=output_path,
                output_path=fx_out,
                effects=fx_list,
                intensity=float(np.clip(max(intensity / 3.0, 0.65), 0.4, 1.0)),
                style_profile=style,
            )
            results["python_effects"] = bool(fx_result.get("ok")) if isinstance(fx_result, dict) else False
            if results["python_effects"]:
                _replace_video_if_valid(output_path, fx_out)
            else:
                fx_out.unlink(missing_ok=True)
        except Exception:
            pass

    return results


def _cpp_preset_for_mood(mood: str) -> str:
    return {
        "hype": "hype_phonk",
        "dark": "dark_villain",
        "dramatic": "anime_cinematic",
        "sad": "sad_emotional",
        "chill": "anime_cinematic",
    }.get(mood, "hype_phonk")


def _default_always_on_effects(mood: str) -> list[str]:
    base = {
        "hype": ["teal_orange", "bloom", "rgb_split", "vignette", "film_grain"],
        "dark": ["anime_vibrant", "bloom", "rgb_split", "vignette", "film_grain"],
        "dramatic": ["teal_orange", "bloom", "vignette", "film_grain"],
        "sad": ["anime_vibrant", "bloom", "vignette", "film_grain"],
        "chill": ["anime_vibrant", "bloom", "vignette", "film_grain"],
    }
    return list(base.get(mood, ["teal_orange", "bloom", "vignette", "film_grain"]))


def _max_render_pixels(style: dict[str, Any], *, fallback_pixels: int = 3840 * 2160) -> int:
    """
    Target output ceiling for timeline renders.
    Default now preserves up to 4K to avoid unintended 720p/1080p downscales.
    """
    if bool(style.get("fast_render", False)) and (not bool(style.get("quality_priority", False))):
        # Keep fast mode lighter but still above 1080p.
        return 2560 * 1440
    return fallback_pixels


def _apply_performance_defaults(prompt: str, style_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Prefer the best edit path by default: brain/C++ effects on selected hero moments,
    while keeping the fast/global renderer out of effect-heavy renders.
    """
    profile = dict(style_profile or {})
    if not ARCHIVE_SPEED_MODE:
        profile["brain_render"] = ARCHIVE_DEFAULT_BRAIN_RENDER
        profile["fast_render"] = False
        profile["disable_fast_renderer"] = True
        profile["quality_priority"] = True
        profile["advanced_compositor"] = True
        profile["cpp_render_clip"] = True
        profile["selective_beats"] = True
        profile["all_beats"] = False
        profile["heavy_effects_engine"] = True
        if not bool(profile.get("simple_mode", False)):
            profile["intensity"] = max(2, int(profile.get("intensity", 2) or 2))
        return profile

    text = (prompt or "").lower()
    heavy_requested = any(
        token in text
        for token in (
            "all effects",
            "everything",
            "max effects",
            "maximum quality",
            "highest quality",
            "quality priority",
            "pro pack",
            "pro edit",
            "brain render",
            "elite brain",
            "master brain",
            "4k",
        )
    )
    fast_requested = any(
        token in text
        for token in (
            "fast render",
            "quick render",
            "minimum time",
            "low time",
            "faster",
            "speed mode",
        )
    )

    if ARCHIVE_DEFAULT_FAST_RENDER or fast_requested:
        profile["fast_render"] = True
        profile["disable_fast_renderer"] = False

    if not heavy_requested:
        explicit_effects = set(profile.get("explicit_effects", []) or [])
        if int(profile.get("intensity", 2) or 2) > 2 and not any(
            token in text for token in ("intense", "insane", "hard", "aggressive", "maximum", "all effects")
        ):
            profile["intensity"] = 2
        for key in (
            "overlap_transitions",
            "mask_transition",
            "loop_clips",
            "loop_ending",
            "particles",
            "energy_aura",
            "turbulent",
            "turbulent_displace",
            "blend_overlay",
            "mask_roto",
            "chroma_key",
            "difference_matte",
            "object_inpaint",
            "fire_lines",
            "boomerang",
            "pro_edit_pack",
        ):
            if key not in explicit_effects:
                profile[key] = False
        profile["quality_priority"] = False
        profile["brain_render"] = ARCHIVE_DEFAULT_BRAIN_RENDER
        profile["heavy_effects_engine"] = False
    else:
        profile["brain_render"] = ARCHIVE_DEFAULT_BRAIN_RENDER or any(
            token in text for token in ("brain render", "elite brain", "master brain")
        )

    return profile


def _enforce_high_energy_style(prompt: str, mood: str, style_profile: dict[str, Any]) -> dict[str, Any]:
    """
    Prevent accidental low-intensity/simple-mode outputs for high-energy intents.
    """
    profile = dict(style_profile or {})
    text = (prompt or "").lower()
    high_energy_terms = (
        "phonk",
        "dramatic",
        "cinematic",
        "epic",
        "villain",
        "dark",
        "hype",
        "intense",
        "hard",
        "aggressive",
        "impact",
    )
    soft_only_terms = ("romantic", "love", "cute", "sweet", "tender", "gentle")
    has_high_energy_intent = mood in {"hype", "dark", "dramatic"} or any(t in text for t in high_energy_terms)
    has_soft_only_intent = any(t in text for t in soft_only_terms) and (not any(t in text for t in ("dark", "villain", "phonk", "dramatic")))

    if has_high_energy_intent and not has_soft_only_intent:
        if bool(profile.get("simple_mode", False)):
            profile["simple_mode"] = False
        profile["intensity"] = max(2, int(profile.get("intensity", 1)))
        # Ensure visible baseline edit language for high-energy tracks.
        for key in ("color_grade", "motion_blur", "vignette"):
            profile[key] = True
        if int(profile.get("intensity", 2)) >= 3:
            for key in ("shake", "zoom", "flash", "glow", "chroma_pulse"):
                profile[key] = True
    return profile


_load_track_analysis_cache_from_disk()


def _tempo_value(tempo: Any) -> float:
    if hasattr(tempo, "__len__"):
        return float(tempo[0]) if len(tempo) else 0.0
    return float(tempo)


def _safe_resolve_str(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _compress_cut_points(points: list[float], min_gap: float, max_points: int) -> list[float]:
    if not points:
        return [0.0]

    ordered = sorted(set(float(p) for p in points))
    compact = [ordered[0]]
    for p in ordered[1:]:
        if p - compact[-1] >= min_gap:
            compact.append(p)

    if compact[-1] != ordered[-1]:
        compact.append(ordered[-1])

    if len(compact) <= max_points:
        return compact

    max_points = max(3, max_points)
    core = compact[1:-1]
    stride = max(1, math.ceil(len(core) / (max_points - 2)))
    reduced_core = core[::stride]
    return [compact[0], *reduced_core[: max_points - 2], compact[-1]]


def infer_prompt_music_profile(prompt: str, mood: str) -> dict[str, float]:
    try:
        return infer_music_profile_from_prompt(prompt, mood)
    except Exception:
        text = (prompt or "").lower()
        energy = {
            "hype": 0.86,
            "dark": 0.7,
            "dramatic": 0.68,
            "sad": 0.38,
            "chill": 0.3,
        }.get(mood, 0.6)
        onset = {
            "hype": 0.84,
            "dark": 0.62,
            "dramatic": 0.58,
            "sad": 0.3,
            "chill": 0.26,
        }.get(mood, 0.5)
        brightness = {
            "hype": 0.62,
            "dark": 0.3,
            "dramatic": 0.5,
            "sad": 0.36,
            "chill": 0.48,
        }.get(mood, 0.5)

        if any(k in text for k in ["aggressive", "hard", "intense", "fight", "crazy"]):
            energy += 0.14
            onset += 0.12
        if any(k in text for k in ["emotional", "sad", "slow", "heartbreak"]):
            energy -= 0.16
            onset -= 0.12
        if any(k in text for k in ["dark", "villain", "shadow"]):
            brightness -= 0.2
        if any(k in text for k in ["uplifting", "hope", "bright"]):
            brightness += 0.18
        if any(k in text for k in ["romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"]) and not any(
            k in text for k in ["sad", "heartbreak", "cry", "tears", "melancholy", "tragic"]
        ):
            energy -= 0.12
            onset -= 0.14
            brightness += 0.1

        return {
            "target_energy": float(np.clip(energy, 0.0, 1.0)),
            "target_onset": float(np.clip(onset, 0.0, 1.0)),
            "target_brightness": float(np.clip(brightness, 0.0, 1.0)),
        }


def _infer_prompt_music_mood(prompt: str, default_mood: str) -> tuple[str, bool]:
    text = re.sub(r"\s+", " ", str(prompt or "").lower()).strip()
    if not text:
        return default_mood, False
    no_phonk_intent = any(
        phrase in text
        for phrase in (
            "no phonk",
            "without phonk",
            "dont add phonk",
            "don't add phonk",
            "do not add phonk",
            "avoid phonk",
            "not phonk",
        )
    )

    romantic_terms = ("romantic", "romance", "pleasant", "love", "sweet", "cute", "wholesome", "gentle", "tender")
    happy_terms = ("happy", "joy", "cheerful", "uplifting", "smile", "positive", "fun", "feel good")
    sad_terms = ("sad", "heartbreak", "cry", "tears", "melancholy", "tragic", "dukhi", "udaas", "dard")
    dark_terms = ("dark", "villain", "shadow", "horror", "noir", "sinister")
    hype_terms = ("hype", "hard", "aggressive", "fight", "drift", "energy", "adrenaline")
    if not no_phonk_intent:
        hype_terms = hype_terms + ("phonk",)
    dramatic_terms = ("dramatic", "cinematic", "epic", "story")

    romantic_hit = any(t in text for t in romantic_terms)
    happy_hit = any(t in text for t in happy_terms)
    sad_hit = any(t in text for t in sad_terms)
    dark_hit = any(t in text for t in dark_terms)
    hype_hit = any(t in text for t in hype_terms)
    dramatic_hit = any(t in text for t in dramatic_terms)

    if romantic_hit and (not sad_hit) and (not dark_hit):
        return "chill", True
    if happy_hit and (not sad_hit) and (not dark_hit):
        return "chill", True
    if sad_hit and not romantic_hit:
        return "sad", True
    if dark_hit:
        return "dark", True
    if dramatic_hit and not hype_hit:
        return "dramatic", True
    if hype_hit:
        return "hype", True
    return default_mood, False


def _load_edit_brain_artifact() -> dict[str, Any] | None:
    if _EDIT_BRAIN_CACHE.get("loaded", False):
        return _EDIT_BRAIN_CACHE.get("artifact")

    _EDIT_BRAIN_CACHE["loaded"] = True
    if not JOBLIB_AVAILABLE:
        _EDIT_BRAIN_CACHE["artifact"] = None
        return None

    model_candidates = [
        ARTIFACT_DIR / "edit_brain_model.joblib",
        BASE_DIR / "edit_brain_model.joblib",
        BASE_DIR / "outputs" / "edit_brain_model.joblib",  # legacy location
        OUTPUT_DIR / "edit_brain_model.joblib",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        _EDIT_BRAIN_CACHE["artifact"] = None
        return None

    try:
        artifact = joblib.load(str(model_path))
        if not isinstance(artifact, dict) or "model" not in artifact or "meta" not in artifact:
            _EDIT_BRAIN_CACHE["artifact"] = None
            return None
        _EDIT_BRAIN_CACHE["artifact"] = artifact
        return artifact
    except Exception:
        _EDIT_BRAIN_CACHE["artifact"] = None
        return None


def _build_edit_brain_feature_vector(
    row: dict[str, Any],
    mood: str,
    archetype: str,
    intensity: int,
    meta: dict[str, Any],
) -> np.ndarray:
    numeric_features = list(meta.get("numeric_features", []))
    mood_vocab = list(meta.get("mood_vocab", []))
    archetype_vocab = list(meta.get("archetype_vocab", []))

    row_values = dict(row)
    row_values["intensity"] = intensity
    x_num = [float(row_values.get(col, 0.0)) for col in numeric_features]

    cat = [0.0] * (len(mood_vocab) + len(archetype_vocab))
    if mood in mood_vocab:
        cat[mood_vocab.index(mood)] = 1.0
    if archetype in archetype_vocab:
        cat[len(mood_vocab) + archetype_vocab.index(archetype)] = 1.0

    return np.array(x_num + cat, dtype=np.float32).reshape(1, -1)


def predict_effect_hints_for_segments(
    intelligence_rows: list[dict[str, Any]], mood: str, style: dict[str, Any]
) -> list[dict[str, bool]]:
    artifact = _load_edit_brain_artifact()
    if not artifact:
        return [{} for _ in intelligence_rows]

    model = artifact.get("model")
    meta = artifact.get("meta", {})
    targets = list(meta.get("target_columns", []))
    if not model or not targets:
        return [{} for _ in intelligence_rows]

    intensity = int(style.get("intensity", 1))
    if not intelligence_rows:
        return []

    features: list[np.ndarray] = []
    for row in intelligence_rows:
        archetype = str(row.get("archetype", "neutral"))
        x = _build_edit_brain_feature_vector(row, mood, archetype, intensity, meta)
        features.append(x[0])

    try:
        batch_x = np.vstack(features).astype(np.float32)
        pred = model.predict(batch_x)
    except Exception:
        return [{} for _ in intelligence_rows]

    hints: list[dict[str, bool]] = []
    selective_beats = bool(style.get("selective_beats", True)) and (not bool(style.get("all_beats", False)))
    heavy_hint_keys = {
        "zoom_applied",
        "shake_applied",
        "glitch_applied",
        "flash_applied",
        "chroma_pulse_applied",
    }
    for idx in range(len(intelligence_rows)):
        try:
            row_pred = pred[idx]
            pred_row = row_pred.tolist() if hasattr(row_pred, "tolist") else list(row_pred)
            hint_row = {targets[i]: bool(pred_row[i]) for i in range(min(len(targets), len(pred_row)))}
            row = intelligence_rows[idx] if idx < len(intelligence_rows) else {}
            is_peak = bool(row.get("is_peak_beat", False))
            impact_rank = float(row.get("impact_rank", 0.5))
            if selective_beats and (not is_peak) and impact_rank < 0.72:
                for k in heavy_hint_keys:
                    if k in hint_row:
                        hint_row[k] = False
            hints.append(hint_row)
        except Exception:
            hints.append({})
    return hints


def _annotate_peak_beats(intelligence_rows: list[dict[str, Any]], style: dict[str, Any]) -> list[dict[str, Any]]:
    if not intelligence_rows:
        return intelligence_rows

    all_beats = bool(style.get("all_beats", False))
    selective_beats = bool(style.get("selective_beats", True)) and (not all_beats)
    intensity = int(style.get("intensity", 1))
    fast_mode = bool(style.get("fast_render", False))
    quality_mode = bool(style.get("quality_priority", False))

    scores: list[float] = []
    for row in intelligence_rows:
        score = float(
            np.clip(
                (float(row.get("edit_priority", 0.5)) * 0.32)
                + (float(row.get("cut_priority", 0.5)) * 0.24)
                + (float(row.get("scene_change", 0.0)) * 0.16)
                + (float(row.get("motion", 0.0)) * 0.1)
                + (float(row.get("scene_energy", 0.0)) * 0.1)
                + (float(row.get("emotion_weight", 0.0)) * 0.08),
                0.0,
                1.0,
            )
        )
        scores.append(score)

    keep_ratio = 0.3 if intensity <= 1 else (0.34 if intensity == 2 else 0.38)
    if fast_mode:
        keep_ratio -= 0.06
    if quality_mode and (not fast_mode):
        keep_ratio += 0.06
    keep_ratio = float(np.clip(keep_ratio, 0.16, 0.62))

    threshold = float(np.quantile(scores, 1.0 - keep_ratio))
    max_score = max(scores) if scores else 1.0
    max_score = max(max_score, 1e-6)

    for idx, row in enumerate(intelligence_rows):
        score = scores[idx]
        prev_s = scores[idx - 1] if idx > 0 else score
        next_s = scores[idx + 1] if idx < len(scores) - 1 else score
        local_peak = (score >= prev_s) and (score >= next_s) and (score > 0.54)
        impact_rank = float(np.clip(score / max_score, 0.0, 1.0))

        if all_beats:
            is_peak = True
        elif selective_beats:
            is_peak = bool((score >= threshold and (local_peak or score > 0.62 or impact_rank > 0.72)))
        else:
            is_peak = bool(score >= max(0.48, threshold - 0.08))

        row["impact_score"] = round(score, 4)
        row["impact_rank"] = round(impact_rank, 4)
        row["is_peak_beat"] = bool(is_peak)

    return intelligence_rows


def build_global_edit_plan(
    intelligence_rows: list[dict[str, Any]],
    mood: str,
    style: dict[str, Any],
) -> dict[str, Any]:
    n = len(intelligence_rows)
    if n == 0:
        return {"phase": [], "heavy_allow": [], "energy_rank": [], "transition_after": [], "transition_hint": []}

    intensity = int(style.get("intensity", 1))
    pro_pack = bool(style.get("pro_edit_pack", False))
    cut_density = float(style.get("cut_density", 1.0))
    transition_density = float(style.get("transition_density", 1.0))
    energies = [
        float(
            np.clip(
                (float(row.get("scene_energy", 0.5)) * 0.52)
                + (float(row.get("cut_priority", row.get("visual_importance", 0.5))) * 0.3)
                + (float(row.get("emotion_weight", 0.0)) * 0.18),
                0.0,
                1.0,
            )
        )
        for row in intelligence_rows
    ]

    smooth: list[float] = []
    for i in range(n):
        window = energies[max(0, i - 1) : min(n, i + 2)]
        smooth.append(float(np.mean(window)))

    hi_quantile_base = 0.78 if intensity <= 1 else (0.74 if intensity == 2 else 0.7)
    hi_thr = float(np.quantile(smooth, float(np.clip(hi_quantile_base - ((cut_density - 1.0) * 0.06), 0.58, 0.86))))
    lo_thr = float(np.quantile(smooth, 0.36 if mood in {"sad", "chill"} else 0.3))

    phases: list[str] = []
    for i, val in enumerate(smooth):
        if val >= hi_thr:
            phase = "impact"
        elif val <= lo_thr:
            phase = "buildup"
        else:
            phase = "flow"
        phases.append(phase)

    intro_len = max(1, int(round(n * 0.14)))
    outro_len = max(1, int(round(n * 0.14)))
    for i in range(intro_len):
        if phases[i] != "impact":
            phases[i] = "intro"
    for i in range(n - outro_len, n):
        if phases[i] != "impact":
            phases[i] = "outro"

    ranks = np.argsort(np.argsort(smooth)).astype(np.float32)
    energy_rank = (ranks / max(1.0, float(n - 1))).tolist()

    heavy_quota = max(2, int(round(n * (0.22 + (0.03 * max(0, intensity - 1))) * float(np.clip(cut_density, 0.55, 1.5)))))
    if pro_pack:
        heavy_quota = min(n, heavy_quota + 1)

    ranked_indices = sorted(range(n), key=lambda i: smooth[i], reverse=True)
    heavy_allow = [False] * n
    for idx in ranked_indices[:heavy_quota]:
        heavy_allow[idx] = True
    for i, phase in enumerate(phases):
        if phase in {"intro", "outro"} and not pro_pack:
            heavy_allow[i] = False

    transition_after = [False] * n
    transition_hint = ["cut"] * n
    for i in range(n - 1):
        phase_change = phases[i] != phases[i + 1]
        scene_change = float(intelligence_rows[i].get("scene_change", 0.0))
        archetype = str(intelligence_rows[i].get("archetype", "neutral"))
        cut_priority = float(intelligence_rows[i].get("cut_priority", 0.0))
        transition_hint[i] = transition_hint_from_row(intelligence_rows[i])
        transition_threshold = 0.12 / float(np.clip(transition_density, 0.55, 1.6))
        if phase_change and (scene_change > transition_threshold or archetype in {"hard_cut", "action"} or cut_priority > 0.72):
            transition_after[i] = True
        elif pro_pack and (scene_change > 0.2 or archetype == "hard_cut" or cut_priority > 0.78):
            transition_after[i] = True

    return {
        "phase": phases,
        "heavy_allow": heavy_allow,
        "energy_rank": energy_rank,
        "transition_after": transition_after,
        "transition_hint": transition_hint,
    }


def select_signature_pack(mood: str, visual: dict[str, Any] | None, style: dict[str, Any]) -> list[str]:
    if bool(style.get("simple_mode", False)):
        return ["flow"]
    visual_mood = str((visual or {}).get("visual_mood", mood))
    if mood in {"hype"} or visual_mood in {"hype"}:
        return ["impact", "aura"]
    if mood in {"dark"} or visual_mood in {"dark"}:
        return ["glitch", "cinematic"]
    if mood in {"dramatic"} or visual_mood in {"dramatic"}:
        return ["cinematic", "impact"]
    if mood in {"sad", "chill"} or visual_mood in {"sad", "chill"}:
        return ["flow", "cinematic"]
    return ["cinematic", "impact"]


def analyze_video_unified(
    video_path: Path,
    *,
    max_frames_vibe: int = 140,
    max_frames_anime: int = 72,
    max_frames_explicit: int = 84,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "visual": {
                "visual_mood": "dramatic",
                "motion": 0.5,
                "brightness": 0.5,
                "saturation": 0.5,
                "contrast": 0.5,
                "desired_tempo": 130,
            },
            "anime_check": {
                "is_anime": True,
                "anime_confidence": 0.5,
                "real_confidence": 0.5,
                "method": "heuristic_fallback_open_failed",
                "reason": "open_failed",
            },
            "content_check": {
                "is_explicit": False,
                "confidence": 0.0,
                "method": "heuristic_fallback_open_failed",
                "reason": "open_failed",
            },
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = max(1, max_frames_vibe, max_frames_anime, max_frames_explicit)
    stride = max(1, frame_count // max_frames) if frame_count > 0 else 2

    motion_values: list[float] = []
    brightness_values: list[float] = []
    saturation_values: list[float] = []
    contrast_values: list[float] = []
    prev_gray_small: np.ndarray | None = None

    flat_values: list[float] = []
    edge_values: list[float] = []
    sat_values: list[float] = []
    texture_values: list[float] = []
    contrast_values_anime: list[float] = []

    suspicious_frames = 0
    romantic_like_frames = 0
    max_skin_ratio = 0.0
    max_region_ratio = 0.0
    max_face_area = 0.0

    i = 0
    sampled = 0
    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride != 0:
            continue

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        if len(brightness_values) < max_frames_vibe:
            brightness_values.append(float(np.mean(gray)) / 255.0)
            saturation_values.append(float(np.mean(hsv[:, :, 1])) / 255.0)
            contrast_values.append(float(np.std(gray)) / 128.0)
            if prev_gray_small is not None:
                diff = cv2.absdiff(gray, prev_gray_small)
                motion_values.append(float(np.mean(diff)) / 255.0)
            prev_gray_small = gray

        if len(flat_values) < max_frames_anime:
            detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
            detail_ratio = float(np.mean(detail > 8))
            flat_ratio = 1.0 - detail_ratio
            edges = cv2.Canny(gray, 70, 170)
            edge_density = float(np.mean(edges > 0))
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            texture = float(np.mean(np.abs(lap)) / 255.0)
            sat = float(np.mean(hsv[:, :, 1])) / 255.0
            cont = float(np.std(gray) / 128.0)
            flat_values.append(float(np.clip(flat_ratio, 0.0, 1.0)))
            edge_values.append(float(np.clip(edge_density, 0.0, 1.0)))
            texture_values.append(float(np.clip(texture, 0.0, 1.0)))
            sat_values.append(float(np.clip(sat, 0.0, 1.0)))
            contrast_values_anime.append(float(np.clip(cont, 0.0, 1.0)))

        if (len(flat_values) + len(brightness_values)) >= (max_frames_vibe + max_frames_anime):
            pass

        if sampled < max_frames_explicit:
            h, w = small.shape[:2]
            area = float(max(1, h * w))
            ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            skin_ycrcb = cv2.inRange(
                ycrcb,
                np.array([0, 133, 77], dtype=np.uint8),
                np.array([255, 173, 127], dtype=np.uint8),
            )
            skin_hsv = cv2.inRange(
                hsv,
                np.array([0, 20, 45], dtype=np.uint8),
                np.array([25, 200, 255], dtype=np.uint8),
            )
            skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
            skin_mask = cv2.morphologyEx(
                skin_mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            skin_mask = cv2.morphologyEx(
                skin_mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
                iterations=1,
            )
            skin_ratio = float(np.mean(skin_mask > 0))
            max_skin_ratio = max(max_skin_ratio, skin_ratio)

            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0.0
            for c in contours:
                largest_area = max(largest_area, float(cv2.contourArea(c)))
            largest_region_ratio = float(np.clip(largest_area / area, 0.0, 1.0))
            max_region_ratio = max(max_region_ratio, largest_region_ratio)

            face_area_ratio = 0.0
            if not FACE_CASCADE.empty():
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
                if len(faces) > 0:
                    face_area_ratio = float(sum(float(fw * fh) for (_x, _y, fw, fh) in faces) / area)
            max_face_area = max(max_face_area, face_area_ratio)

            edges = cv2.Canny(gray, 70, 170)
            edge_density = float(np.mean(edges > 0))
            texture = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))) / 255.0)

            high_skin = skin_ratio > 0.48 and largest_region_ratio > 0.34
            extreme_skin = skin_ratio > 0.62 and largest_region_ratio > 0.42
            low_structure = edge_density < 0.09 and texture < 0.11
            likely_romantic_kiss = (face_area_ratio > 0.05 and skin_ratio < 0.5) or (face_area_ratio > 0 and skin_ratio < 0.46)
            explicit_frame = bool(extreme_skin or (high_skin and low_structure and face_area_ratio < 0.12))
            if likely_romantic_kiss and skin_ratio < 0.5:
                explicit_frame = False
                romantic_like_frames += 1
            if explicit_frame:
                suspicious_frames += 1

        sampled += 1

    cap.release()

    if not brightness_values:
        visual = {
            "visual_mood": "dramatic",
            "motion": 0.5,
            "brightness": 0.5,
            "saturation": 0.5,
            "contrast": 0.5,
            "desired_tempo": 130,
        }
    else:
        motion = float(np.clip(np.mean(motion_values) if motion_values else 0.35, 0.0, 1.0))
        brightness = float(np.clip(np.mean(brightness_values), 0.0, 1.0))
        saturation = float(np.clip(np.mean(saturation_values), 0.0, 1.0))
        contrast = float(np.clip(np.mean(contrast_values), 0.0, 1.0))
        mood_scores = {
            "hype": (motion * 1.5) + (saturation * 0.7) + (contrast * 0.5),
            "dark": ((1.0 - brightness) * 1.4) + (contrast * 0.6),
            "sad": ((1.0 - motion) * 1.0) + ((1.0 - saturation) * 0.9) + ((1.0 - contrast) * 0.4),
            "dramatic": (contrast * 1.0) + (motion * 0.75) + ((1.0 - saturation) * 0.2),
            "chill": ((1.0 - motion) * 1.1) + (brightness * 0.55),
        }
        visual = {
            "visual_mood": max(mood_scores, key=mood_scores.get),
            "motion": round(motion, 3),
            "brightness": round(brightness, 3),
            "saturation": round(saturation, 3),
            "contrast": round(contrast, 3),
            "desired_tempo": int(np.clip(90 + (motion * 75) + (contrast * 18), 90, 180)),
        }

    if flat_values:
        flat = float(np.mean(flat_values))
        edge = float(np.mean(edge_values))
        texture = float(np.mean(texture_values))
        sat = float(np.mean(sat_values))
        contrast = float(np.mean(contrast_values_anime))
        anime_score = (
            (flat * 1.55)
            + (edge * 0.9)
            + (sat * 0.45)
            + (contrast * 0.2)
            - (texture * 1.22)
        )
        raw_conf = (anime_score - 0.16) / 0.42
        anime_conf = float(np.clip(raw_conf, 0.0, 1.0))
        real_conf = float(np.clip(1.0 - anime_conf, 0.0, 1.0))
        likely_real = bool((texture > 0.165 and flat < 0.43) or (texture > 0.19 and edge < 0.075))
        is_anime = bool((anime_conf >= 0.52) and (not likely_real))
        anime_check = {
            "is_anime": is_anime,
            "anime_confidence": round(anime_conf, 3),
            "real_confidence": round(real_conf, 3),
            "method": "opencv_heuristic_anime_gate_v1",
            "sampled_frames": len(flat_values),
            "flat_ratio": round(flat, 3),
            "edge_density": round(edge, 3),
            "texture": round(texture, 3),
            "saturation": round(sat, 3),
            "contrast": round(contrast, 3),
            "score": round(float(anime_score), 3),
        }
    else:
        anime_check = {
            "is_anime": True,
            "anime_confidence": 0.5,
            "real_confidence": 0.5,
            "method": "heuristic_fallback_no_frames",
            "reason": "no_frames",
        }

    if sampled <= 0:
        content_check = {
            "is_explicit": False,
            "confidence": 0.0,
            "method": "heuristic_fallback_no_frames",
            "reason": "no_frames",
        }
    else:
        ratio = float(suspicious_frames / max(1, sampled))
        trigger_frames = max(3, int(math.ceil(sampled * 0.16)))
        explicit_conf = float(np.clip((ratio - 0.08) / 0.32, 0.0, 1.0))
        is_explicit = bool(suspicious_frames >= trigger_frames and ratio >= 0.13)
        if romantic_like_frames >= suspicious_frames and ratio < 0.2:
            is_explicit = False
            explicit_conf = min(explicit_conf, 0.45)
        content_check = {
            "is_explicit": is_explicit,
            "confidence": round(explicit_conf, 3),
            "method": "opencv_skin_face_heuristic_v1",
            "sampled_frames": sampled,
            "suspicious_frames": suspicious_frames,
            "romantic_like_frames": romantic_like_frames,
            "max_skin_ratio": round(float(max_skin_ratio), 3),
            "max_region_ratio": round(float(max_region_ratio), 3),
            "max_face_area": round(float(max_face_area), 3),
            "frame_ratio": round(ratio, 3),
        }

    return {
        "visual": visual,
        "anime_check": anime_check,
        "content_check": content_check,
    }


def analyze_video_vibe(video_path: Path, max_frames: int = 140) -> dict[str, Any]:
    unified = analyze_video_unified(video_path, max_frames_vibe=max_frames)
    return unified.get("visual", {}) or {
        "visual_mood": "dramatic",
        "motion": 0.5,
        "brightness": 0.5,
        "saturation": 0.5,
        "contrast": 0.5,
        "desired_tempo": 130,
    }


def detect_anime_video(video_path: Path, max_frames: int = 72) -> dict[str, Any]:
    """
    Lightweight heuristic classifier.
    Anime tends to have flatter regions + clean edges, while real footage has denser natural texture.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "is_anime": True,
            "anime_confidence": 0.5,
            "real_confidence": 0.5,
            "method": "heuristic_fallback_open_failed",
            "reason": "open_failed",
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, frame_count // max_frames) if frame_count > 0 else 2

    sampled = 0
    i = 0
    flat_values: list[float] = []
    edge_values: list[float] = []
    sat_values: list[float] = []
    texture_values: list[float] = []
    contrast_values: list[float] = []

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride != 0:
            continue

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        detail_ratio = float(np.mean(detail > 8))
        flat_ratio = 1.0 - detail_ratio
        edges = cv2.Canny(gray, 70, 170)
        edge_density = float(np.mean(edges > 0))
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture = float(np.mean(np.abs(lap)) / 255.0)
        sat = float(np.mean(hsv[:, :, 1]) / 255.0)
        cont = float(np.std(gray) / 128.0)

        flat_values.append(float(np.clip(flat_ratio, 0.0, 1.0)))
        edge_values.append(float(np.clip(edge_density, 0.0, 1.0)))
        texture_values.append(float(np.clip(texture, 0.0, 1.0)))
        sat_values.append(float(np.clip(sat, 0.0, 1.0)))
        contrast_values.append(float(np.clip(cont, 0.0, 1.0)))
        sampled += 1

    cap.release()

    if not flat_values:
        return {
            "is_anime": True,
            "anime_confidence": 0.5,
            "real_confidence": 0.5,
            "method": "heuristic_fallback_no_frames",
            "reason": "no_frames",
        }

    flat = float(np.mean(flat_values))
    edge = float(np.mean(edge_values))
    texture = float(np.mean(texture_values))
    sat = float(np.mean(sat_values))
    contrast = float(np.mean(contrast_values))

    anime_score = (
        (flat * 1.55)
        + (edge * 0.9)
        + (sat * 0.45)
        + (contrast * 0.2)
        - (texture * 1.22)
    )
    raw_conf = (anime_score - 0.16) / 0.42
    anime_conf = float(np.clip(raw_conf, 0.0, 1.0))
    real_conf = float(np.clip(1.0 - anime_conf, 0.0, 1.0))

    # Strong real-footage override for safety.
    likely_real = bool((texture > 0.165 and flat < 0.43) or (texture > 0.19 and edge < 0.075))
    is_anime = bool((anime_conf >= 0.52) and (not likely_real))

    return {
        "is_anime": is_anime,
        "anime_confidence": round(anime_conf, 3),
        "real_confidence": round(real_conf, 3),
        "method": "opencv_heuristic_anime_gate_v1",
        "sampled_frames": sampled,
        "flat_ratio": round(flat, 3),
        "edge_density": round(edge, 3),
        "texture": round(texture, 3),
        "saturation": round(sat, 3),
        "contrast": round(contrast, 3),
        "score": round(float(anime_score), 3),
    }


def detect_explicit_content(video_path: Path, max_frames: int = 84) -> dict[str, Any]:
    """
    Conservative safety gate:
    block likely nudity/porn frames, but avoid flagging typical romantic/kissing scenes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "is_explicit": False,
            "confidence": 0.0,
            "method": "heuristic_fallback_open_failed",
            "reason": "open_failed",
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, frame_count // max_frames) if frame_count > 0 else 2

    sampled = 0
    i = 0
    suspicious_frames = 0
    romantic_like_frames = 0
    max_skin_ratio = 0.0
    max_region_ratio = 0.0
    max_face_area = 0.0

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride != 0:
            continue

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        area = float(max(1, h * w))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)

        skin_ycrcb = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        skin_hsv = cv2.inRange(
            hsv,
            np.array([0, 20, 45], dtype=np.uint8),
            np.array([25, 200, 255], dtype=np.uint8),
        )
        skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
        skin_mask = cv2.morphologyEx(
            skin_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        skin_mask = cv2.morphologyEx(
            skin_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )

        skin_ratio = float(np.mean(skin_mask > 0))
        max_skin_ratio = max(max_skin_ratio, skin_ratio)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0.0
        for c in contours:
            largest_area = max(largest_area, float(cv2.contourArea(c)))
        largest_region_ratio = float(np.clip(largest_area / area, 0.0, 1.0))
        max_region_ratio = max(max_region_ratio, largest_region_ratio)

        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(18, 18))
        face_area_ratio = 0.0
        if len(faces) > 0:
            face_area_ratio = float(sum(float(fw * fh) for (_x, _y, fw, fh) in faces) / area)
        max_face_area = max(max_face_area, face_area_ratio)

        edges = cv2.Canny(gray, 70, 170)
        edge_density = float(np.mean(edges > 0))
        texture = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))) / 255.0)

        high_skin = skin_ratio > 0.48 and largest_region_ratio > 0.34
        extreme_skin = skin_ratio > 0.62 and largest_region_ratio > 0.42
        low_structure = edge_density < 0.09 and texture < 0.11
        likely_romantic_kiss = (face_area_ratio > 0.05 and skin_ratio < 0.5) or (len(faces) >= 1 and skin_ratio < 0.46)

        explicit_frame = bool(extreme_skin or (high_skin and low_structure and face_area_ratio < 0.12))
        if likely_romantic_kiss and skin_ratio < 0.5:
            explicit_frame = False
            romantic_like_frames += 1

        if explicit_frame:
            suspicious_frames += 1

        sampled += 1

    cap.release()

    if sampled <= 0:
        return {
            "is_explicit": False,
            "confidence": 0.0,
            "method": "heuristic_fallback_no_frames",
            "reason": "no_frames",
        }

    ratio = float(suspicious_frames / max(1, sampled))
    trigger_frames = max(3, int(math.ceil(sampled * 0.16)))
    explicit_conf = float(np.clip((ratio - 0.08) / 0.32, 0.0, 1.0))
    is_explicit = bool(suspicious_frames >= trigger_frames and ratio >= 0.13)
    if romantic_like_frames >= suspicious_frames and ratio < 0.2:
        is_explicit = False
        explicit_conf = min(explicit_conf, 0.45)

    return {
        "is_explicit": is_explicit,
        "confidence": round(explicit_conf, 3),
        "method": "opencv_skin_face_heuristic_v1",
        "sampled_frames": sampled,
        "suspicious_frames": suspicious_frames,
        "romantic_like_frames": romantic_like_frames,
        "max_skin_ratio": round(float(max_skin_ratio), 3),
        "max_region_ratio": round(float(max_region_ratio), 3),
        "max_face_area": round(float(max_face_area), 3),
        "frame_ratio": round(ratio, 3),
    }


def analyze_track_audio_vibe(track_path: Path, duration_sec: float = 18.0) -> dict[str, Any]:
    global TRACK_ANALYSIS_CACHE_DIRTY
    cache_key = _safe_resolve_str(track_path)
    signature = _track_signature(track_path)
    cached = TRACK_ANALYSIS_CACHE.get(cache_key)
    if cached and str(cached.get("sig", "")) == signature and isinstance(cached.get("data"), dict):
        return cached["data"]

    try:
        y, sr = _fast_audio_load(track_path, duration_sec=duration_sec)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = _tempo_value(tempo)

        rms = float(np.mean(librosa.feature.rms(y=y))) if len(y) else 0.0
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) if len(y) else 1200.0
        onset = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))) if len(y) else 0.0

        tempo_n = float(np.clip((tempo_value - 80.0) / 100.0, 0.0, 1.0))
        energy_n = float(np.clip(rms / 0.14, 0.0, 1.0))
        centroid_n = float(np.clip((centroid - 700.0) / 3000.0, 0.0, 1.0))
        onset_n = float(np.clip(onset / 6.0, 0.0, 1.0))

        mood_scores = {
            "hype": (tempo_n * 1.2) + (energy_n * 1.2) + (onset_n * 1.1) + (centroid_n * 0.4),
            "dark": ((1.0 - centroid_n) * 0.9) + (energy_n * 0.7) + (onset_n * 0.45) + (tempo_n * 0.35),
            "sad": ((1.0 - tempo_n) * 1.0) + ((1.0 - energy_n) * 1.1) + ((1.0 - onset_n) * 0.9),
            "dramatic": (energy_n * 0.8)
            + (onset_n * 0.7)
            + ((1.0 - abs(tempo_n - 0.55)) * 0.9),
            "chill": ((1.0 - tempo_n) * 0.95) + ((1.0 - onset_n) * 1.0) + ((1.0 - energy_n) * 0.8),
        }
        track_mood = max(mood_scores, key=mood_scores.get)

        data = {
            "tempo": round(tempo_value if tempo_value > 0 else 120.0, 2),
            "energy": round(energy_n, 3),
            "brightness": round(centroid_n, 3),
            "onset": round(onset_n, 3),
            "track_mood": track_mood,
        }
    except Exception:
        rel_key = _relative_track_key(track_path)
        data = {
            "tempo": float(MOOD_TO_BPM.get(infer_track_mood_from_name(rel_key), 120)),
            "energy": 0.5,
            "brightness": 0.5,
            "onset": 0.5,
            "track_mood": infer_track_mood_from_name(rel_key),
        }

    TRACK_ANALYSIS_CACHE[cache_key] = {"sig": signature, "data": data}
    TRACK_ANALYSIS_CACHE_DIRTY += 1
    _persist_track_analysis_cache()
    return data


def pick_local_phonk_track(
    mood: str,
    video_path: Path | None = None,
    prompt_text: str = "",
    *,
    allow_phonk: bool = True,
    variation_seed: int | None = None,
    visual_override: dict[str, Any] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    global LAST_SELECTED_TRACK_KEY
    _load_recent_tracks()
    files = list_local_phonk_files(allow_phonk=allow_phonk)
    if not files:
        return None, {"reason": "no_local_tracks"}

    folder_hints, folder_hints_strict = _infer_prompt_music_folder_hints(prompt_text)
    folder_hints_applied = False
    folder_hint_matches = 0
    if folder_hints:
        hinted_tracks = [f for f in files if _track_matches_folder_hints(f, folder_hints)]
        folder_hint_matches = len(hinted_tracks)
        if hinted_tracks:
            files = hinted_tracks
            folder_hints_applied = True
        elif folder_hints_strict:
            return None, {
                "reason": "no_folder_match",
                "folder_hints": folder_hints,
                "folder_hints_strict": True,
            }

    visual = visual_override if isinstance(visual_override, dict) else (
        analyze_video_vibe(video_path) if video_path else None
    )
    if not visual:
        visual = {
            "visual_mood": mood,
            "motion": 0.5,
            "brightness": 0.5,
            "saturation": 0.5,
            "contrast": 0.5,
            "desired_tempo": MOOD_TO_BPM.get(mood, 130),
        }
    style = build_style_profile(prompt_text, mood)
    prompt_profile = infer_prompt_music_profile(prompt_text, mood)
    fast_render = bool(style.get("fast_render", False))
    quality_priority = bool(style.get("quality_priority", False))
    smart_music = bool(style.get("smart_music", True))
    requested_music_mood, strict_music_mood = _infer_prompt_music_mood(prompt_text, mood)
    prompt_l = str(prompt_text or "").lower()
    romantic_intent = any(t in prompt_l for t in ("romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"))
    happy_intent = any(t in prompt_l for t in ("happy", "joy", "cheerful", "uplifting", "smile", "positive"))
    sad_or_dark_intent = any(t in prompt_l for t in ("sad", "heartbreak", "cry", "tears", "dark", "villain", "shadow"))
    visual_energy = float(
        np.clip(
            (float(visual.get("motion", 0.5)) * 0.5)
            + (float(visual.get("contrast", 0.5)) * 0.3)
            + (float(visual.get("saturation", 0.5)) * 0.2),
            0.0,
            1.0,
        )
    )
    require_non_phonk = bool(
        strict_music_mood and requested_music_mood == "chill" and (romantic_intent or happy_intent) and (not sad_or_dark_intent)
    )
    if require_non_phonk:
        non_phonk_tracks = [f for f in files if not _is_phonk_like_track(f)]
        if non_phonk_tracks:
            files = non_phonk_tracks
        else:
            return None, {
                "reason": "no_non_phonk_for_chill",
                "requested_music_mood": requested_music_mood,
                "strict_music_mood": strict_music_mood,
                "romantic_intent": romantic_intent,
                "happy_intent": happy_intent,
            }
    target_tempo = int(
        round((MOOD_TO_BPM.get(requested_music_mood, MOOD_TO_BPM.get(mood, 130)) * 0.68) + (float(visual["desired_tempo"]) * 0.32))
    )
    if int(style.get("intensity", 1)) >= 3:
        target_tempo = min(185, target_tempo + 8)

    # Fast pre-filter: analyze only the most likely candidates.
    prelim: list[tuple[float, Path]] = []
    for f in files:
        rel_key = _relative_track_key(f)
        filename_mood = infer_track_mood_from_name(rel_key)
        name_lower = rel_key.lower()
        pscore = 0.0
        if filename_mood == requested_music_mood:
            pscore += 2.8
        if filename_mood == visual["visual_mood"]:
            pscore += 2.1
        for word in MOOD_KEYWORDS.get(requested_music_mood, []):
            if word in name_lower:
                pscore += 0.35
        for word in MOOD_KEYWORDS.get(str(visual["visual_mood"]), []):
            if word in name_lower:
                pscore += 0.2
        prelim.append((pscore, f))
    prelim.sort(key=lambda item: (-item[0], item[1].name.lower()))

    if strict_music_mood:
        strict_pool = [
            item
            for item in prelim
            if infer_track_mood_from_name(_relative_track_key(item[1])) == requested_music_mood
            or requested_music_mood in _relative_track_key(item[1]).lower()
        ]
        if strict_pool:
            prelim = strict_pool

    if len(files) <= 8:
        shortlist_limit = len(files)
    elif fast_render:
        shortlist_limit = 4
    elif quality_priority and smart_music:
        shortlist_limit = 8
    else:
        shortlist_limit = 6
    shortlist_files = [item[1] for item in prelim[:shortlist_limit]]
    if not shortlist_files:
        shortlist_files = files

    analysis_duration_sec = 6.0 if fast_render else (10.0 if (quality_priority and smart_music) else 8.0)
    if len(files) > 24 and analysis_duration_sec > 6.0:
        analysis_duration_sec = 6.0
    tempo_weight = 3.0
    profile_weight = 3.2
    if smart_music and quality_priority:
        tempo_weight = 3.4
        profile_weight = 3.9
    elif smart_music:
        tempo_weight = 3.25
        profile_weight = 3.55

    scored: list[tuple[float, Path, dict[str, Any]]] = []
    for f in shortlist_files:
        rel_key = _relative_track_key(f)
        filename_mood = infer_track_mood_from_name(rel_key)
        audio_vibe = analyze_track_audio_vibe(f, duration_sec=analysis_duration_sec)
        track_mood = str(audio_vibe.get("track_mood", filename_mood))
        tempo = float(audio_vibe.get("tempo", 120.0))
        energy = float(audio_vibe.get("energy", 0.5))
        onset = float(audio_vibe.get("onset", 0.5))
        brightness = float(audio_vibe.get("brightness", 0.5))

        score = 0.0
        if filename_mood == requested_music_mood:
            score += 2.3
        if track_mood == requested_music_mood:
            score += 3.3
        if filename_mood == visual["visual_mood"]:
            score += 1.7
        if track_mood == visual["visual_mood"]:
            score += 2.4

        tempo_score = max(0.0, 1.0 - (abs(tempo - target_tempo) / 75.0))
        score += tempo_score * tempo_weight
        profile_distance = (
            abs(energy - prompt_profile["target_energy"]) * 0.45
            + abs(onset - prompt_profile["target_onset"]) * 0.4
            + abs(brightness - prompt_profile["target_brightness"]) * 0.25
        )
        profile_score = max(0.0, 1.0 - profile_distance)
        score += profile_score * profile_weight
        if track_mood == requested_music_mood and profile_score > 0.72:
            score += 0.7
        if smart_music:
            visual_energy_dist = abs(energy - visual_energy)
            score -= visual_energy_dist * 0.9
        if smart_music and track_mood not in {requested_music_mood, str(visual["visual_mood"])}:
            score -= 0.55
        if strict_music_mood and track_mood != requested_music_mood and filename_mood != requested_music_mood:
            score -= 1.15
        if strict_music_mood and requested_music_mood == "chill" and track_mood == "sad":
            score -= 2.2
        if strict_music_mood and requested_music_mood == "chill" and tempo > 130:
            score -= 0.75
        if strict_music_mood and requested_music_mood == "chill" and track_mood == "hype" and (not happy_intent):
            score -= 1.1
        if strict_music_mood and romantic_intent and (not sad_or_dark_intent):
            if track_mood in {"hype", "dark", "sad"}:
                score -= 2.6
            if filename_mood in {"hype", "dark", "sad"}:
                score -= 1.6
        if strict_music_mood and happy_intent and (not sad_or_dark_intent):
            if track_mood in {"sad", "dark"}:
                score -= 2.4
            if filename_mood in {"sad", "dark"}:
                score -= 1.6
            if tempo < 98:
                score -= 0.55
        if strict_music_mood and (romantic_intent or happy_intent) and _is_phonk_like_track(f):
            score -= 1.2

        if len(shortlist_files) > 1 and LAST_SELECTED_TRACK_KEY and rel_key == LAST_SELECTED_TRACK_KEY:
            score -= 1.4
        if rel_key in RECENT_TRACKS:
            score -= 0.9

        if bool(style.get("glitch", False)) and track_mood in {"dark", "hype"}:
            score += 0.6
        if bool(style.get("speed_ramp", False)) and tempo > 135:
            score += 0.6
        if bool(style.get("vignette", False)) and track_mood == "dark":
            score += 0.4

        scored.append(
            (
                score,
                f,
                {
                    "file_key": rel_key,
                    "filename_mood": filename_mood,
                    "track_mood": track_mood,
                    "tempo": round(tempo, 2),
                    "energy": round(energy, 3),
                    "onset": round(onset, 3),
                    "brightness": round(brightness, 3),
                    "profile_score": round(profile_score, 3),
                    "score": round(score, 3),
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1].name.lower()))
    winner = scored[0]
    variation_enabled = variation_seed is not None
    if variation_enabled and len(scored) > 1:
        rng = random.Random(int(variation_seed))
        top_pool_size = min(3, len(scored))
        top_pool = scored[:top_pool_size]
        top_score = float(top_pool[0][0])
        # Keep quality high: only allow alternatives close to the best score.
        allowed = [item for item in top_pool if (top_score - float(item[0])) <= 0.95]
        if not allowed:
            allowed = top_pool[:1]
        weights = [math.exp(max(0.0, float(item[0]))) for item in allowed]
        winner = rng.choices(allowed, weights=weights, k=1)[0]
    elif len(scored) > 1:
        top_score = float(scored[0][0])
        pool = [item for item in scored[:5] if (top_score - float(item[0])) <= 0.7]
        pool = [item for item in pool if item[2].get("file_key") not in RECENT_TRACKS] or pool
        if pool:
            rng = random.Random(int(time.time()) ^ (len(scored) << 8))
            weights = [math.exp(max(0.0, float(item[0]))) for item in pool]
            winner = rng.choices(pool, weights=weights, k=1)[0]
    top_candidates = [
        {
            "file": item[2]["file_key"],
            "score": item[2]["score"],
            "track_mood": item[2]["track_mood"],
            "tempo": item[2]["tempo"],
            "energy": item[2]["energy"],
            "onset": item[2]["onset"],
            "brightness": item[2]["brightness"],
            "profile_score": item[2]["profile_score"],
        }
        for item in scored[:3]
    ]
    info = {
        "visual_vibe": visual,
        "tracks_considered": len(files),
        "tracks_analyzed": len(shortlist_files),
        "variation_enabled": variation_enabled,
        "folder_hints": folder_hints,
        "folder_hints_applied": folder_hints_applied,
        "folder_hint_matches": folder_hint_matches,
        "folder_hints_strict": folder_hints_strict,
        "requested_music_mood": requested_music_mood,
        "requested_music_mood_strict": strict_music_mood,
        "require_non_phonk": require_non_phonk,
        "target_tempo": target_tempo,
        "selected_track_mood": winner[2]["track_mood"],
        "selected_filename_mood": winner[2]["filename_mood"],
        "selected_track_tempo": winner[2]["tempo"],
        "prompt_profile": prompt_profile,
        "top_candidates": top_candidates,
    }
    _persist_track_analysis_cache(force=True)
    if winner[1] is not None:
        key = _relative_track_key(winner[1])
        if key:
            if key in RECENT_TRACKS:
                RECENT_TRACKS.remove(key)
            RECENT_TRACKS.insert(0, key)
            RECENT_TRACKS[:] = RECENT_TRACKS[:8]
            _persist_recent_tracks()

    return winner[1], info


def _classify_segment_archetype(
    motion: float,
    scene_change: float,
    contrast: float,
    brightness: float,
    face_count: int,
    flow_mag: float,
    scene_energy: float,
) -> str:
    if face_count > 0 and motion < 0.1 and flow_mag < 0.08:
        return "character_focus"
    if scene_change > 0.36:
        return "hard_cut"
    if motion > 0.2 or flow_mag > 0.18 or scene_energy > 0.72:
        return "action"
    if brightness < 0.34 and contrast > 0.36:
        return "dark_tension"
    if scene_energy < 0.3:
        return "buildup"
    return "neutral"


def analyze_segment_intelligence(
    video_path: Path, cut_points: list[float], source_fps: float
) -> list[dict[str, Any]]:
    if len(cut_points) < 2:
        return []

    ANALYSIS_SIZE = (160, 90)
    FLOW_SIZE = (96, 54)
    SUBJECT_SIZE = (320, 180)
    MULTI_SAMPLE_MIN_SEC = 1.1
    MAX_SAMPLES_PER_SEGMENT = 3

    def _default_row() -> dict[str, Any]:
        return {
            "scene_energy": 0.5,
            "motion": 0.5,
            "flow_mag": 0.0,
            "contrast": 0.5,
            "brightness": 0.5,
            "saturation": 0.5,
            "scene_change": 0.0,
            "face_count": 0,
            "face_detected": False,
            "subject_conf": 0.0,
            "face_center_x": 0.5,
            "face_center_y": 0.42,
            "face_area": 0.0,
            "archetype": "neutral",
            "edit_priority": 0.5,
        }

    def _detect_subject_blob(
        gray_small: np.ndarray,
        prev_gray_small: np.ndarray | None,
        anchor_x: float,
        anchor_y: float,
    ) -> dict[str, float] | None:
        h_s, w_s = gray_small.shape[:2]
        frame_area = float(max(1, w_s * h_s))

        edges = cv2.Canny(gray_small, 64, 160)
        edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)

        motion_mask = np.zeros_like(edges)
        if prev_gray_small is not None and prev_gray_small.shape == gray_small.shape:
            diff = cv2.absdiff(gray_small, prev_gray_small)
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )

        texture = cv2.Laplacian(gray_small, cv2.CV_32F)
        texture_abs = np.abs(texture)
        tex_thr = float(np.percentile(texture_abs, 78.0))
        tex_mask = (texture_abs >= tex_thr).astype(np.uint8) * 255
        tex_mask = cv2.morphologyEx(
            tex_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        combined = cv2.bitwise_or(edges, motion_mask)
        combined = cv2.bitwise_or(combined, tex_mask)
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        combined = cv2.dilate(combined, np.ones((2, 2), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best: dict[str, float] | None = None
        best_score = -1.0
        ax = float(np.clip(anchor_x, 0.1, 0.9)) * w_s
        ay = float(np.clip(anchor_y, 0.1, 0.9)) * h_s
        diag = math.sqrt((w_s * w_s) + (h_s * h_s)) + 1e-6
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            area_n = area / frame_area
            if area_n < 0.012 or area_n > 0.62:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 12:
                continue
            extent = area / float(max(1, w * h))
            if extent < 0.08:
                continue

            m = cv2.moments(cnt)
            if m["m00"] > 1e-6:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx = float(x + (w * 0.5))
                cy = float(y + (h * 0.5))
            cdist = math.sqrt(((cx - ax) ** 2) + ((cy - ay) ** 2)) / diag
            center_weight = max(0.15, 1.0 - (cdist * 1.45))

            motion_overlap = 0.0
            if motion_mask is not None:
                roi_m = motion_mask[y : y + h, x : x + w]
                if roi_m.size > 0:
                    motion_overlap = float(np.mean(roi_m > 0))

            score = (area_n * 2.2) + (center_weight * 0.9) + (motion_overlap * 1.2) + (min(1.0, extent) * 0.35)
            if score > best_score:
                best_score = score
                best = {
                    "x": float(np.clip(cx / w_s, 0.0, 1.0)),
                    "y": float(np.clip(cy / h_s, 0.0, 1.0)),
                    "area": float(np.clip(area_n, 0.0, 1.0)),
                    "score": float(score),
                }

        return best

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [_default_row() for _ in range(len(cut_points) - 1)]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(source_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = max(1.0, fps)

    segment_frame_plan: list[tuple[int, int, bool]] = []
    segment_mid_frame: list[int] = []
    for idx in range(len(cut_points) - 1):
        start = float(cut_points[idx])
        end = float(cut_points[idx + 1])
        duration = max(0.0, end - start)
        mid = (start + end) * 0.5

        offsets = [0.0]
        if duration >= MULTI_SAMPLE_MIN_SEC:
            offsets = [-0.25, 0.0, 0.25]
        if len(offsets) > MAX_SAMPLES_PER_SEGMENT:
            offsets = offsets[:MAX_SAMPLES_PER_SEGMENT]

        frame_indices: list[int] = []
        for off in offsets:
            t = mid + (off * duration)
            if duration >= 0.2:
                t = float(np.clip(t, start + 0.05, end - 0.05))
            frame_idx = int(np.clip(int(t * fps), 0, max(0, total_frames - 1)))
            frame_indices.append(frame_idx)

        frame_indices = sorted(set(frame_indices))
        if not frame_indices:
            frame_indices = [int(np.clip(int(mid * fps), 0, max(0, total_frames - 1)))]
        mid_frame = min(frame_indices, key=lambda fi: abs(fi - int(mid * fps)))
        segment_mid_frame.append(mid_frame)

        for frame_idx in frame_indices:
            segment_frame_plan.append((frame_idx, idx, frame_idx == mid_frame))

    segment_frame_plan.sort(key=lambda item: item[0])

    intelligence: list[dict[str, Any]] = []
    prev_gray_small = None
    prev_hist = None
    current_frame_idx = 0
    last_face = {"count": 0, "x": 0.5, "y": 0.42, "area": 0.0}
    last_subject = {"x": 0.5, "y": 0.42, "area": 0.0, "score": 0.0}
    prev_gray_subject_small: np.ndarray | None = None
    last_flow_mag = 0.0

    seg_count = len(cut_points) - 1
    seg_acc = [
        {
            "count": 0,
            "motion": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
            "saturation": 0.0,
            "scene_change": 0.0,
            "edge_density": 0.0,
            "flow_mag": 0.0,
            "flow_ready": False,
            "face_count": int(last_face["count"]),
            "face_detected": False,
            "subject_conf": float(last_subject["score"]),
            "face_center_x": float(last_face["x"]),
            "face_center_y": float(last_face["y"]),
            "face_area": float(last_face["area"]),
        }
        for _ in range(seg_count)
    ]

    for frame_idx, seg_idx, is_mid in segment_frame_plan:
        if frame_idx < current_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame_idx = frame_idx

        while current_frame_idx < frame_idx:
            if not cap.grab():
                break
            current_frame_idx += 1

        ok, frame = cap.read()
        if ok:
            current_frame_idx += 1
        if (not ok) or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        hsv_small = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2BGR)
        hsv_small = cv2.cvtColor(hsv_small, cv2.COLOR_BGR2HSV)
        contrast = float(np.clip(np.std(gray_small) / 128.0, 0.0, 1.0))
        brightness = float(np.clip(np.mean(gray_small) / 255.0, 0.0, 1.0))
        saturation = float(np.clip(np.mean(hsv_small[:, :, 1]) / 255.0, 0.0, 1.0))

        motion = 0.0
        if prev_gray_small is not None and prev_gray_small.shape == gray_small.shape:
            diff = cv2.absdiff(gray_small, prev_gray_small)
            motion = float(np.clip(np.mean(diff) / 255.0, 0.0, 1.0))

        hist = cv2.calcHist([gray_small], [0], None, [24], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        scene_change = 0.0
        if prev_hist is not None:
            corr = float(cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
            scene_change = float(np.clip(1.0 - ((corr + 1.0) * 0.5), 0.0, 1.0))

        edges = cv2.Canny(gray_small, 64, 160)
        edge_density = float(np.clip(np.mean(edges > 0), 0.0, 1.0))

        if is_mid and prev_gray_small is not None:
            should_calc_flow = (seg_idx % 2 == 0) or (scene_change > 0.2) or (motion > 0.12)
            if should_calc_flow:
                prev_flow_small = cv2.resize(prev_gray_small, FLOW_SIZE, interpolation=cv2.INTER_AREA)
                flow_small = cv2.resize(gray_small, FLOW_SIZE, interpolation=cv2.INTER_AREA)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_flow_small,
                    flow_small,
                    None,
                    0.5,
                    2,
                    7,
                    2,
                    5,
                    1.1,
                    0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mag = float(np.clip(np.mean(mag) / 3.2, 0.0, 1.0))
                last_flow_mag = flow_mag
            else:
                flow_mag = float(np.clip(last_flow_mag * 0.96, 0.0, 1.0))
            seg_acc[seg_idx]["flow_mag"] = flow_mag
            seg_acc[seg_idx]["flow_ready"] = True

        acc = seg_acc[seg_idx]
        acc["count"] += 1
        acc["motion"] += motion
        acc["contrast"] += contrast
        acc["brightness"] += brightness
        acc["saturation"] += saturation
        acc["edge_density"] += edge_density
        if scene_change > acc["scene_change"]:
            acc["scene_change"] = scene_change

        if is_mid:
            face_count = int(last_face["count"])
            face_detected = False
            subject_conf = float(last_subject["score"])
            face_center_x = float(last_face["x"])
            face_center_y = float(last_face["y"])
            face_area = float(last_face["area"])

            gray_subject_small = cv2.resize(gray, SUBJECT_SIZE, interpolation=cv2.INTER_AREA)
            if not FACE_CASCADE.empty():
                should_detect_face = (seg_idx % 2 == 0) or (face_count == 0) or (scene_change > 0.2)
                if should_detect_face:
                    faces = FACE_CASCADE.detectMultiScale(
                        gray_subject_small,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(24, 24),
                    )
                    face_count = int(len(faces))
                    if face_count > 0:
                        face_detected = True
                        x, y, w, h = max(faces, key=lambda arr: int(arr[2]) * int(arr[3]))
                        face_center_x = float(np.clip((x + (w * 0.5)) / SUBJECT_SIZE[0], 0.0, 1.0))
                        face_center_y = float(np.clip((y + (h * 0.5)) / SUBJECT_SIZE[1], 0.0, 1.0))
                        face_area = float(np.clip((w * h) / float(SUBJECT_SIZE[0] * SUBJECT_SIZE[1]), 0.0, 1.0))
                    else:
                        face_center_x = 0.5
                        face_center_y = 0.42
                        face_area = 0.0
                    last_face = {"count": face_count, "x": face_center_x, "y": face_center_y, "area": face_area}

            if face_count <= 0 or face_area < 0.012:
                should_detect_subject = (seg_idx % 3 == 0) or (scene_change > 0.2) or (motion > 0.1) or (face_count <= 0)
                if should_detect_subject:
                    anchor_x = float(last_subject["x"]) if float(last_subject["score"]) > 0.1 else 0.5
                    anchor_y = float(last_subject["y"]) if float(last_subject["score"]) > 0.1 else 0.42
                    subject = _detect_subject_blob(
                        gray_subject_small,
                        prev_gray_subject_small,
                        anchor_x=anchor_x,
                        anchor_y=anchor_y,
                    )
                    if subject is not None:
                        conf = float(subject["score"])
                        last_subject = {
                            "x": float(subject["x"]),
                            "y": float(subject["y"]),
                            "area": float(subject["area"]),
                            "score": conf,
                        }
                        subject_conf = conf
                        if conf >= 0.5:
                            face_count = max(1, face_count)
                            face_center_x = float(subject["x"])
                            face_center_y = float(subject["y"])
                            face_area = float(np.clip(max(face_area, subject["area"]), 0.0, 1.0))
                elif float(last_subject["score"]) >= 0.36:
                    face_count = max(1, face_count)
                    face_center_x = float(last_subject["x"])
                    face_center_y = float(last_subject["y"])
                    face_area = float(np.clip(max(face_area, float(last_subject["area"]) * 0.98), 0.0, 1.0))
                    subject_conf = float(last_subject["score"]) * 0.95

            prev_gray_subject_small = gray_subject_small

            acc["face_count"] = face_count
            acc["face_detected"] = bool(face_detected)
            acc["subject_conf"] = subject_conf
            acc["face_center_x"] = face_center_x
            acc["face_center_y"] = face_center_y
            acc["face_area"] = face_area

        prev_gray_small = gray_small
        prev_hist = hist

    for seg_idx in range(seg_count):
        acc = seg_acc[seg_idx]
        if acc["count"] <= 0:
            intelligence.append(_default_row())
            continue

        inv = 1.0 / float(acc["count"])
        motion = float(np.clip(acc["motion"] * inv, 0.0, 1.0))
        contrast = float(np.clip(acc["contrast"] * inv, 0.0, 1.0))
        brightness = float(np.clip(acc["brightness"] * inv, 0.0, 1.0))
        saturation = float(np.clip(acc["saturation"] * inv, 0.0, 1.0))
        edge_density = float(np.clip(acc["edge_density"] * inv, 0.0, 1.0))
        scene_change = float(np.clip(acc["scene_change"], 0.0, 1.0))
        flow_mag = float(acc["flow_mag"] if acc["flow_ready"] else np.clip(last_flow_mag * 0.96, 0.0, 1.0))

        scene_energy = float(
            np.clip(
                (motion * 0.42)
                + (contrast * 0.28)
                + (saturation * 0.18)
                + (scene_change * 0.32)
                + (edge_density * 0.2),
                0.0,
                1.0,
            )
        )

        archetype = _classify_segment_archetype(
            motion=motion,
            scene_change=scene_change,
            contrast=contrast,
            brightness=brightness,
            face_count=int(acc["face_count"]),
            flow_mag=flow_mag,
            scene_energy=scene_energy,
        )
        edit_priority = float(
            np.clip(
                (scene_energy * 0.55)
                + (scene_change * 0.35)
                + (flow_mag * 0.25)
                + (0.1 if archetype == "action" else 0.0),
                0.0,
                1.0,
            )
        )

        intelligence.append(
            {
                "scene_energy": round(scene_energy, 4),
                "motion": round(motion, 4),
                "flow_mag": round(flow_mag, 4),
                "contrast": round(contrast, 4),
                "brightness": round(brightness, 4),
                "saturation": round(saturation, 4),
                "scene_change": round(scene_change, 4),
                "face_count": int(acc["face_count"]),
                "face_detected": bool(acc["face_detected"]),
                "subject_conf": round(float(acc["subject_conf"]), 4),
                "face_center_x": round(float(acc["face_center_x"]), 4),
                "face_center_y": round(float(acc["face_center_y"]), 4),
                "face_area": round(float(acc["face_area"]), 4),
                "archetype": archetype,
                "edit_priority": round(edit_priority, 4),
            }
        )

    cap.release()
    if len(intelligence) < (len(cut_points) - 1):
        intelligence.extend(_default_row() for _ in range((len(cut_points) - 1) - len(intelligence)))
    return enrich_segment_rows(intelligence)


def append_segment_labels_csv(
    csv_path: Path,
    render_name: str,
    mood: str,
    style: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = [
        "render_name",
        "mood",
        "intensity",
        "segment_index",
        "scene_energy",
        "motion",
        "flow_mag",
        "contrast",
        "brightness",
        "saturation",
        "scene_change",
        "face_count",
        "face_center_x",
        "face_center_y",
        "face_area",
        "archetype",
        "edit_priority",
        "cut_priority",
        "hype_score",
        "emotion_weight",
        "shot_type",
        "motion_class",
        "motion_direction",
        "zoom_applied",
        "shake_applied",
        "glitch_applied",
        "glow_applied",
        "flash_applied",
        "grade_applied",
        "ken_burns_applied",
        "anime_tone_applied",
        "chroma_pulse_applied",
        "speed_ramp_applied",
        "slow_motion_applied",
        "fast_motion_applied",
        "reverse_applied",
        "boomerang_applied",
        "black_white_applied",
        "fire_lines_applied",
        "focus_zoom_applied",
        "particles_applied",
        "energy_aura_applied",
        "turbulent_applied",
        "posterize_time_applied",
        "blend_overlay_applied",
        "mask_roto_applied",
        "chroma_key_applied",
        "difference_matte_applied",
        "object_inpaint_applied",
        "clip_repeat_applied",
        "clip_repeat_count",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "render_name": render_name,
                    "mood": mood,
                    "intensity": style.get("intensity", 1),
                    "segment_index": row.get("segment_index", 0),
                    "scene_energy": row.get("scene_energy", 0.0),
                    "motion": row.get("motion", 0.0),
                    "flow_mag": row.get("flow_mag", 0.0),
                    "contrast": row.get("contrast", 0.0),
                    "brightness": row.get("brightness", 0.0),
                    "saturation": row.get("saturation", 0.0),
                    "scene_change": row.get("scene_change", 0.0),
                    "face_count": row.get("face_count", 0),
                    "face_center_x": row.get("face_center_x", 0.5),
                    "face_center_y": row.get("face_center_y", 0.42),
                    "face_area": row.get("face_area", 0.0),
                    "archetype": row.get("archetype", "neutral"),
                    "edit_priority": row.get("edit_priority", 0.0),
                    "cut_priority": row.get("cut_priority", 0.0),
                    "hype_score": row.get("hype_score", 0.0),
                    "emotion_weight": row.get("emotion_weight", 0.0),
                    "shot_type": row.get("shot_type", "wide"),
                    "motion_class": row.get("motion_class", "low"),
                    "motion_direction": row.get("motion_direction", "still"),
                    "zoom_applied": row.get("zoom_applied", False),
                    "shake_applied": row.get("shake_applied", False),
                    "glitch_applied": row.get("glitch_applied", False),
                    "glow_applied": row.get("glow_applied", False),
                    "flash_applied": row.get("flash_applied", False),
                    "grade_applied": row.get("grade_applied", False),
                    "ken_burns_applied": row.get("ken_burns_applied", False),
                    "anime_tone_applied": row.get("anime_tone_applied", False),
                    "chroma_pulse_applied": row.get("chroma_pulse_applied", False),
                    "speed_ramp_applied": row.get("speed_ramp_applied", False),
                    "slow_motion_applied": row.get("slow_motion_applied", False),
                    "fast_motion_applied": row.get("fast_motion_applied", False),
                    "reverse_applied": row.get("reverse_applied", False),
                    "boomerang_applied": row.get("boomerang_applied", False),
                    "black_white_applied": row.get("black_white_applied", False),
                    "fire_lines_applied": row.get("fire_lines_applied", False),
                    "focus_zoom_applied": row.get("focus_zoom_applied", False),
                    "particles_applied": row.get("particles_applied", False),
                    "energy_aura_applied": row.get("energy_aura_applied", False),
                    "turbulent_applied": row.get("turbulent_applied", False),
                    "posterize_time_applied": row.get("posterize_time_applied", False),
                    "blend_overlay_applied": row.get("blend_overlay_applied", False),
                    "mask_roto_applied": row.get("mask_roto_applied", False),
                    "chroma_key_applied": row.get("chroma_key_applied", False),
                    "difference_matte_applied": row.get("difference_matte_applied", False),
                    "object_inpaint_applied": row.get("object_inpaint_applied", False),
                    "clip_repeat_applied": row.get("clip_repeat_applied", False),
                    "clip_repeat_count": row.get("clip_repeat_count", 1),
                }
            )


def build_local_tracks_for_mood(mood: str, limit: int = 5, *, allow_phonk: bool = True) -> list[Track]:
    files = list_local_phonk_files(allow_phonk=allow_phonk)
    ranked: list[tuple[int, Path]] = []
    for f in files:
        rel_key = _relative_track_key(f)
        guessed = infer_track_mood_from_name(rel_key)
        score = 2 if guessed == mood else 0
        lower_name = rel_key.lower()
        for word in MOOD_KEYWORDS.get(mood, []):
            if word in lower_name:
                score += 1
        ranked.append((score, f))
    ranked.sort(key=lambda item: (-item[0], item[1].name.lower()))

    out: list[Track] = []
    for idx, (_, f) in enumerate(ranked[:limit], start=1):
        rel_key = _relative_track_key(f)
        out.append(
            Track(
                id=idx,
                title=f.stem.replace("-", " ").replace("_", " "),
                tags=f"{infer_track_mood_from_name(rel_key)}, local",
                duration_sec=None,
                preview_url=f"/api/phonks/{quote(rel_key)}",
                audio_url=f"/api/phonks/{quote(rel_key)}",
                user="archive-local",
                source="local",
            )
        )
    return out


def generate_beat_markers(video_duration_sec: float, bpm: int) -> list[float]:
    if video_duration_sec <= 0:
        return []

    beat_interval = 60.0 / max(bpm, 1)
    total_beats = int(math.floor(video_duration_sec / beat_interval))
    return [round(i * beat_interval, 3) for i in range(total_beats + 1)]


def probe_video_metadata(video_path: Path) -> dict[str, Any]:
    ffprobe_cmd = shutil.which("ffprobe")
    if ffprobe_cmd:
        try:
            proc = run_subprocess(
                [
                    ffprobe_cmd,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-show_format",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.ok:
                data = json.loads(proc.stdout or "{}")
                streams = data.get("streams", [])
                video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                fps_raw = str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1")
                try:
                    num, den = fps_raw.split("/", 1)
                    fps = float(num) / max(1.0, float(den))
                except Exception:
                    fps = 30.0
                duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
                return {
                    "duration": max(0.0, duration),
                    "fps": max(1.0, fps),
                    "width": int(video_stream.get("width") or 0),
                    "height": int(video_stream.get("height") or 0),
                    "has_audio": any(s.get("codec_type") == "audio" for s in streams),
                }
        except Exception:
            pass
    return {"duration": 0.0, "fps": 30.0, "width": 0, "height": 0, "has_audio": False}


def extract_audio_for_analysis(video_path: Path, output_path: Path, duration_sec: float) -> bool:
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        return False
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-t",
        f"{max(0.1, float(duration_sec or BEAT_ANALYSIS_MAX_SEC)):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        proc = run_subprocess(cmd, capture_output=True, text=True, timeout=90)
        return proc.ok and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def is_allowed_video(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS


def detect_beats_with_librosa(video_path: Path, mood: str) -> tuple[int, float, list[float], list[str]]:
    warnings: list[str] = []
    fallback_bpm = MOOD_TO_BPM.get(mood, 130)

    meta = probe_video_metadata(video_path)
    duration = float(meta.get("duration") or 0.0)
    if not meta.get("has_audio"):
        warnings.append("Video has no audio; using fallback BPM.")
        return fallback_bpm, duration, generate_beat_markers(duration, fallback_bpm), warnings

    analysis_window = min(duration if duration > 0 else 0.0, BEAT_ANALYSIS_MAX_SEC)
    if duration > BEAT_ANALYSIS_MAX_SEC:
        warnings.append(
            f"Fast beat mode enabled: analyzing first {int(BEAT_ANALYSIS_MAX_SEC)}s then extending BPM across timeline."
        )

    temp_audio_path = UPLOAD_DIR / f"audio_{uuid.uuid4().hex}.wav"
    if not extract_audio_for_analysis(video_path, temp_audio_path, analysis_window):
        warnings.append("FFmpeg audio extraction failed; using fallback BPM.")
        return fallback_bpm, duration, generate_beat_markers(duration, fallback_bpm), warnings

    try:
        y, sr = _fast_audio_load(temp_audio_path, duration_sec=analysis_window if analysis_window > 0 else 0.0)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        if hasattr(tempo, "__len__"):
            tempo_value = float(tempo[0]) if len(tempo) else 0.0
        else:
            tempo_value = float(tempo)
        detected_bpm = int(round(tempo_value)) if tempo_value > 0 else fallback_bpm
        markers = [round(float(t), 3) for t in beat_times if float(t) <= analysis_window]

        if duration > analysis_window:
            if len(markers) >= 2:
                diffs = [markers[i + 1] - markers[i] for i in range(len(markers) - 1)]
                beat_interval = float(np.clip(np.median(diffs), 0.2, 1.2))
            else:
                beat_interval = 60.0 / max(detected_bpm, 1)
            start_t = markers[0] if markers else 0.0
            t = start_t
            expanded: list[float] = []
            while t <= duration:
                expanded.append(round(float(t), 3))
                t += beat_interval
            markers = expanded

        if not markers:
            warnings.append("Librosa found no beat frames; using fallback BPM markers.")
            markers = generate_beat_markers(duration, detected_bpm)

        return detected_bpm, duration, markers[:1500], warnings
    finally:
        temp_audio_path.unlink(missing_ok=True)


def detect_beats_from_audio_file(
    audio_path: Path, duration_limit_sec: float, mood: str
) -> tuple[int, list[float], list[str]]:
    warnings: list[str] = []
    fallback_bpm = MOOD_TO_BPM.get(mood, 130)

    try:
        sig = _track_signature(audio_path)
    except Exception:
        sig = f"{audio_path}:{audio_path.stat().st_size if audio_path.exists() else 0}"
    cache_key = f"{_safe_resolve_str(audio_path)}|{sig}|{round(float(duration_limit_sec),2)}|{mood}"
    cached = _AUDIO_BEAT_CACHE.get(cache_key)
    if cached is not None:
        bpm_cached, markers_cached = cached
        return int(bpm_cached), list(markers_cached), warnings

    analysis_window = duration_limit_sec if duration_limit_sec <= BEAT_ANALYSIS_MAX_SEC else BEAT_ANALYSIS_MAX_SEC
    if duration_limit_sec > BEAT_ANALYSIS_MAX_SEC:
        warnings.append(
            f"Fast beat mode enabled: analyzing first {int(BEAT_ANALYSIS_MAX_SEC)}s then extending BPM across timeline."
        )
    y, sr = _fast_audio_load(audio_path, duration_sec=analysis_window if analysis_window > 0 else 0.0)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    if hasattr(tempo, "__len__"):
        tempo_value = float(tempo[0]) if len(tempo) else 0.0
    else:
        tempo_value = float(tempo)
    detected_bpm = int(round(tempo_value)) if tempo_value > 0 else fallback_bpm

    markers = [round(float(t), 3) for t in beat_times if float(t) <= analysis_window]
    if duration_limit_sec > analysis_window:
        if len(markers) >= 2:
            diffs = [markers[i + 1] - markers[i] for i in range(len(markers) - 1)]
            beat_interval = float(np.clip(np.median(diffs), 0.2, 1.2))
        else:
            beat_interval = 60.0 / max(detected_bpm, 1)
        start_t = markers[0] if markers else 0.0
        t = start_t
        expanded: list[float] = []
        while t <= duration_limit_sec:
            expanded.append(round(float(t), 3))
            t += beat_interval
        markers = expanded

    if not markers:
        warnings.append("No beats found in selected phonk track; using BPM fallback markers.")
        markers = generate_beat_markers(duration_limit_sec, detected_bpm)
    out_markers = markers[:1500]
    if len(_AUDIO_BEAT_CACHE) > 96:
        _AUDIO_BEAT_CACHE.pop(next(iter(_AUDIO_BEAT_CACHE)))
    _AUDIO_BEAT_CACHE[cache_key] = (int(detected_bpm), list(out_markers))
    return detected_bpm, out_markers, warnings


def detect_music_drops(audio_path: Path, duration_sec: float, max_drops: int = 3) -> list[float]:
    try:
        y, sr = _fast_audio_load(audio_path, duration_sec=min(45.0, duration_sec))
        if len(y) == 0:
            return []
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env.size < 8:
            return []
        onset_env = librosa.util.normalize(onset_env)
        peaks = []
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1]:
                peaks.append(i)
        if not peaks:
            return []
        strengths = [(i, float(onset_env[i])) for i in peaks]
        strengths.sort(key=lambda item: item[1], reverse=True)
        top = strengths[: max_drops * 2]
        hop_length = 512
        times = [float(librosa.frames_to_time(i, sr=sr, hop_length=hop_length)) for i, _ in top]
        chosen: list[float] = []
        for t in times:
            if all(abs(t - other) > 2.5 for other in chosen):
                chosen.append(t)
            if len(chosen) >= max_drops:
                break
        return sorted(chosen)
    except Exception:
        return []


def select_music_excerpt_for_video(
    audio_path: Path,
    target_duration_sec: float,
    mood: str,
    *,
    variation_seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    target_duration_sec = float(max(1.0, target_duration_sec))
    try:
        sig = _track_signature(audio_path)
    except Exception:
        sig = f"{audio_path}:{audio_path.stat().st_size if audio_path.exists() else 0}"
    variation_bucket = int(variation_seed) % 7 if variation_seed is not None else 0
    cache_key = f"{_safe_resolve_str(audio_path)}|{sig}|{round(target_duration_sec,2)}|{mood}|v{variation_bucket}"
    cached = _MUSIC_EXCERPT_CACHE.get(cache_key)
    if cached is not None:
        start_cached, info_cached = cached
        return float(start_cached), dict(info_cached)
    try:
        y, sr = _fast_audio_load(audio_path, duration_sec=0.0)
    except Exception:
        return 0.0, {"method": "fallback", "reason": "audio_load_failed"}
    if len(y) == 0 or sr <= 0:
        return 0.0, {"method": "fallback", "reason": "empty_audio"}

    total_duration = float(len(y) / sr)
    if total_duration <= target_duration_sec + 0.35:
        return 0.0, {"method": "start", "reason": "track_short", "total_duration_sec": round(total_duration, 2)}

    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms_env = librosa.feature.rms(y=y, hop_length=hop)[0]
    n = int(min(len(onset_env), len(rms_env)))
    if n < 8:
        return 0.0, {"method": "start", "reason": "insufficient_features"}

    onset_n = onset_env[:n].astype(np.float32)
    rms_n = rms_env[:n].astype(np.float32)
    onset_n = onset_n / (np.max(onset_n) + 1e-6)
    rms_n = rms_n / (np.max(rms_n) + 1e-6)

    if mood in {"hype", "dark", "dramatic"}:
        metric = (0.58 * onset_n) + (0.42 * rms_n)
    elif mood == "sad":
        target = 0.42
        metric = 1.0 - np.abs(((0.46 * onset_n) + (0.54 * rms_n)) - target)
    else:
        target = 0.36
        metric = 1.0 - np.abs(((0.4 * onset_n) + (0.6 * rms_n)) - target)

    win = int(max(4, round((target_duration_sec * sr) / hop)))
    if win >= n:
        return 0.0, {"method": "start", "reason": "window_too_large"}

    csum = np.concatenate(([0.0], np.cumsum(metric, dtype=np.float64)))
    window_scores = (csum[win:] - csum[:-win]) / float(win)
    if len(window_scores) == 0:
        return 0.0, {"method": "start", "reason": "no_windows"}

    if variation_seed is None:
        start_idx = int(np.argmax(window_scores))
    else:
        rng = random.Random(int(variation_seed) + 53)
        ranked = np.argsort(window_scores)[::-1]
        top_k = int(min(5, len(ranked)))
        if top_k <= 1:
            start_idx = int(ranked[0]) if len(ranked) else int(np.argmax(window_scores))
        else:
            top_idx = [int(ranked[i]) for i in range(top_k)]
            top_scores = [float(window_scores[i]) for i in top_idx]
            max_score = max(top_scores) if top_scores else 0.0
            # Keep quality: only sample among near-best windows.
            allowed = [i for i, s in zip(top_idx, top_scores) if (max_score - s) <= 0.08]
            if not allowed:
                allowed = [top_idx[0]]
            weights = [math.exp(float(window_scores[i]) * 3.0) for i in allowed]
            start_idx = int(rng.choices(allowed, weights=weights, k=1)[0])
    start_sec = float((start_idx * hop) / sr)
    max_start = max(0.0, total_duration - target_duration_sec - 0.05)
    start_sec = float(np.clip(start_sec, 0.0, max_start))

    info = {
        "method": "best_window",
        "start_sec": round(start_sec, 3),
        "target_duration_sec": round(target_duration_sec, 3),
        "track_duration_sec": round(total_duration, 3),
    }
    if len(_MUSIC_EXCERPT_CACHE) > 96:
        _MUSIC_EXCERPT_CACHE.pop(next(iter(_MUSIC_EXCERPT_CACHE)))
    _MUSIC_EXCERPT_CACHE[cache_key] = (float(start_sec), dict(info))
    return start_sec, info


def _build_cut_points_from_markers(
    beat_markers: list[float],
    duration: float,
    mood: str,
    intensity: int,
    max_points: int,
    selective_beats: bool = True,
    all_beats: bool = False,
    fast_render: bool = False,
    variation_seed: int | None = None,
) -> list[float]:
    beat_step = {
        "hype": 2,
        "dark": 2,
        "dramatic": 2,
        "sad": 3,
        "chill": 3,
    }.get(mood, 2)
    beat_step = max(1, beat_step - (1 if intensity >= 2 else 0))
    if intensity >= 3 and duration <= 20:
        beat_step = 1
    elif intensity >= 3:
        beat_step = 2
    if selective_beats and (not all_beats):
        beat_step += 1
    if fast_render:
        beat_step += 1
    rng = random.Random(int(variation_seed or 0))
    beat_phase = rng.randrange(beat_step) if beat_step > 1 else 0

    cut_points = [0.0]
    for idx, t in enumerate(beat_markers):
        if 0 < t < duration and ((idx + beat_phase) % beat_step == 0):
            cut_points.append(float(t))
    if len(cut_points) <= 1:
        interval = 0.8 if mood in {"hype", "dark", "dramatic"} else 1.2
        t = interval
        while t < duration:
            cut_points.append(round(t, 3))
            t += interval
    cut_points = sorted(set(cut_points))
    if cut_points[-1] < duration:
        cut_points.append(duration)
    min_seg = 0.14 if intensity >= 3 else 0.16
    return _compress_cut_points(cut_points, min_gap=min_seg, max_points=max_points)


def _score_proxy_edit_quality(
    cut_points: list[float],
    rows: list[dict[str, Any]],
    mood: str,
    style: dict[str, Any],
    base_score_raw: float,
    profile_score: float,
    target_tempo: float,
    detected_bpm: float,
) -> float:
    if len(cut_points) < 2 or not rows:
        return 0.0
    duration = max(0.1, float(cut_points[-1] - cut_points[0]))
    seg_count = len(cut_points) - 1
    seg_density = seg_count / duration

    target_density = {
        "hype": 1.55,
        "dark": 1.35,
        "dramatic": 1.25,
        "sad": 0.9,
        "chill": 0.75,
    }.get(mood, 1.1)
    target_density *= 1.0 + (0.08 * max(0, int(style.get("intensity", 1)) - 1))
    density_fit = max(0.0, 1.0 - (abs(seg_density - target_density) / max(target_density, 0.3)))

    priority_vals = [float(r.get("edit_priority", 0.5)) for r in rows]
    impact_vals = [
        1.0 if (float(r.get("scene_energy", 0.0)) > 0.58 or float(r.get("scene_change", 0.0)) > 0.24) else 0.0
        for r in rows
    ]
    motion_vals = [float(r.get("motion", 0.0)) for r in rows]
    scene_change_vals = [float(r.get("scene_change", 0.0)) for r in rows]
    face_vals = [1.0 if int(r.get("face_count", 0)) > 0 else 0.0 for r in rows]

    priority_mean = float(np.clip(np.mean(priority_vals), 0.0, 1.0))
    impact_ratio = float(np.clip(np.mean(impact_vals), 0.0, 1.0))
    motion_mean = float(np.clip(np.mean(motion_vals), 0.0, 1.0))
    scene_var = float(np.clip(np.std(scene_change_vals) * 2.5, 0.0, 1.0))
    face_presence = float(np.clip(np.mean(face_vals), 0.0, 1.0))

    tempo_fit = max(0.0, 1.0 - (abs(float(detected_bpm) - float(target_tempo)) / 75.0))
    base_norm = float(np.clip(base_score_raw / 14.0, 0.0, 1.0))
    profile_score = float(np.clip(profile_score, 0.0, 1.0))
    face_bonus = 0.08 * face_presence if bool(style.get("character_focus", False)) else 0.0

    return float(
        np.clip(
            (base_norm * 0.28)
            + (profile_score * 0.14)
            + (density_fit * 0.18)
            + (priority_mean * 0.16)
            + (impact_ratio * 0.1)
            + (tempo_fit * 0.08)
            + (motion_mean * 0.04)
            + (scene_var * 0.02)
            + face_bonus,
            0.0,
            1.0,
        )
    )


def evaluate_top_phonk_candidates(
    video_path: Path,
    mood: str,
    prompt_text: str,
    source_duration: float,
    source_fps: float,
    track_selection: dict[str, Any] | None,
    *,
    allow_phonk: bool = True,
    variation_seed: int | None = None,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    global LAST_SELECTED_TRACK_KEY
    warnings: list[str] = []
    if not track_selection:
        return None, {"method": "none", "candidates": []}, warnings

    top_candidates_raw = track_selection.get("top_candidates", [])
    if not isinstance(top_candidates_raw, list) or not top_candidates_raw:
        return None, {"method": "none", "candidates": []}, warnings

    style = build_style_profile(prompt_text, mood)
    fast_render = bool(style.get("fast_render", False))
    quality_priority = bool(style.get("quality_priority", False))
    smart_music = bool(style.get("smart_music", True))

    if fast_render:
        proxy_duration = min(max(5.0, source_duration), 6.5)
        proxy_max_points = 16
        top_n = 2
    elif quality_priority and smart_music:
        proxy_duration = min(max(7.0, source_duration), 9.5)
        proxy_max_points = 24
        top_n = 3
    else:
        proxy_duration = min(max(6.0, source_duration), 8.0)
        proxy_max_points = 20
        top_n = 2

    existing = {_relative_track_key(p): p for p in list_local_phonk_files(allow_phonk=allow_phonk)}
    candidate_inputs: list[dict[str, Any]] = []
    for c in top_candidates_raw[:top_n]:
        if not isinstance(c, dict):
            continue
        fname = str(c.get("file", "")).strip()
        if not fname:
            continue
        p = existing.get(fname)
        if not p:
            continue
        candidate_inputs.append({"candidate": c, "fname": fname, "path": p})

    def _evaluate_candidate(item: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        c = item["candidate"]
        fname = str(item["fname"])
        p = item["path"]
        base_score_raw = float(c.get("score", 0.0))
        profile_score = float(c.get("profile_score", 0.0))
        target_tempo = float(track_selection.get("target_tempo", MOOD_TO_BPM.get(mood, 130)))

        try:
            bpm, beat_markers, beat_warnings = detect_beats_from_audio_file(p, proxy_duration, mood)
        except Exception:
            bpm = int(MOOD_TO_BPM.get(mood, 130))
            beat_markers = generate_beat_markers(proxy_duration, bpm)
            beat_warnings = ["Proxy beat detection failed for one candidate; fallback markers used."]

        cut_points = _build_cut_points_from_markers(
            beat_markers,
            duration=proxy_duration,
            mood=mood,
            intensity=int(style.get("intensity", 1)),
            max_points=proxy_max_points,
            selective_beats=bool(style.get("selective_beats", True)),
            all_beats=bool(style.get("all_beats", False)),
            fast_render=bool(style.get("fast_render", False)),
            variation_seed=variation_seed,
        )
        rows = analyze_segment_intelligence(video_path, cut_points, source_fps=source_fps)
        proxy_score = _score_proxy_edit_quality(
            cut_points=cut_points,
            rows=rows,
            mood=mood,
            style=style,
            base_score_raw=base_score_raw,
            profile_score=profile_score,
            target_tempo=target_tempo,
            detected_bpm=float(bpm),
        )
        if LAST_SELECTED_TRACK_KEY and fname == LAST_SELECTED_TRACK_KEY and len(top_candidates_raw) > 1:
            proxy_score = float(max(0.0, proxy_score - 0.07))
        return {
            "file": fname,
            "proxy_score": round(proxy_score, 4),
            "proxy_bpm": int(bpm),
            "proxy_beats": len(beat_markers),
            "base_score": round(base_score_raw, 4),
            "profile_score": round(profile_score, 4),
        }, beat_warnings[:1]

    evaluated: list[dict[str, Any]] = []
    analysis_cfg = scene_analysis_config()
    if len(candidate_inputs) > 1 and analysis_cfg.workers > 1:
        worker_count = min(analysis_cfg.workers, len(candidate_inputs))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_evaluate_candidate, item) for item in candidate_inputs]
            for future in as_completed(futures):
                result, candidate_warnings = future.result()
                warnings.extend(candidate_warnings)
                if result is not None:
                    evaluated.append(result)
    else:
        for item in candidate_inputs:
            result, candidate_warnings = _evaluate_candidate(item)
            warnings.extend(candidate_warnings)
            if result is not None:
                evaluated.append(result)

    if not evaluated:
        return None, {"method": "none", "candidates": []}, warnings

    evaluated.sort(key=lambda d: (-float(d.get("proxy_score", 0.0)), str(d.get("file", ""))))
    best_name = str(evaluated[0]["file"])
    variation_enabled = variation_seed is not None
    if variation_enabled and len(evaluated) > 1:
        rng = random.Random(int(variation_seed) + 31)
        top_pool_size = min(3, len(evaluated))
        top_pool = evaluated[:top_pool_size]
        top_score = float(top_pool[0].get("proxy_score", 0.0))
        allowed = [item for item in top_pool if (top_score - float(item.get("proxy_score", 0.0))) <= 0.14]
        if not allowed:
            allowed = top_pool[:1]
        weights = [math.exp(max(0.0, float(item.get("proxy_score", 0.0)) * 2.6)) for item in allowed]
        picked = rng.choices(allowed, weights=weights, k=1)[0]
        best_name = str(picked.get("file", best_name))
    best_path = existing.get(best_name)
    info = {
        "method": "smart_score_plus_proxy_rank",
        "proxy_duration_sec": round(proxy_duration, 2),
        "variation_enabled": variation_enabled,
        "candidates": evaluated,
    }
    return best_path, info, warnings


def _ensure_uint8_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if arr.size:
            max_val = float(np.max(arr))
            if max_val <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _get_mediapipe_segmenter() -> Any | None:
    if _MEDIAPIPE_SEG_CACHE.get("loaded", False):
        return _MEDIAPIPE_SEG_CACHE.get("segmenter")
    _MEDIAPIPE_SEG_CACHE["loaded"] = True
    if not MEDIAPIPE_AVAILABLE or mp is None:
        _MEDIAPIPE_SEG_CACHE["segmenter"] = None
        return None
    try:
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    except Exception:
        segmenter = None
    _MEDIAPIPE_SEG_CACHE["segmenter"] = segmenter
    return segmenter


def _resolve_matte_model_path() -> Path | None:
    env_path = str(os.getenv("ARCHIVE_MATTE_MODEL", "")).strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            BASE_DIR / "models" / "matte.onnx",
            BASE_DIR / "models" / "u2netp.onnx",
            BASE_DIR / "models" / "modnet.onnx",
        ]
    )
    for p in candidates:
        if p and str(p).strip() and p.exists() and p.is_file():
            return p
    return None


def _get_matte_session() -> tuple[Any | None, str, tuple[int, int]]:
    if _MATTE_MODEL_CACHE.get("loaded", False):
        return (
            _MATTE_MODEL_CACHE.get("session"),
            str(_MATTE_MODEL_CACHE.get("input_name", "")),
            tuple(_MATTE_MODEL_CACHE.get("input_hw", (320, 320))),
        )

    _MATTE_MODEL_CACHE["loaded"] = True
    if not ONNXRUNTIME_AVAILABLE or ort is None:
        return None, "", (320, 320)
    model_path = _resolve_matte_model_path()
    if model_path is None:
        return None, "", (320, 320)

    try:
        providers = ["CPUExecutionProvider"]
        try:
            avail = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        ishape = session.get_inputs()[0].shape
        in_h = 320
        in_w = 320
        try:
            if len(ishape) >= 4:
                h_raw = ishape[2]
                w_raw = ishape[3]
                if isinstance(h_raw, int) and h_raw > 0:
                    in_h = int(h_raw)
                if isinstance(w_raw, int) and w_raw > 0:
                    in_w = int(w_raw)
        except Exception:
            pass
        _MATTE_MODEL_CACHE["session"] = session
        _MATTE_MODEL_CACHE["input_name"] = input_name
        _MATTE_MODEL_CACHE["input_hw"] = (in_h, in_w)
        _MATTE_MODEL_CACHE["path"] = str(model_path)
        return session, input_name, (in_h, in_w)
    except Exception:
        _MATTE_MODEL_CACHE["session"] = None
        return None, "", (320, 320)


def _predict_person_mask_mediapipe(small_bgr: np.ndarray) -> np.ndarray | None:
    seg = _get_mediapipe_segmenter()
    if seg is None:
        return None
    try:
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        segm = getattr(res, "segmentation_mask", None)
        if segm is None:
            return None
        mask = np.clip(np.asarray(segm, dtype=np.float32), 0.0, 1.0)
        mask = (mask * 255.0).astype(np.uint8)
        return mask
    except Exception:
        return None


def _predict_matte_mask_onnx(small_bgr: np.ndarray) -> np.ndarray | None:
    session, input_name, input_hw = _get_matte_session()
    if session is None or not input_name:
        return None
    try:
        ih, iw = int(input_hw[0]), int(input_hw[1])
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = session.run(None, {input_name: x})[0]
        out_arr = np.asarray(out)
        out_arr = np.squeeze(out_arr)
        if out_arr.ndim == 3:
            out_arr = out_arr[0]
        if out_arr.ndim != 2:
            return None
        out_arr = np.nan_to_num(out_arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        mn = float(np.min(out_arr))
        mx = float(np.max(out_arr))
        if mx - mn > 1e-6:
            out_arr = (out_arr - mn) / (mx - mn)
        else:
            out_arr = np.clip(out_arr, 0.0, 1.0)
        mask_small = (np.clip(out_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = small_bgr.shape[:2]
        if mask_small.shape[0] != h or mask_small.shape[1] != w:
            mask_small = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask_small
    except Exception:
        return None


def _extract_subject_soft_mask(
    frame: np.ndarray,
    center_x: float = 0.5,
    center_y: float = 0.42,
    prev_gray_small: np.ndarray | None = None,
    prev_mask_small: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _ensure_uint8_frame(frame)
    h, w = frame.shape[:2]
    sw = max(224, w // 3)
    sh = max(126, h // 3)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    edges = cv2.Canny(blur, 70, 165)
    edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)

    motion = np.zeros_like(edges)
    if prev_gray_small is not None and prev_gray_small.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray_small)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(
            motion,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex_abs = np.abs(lap)
    tex_thr = float(np.percentile(tex_abs, 80.0))
    tex_mask = (tex_abs >= tex_thr).astype(np.uint8) * 255
    tex_mask = cv2.morphologyEx(
        tex_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    combined = cv2.bitwise_or(edges, motion)
    combined = cv2.bitwise_or(combined, tex_mask)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    cx = int(np.clip(center_x, 0.1, 0.9) * sw)
    cy = int(np.clip(center_y, 0.1, 0.9) * sh)
    gate = np.zeros((sh, sw), dtype=np.uint8)
    cv2.ellipse(
        gate,
        (cx, cy),
        (max(28, int(sw * 0.34)), max(24, int(sh * 0.42))),
        0,
        0,
        360,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    combined = cv2.bitwise_and(combined, gate)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_small = np.zeros((sh, sw), dtype=np.uint8)
    best = None
    best_score = -1.0
    frame_area = float(max(1, sw * sh))
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < (frame_area * 0.006):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        extent = area / float(max(1, bw * bh))
        if area > (frame_area * 0.9) and extent > 0.72:
            continue
        m = cv2.moments(cnt)
        if m["m00"] > 1e-6:
            ccx = float(m["m10"] / m["m00"])
            ccy = float(m["m01"] / m["m00"])
        else:
            ccx, ccy = float(cx), float(cy)
        dist = math.sqrt(((ccx - cx) ** 2) + ((ccy - cy) ** 2))
        center_bonus = max(0.2, 1.0 - (dist / (math.sqrt(sw * sw + sh * sh) + 1e-6)))
        score = (area / frame_area) * 3.1 + (extent * 0.35) + (center_bonus * 0.8)
        if score > best_score:
            best_score = score
            best = cnt

    if best is not None:
        cv2.drawContours(mask_small, [best], -1, 255, thickness=cv2.FILLED)
    else:
        cv2.ellipse(
            mask_small,
            (cx, cy),
            (max(20, int(sw * 0.18)), max(24, int(sh * 0.26))),
            0,
            0,
            360,
            255,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    mask_small = cv2.morphologyEx(
        mask_small,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    # Advanced segmentation fusion (MediaPipe + ONNX matte) with OpenCV fallback.
    mp_mask = _predict_person_mask_mediapipe(small)
    onnx_mask = _predict_matte_mask_onnx(small)
    if (mp_mask is not None) or (onnx_mask is not None):
        fused = mask_small.astype(np.float32) * 0.46
        if mp_mask is not None:
            fused = np.maximum(fused, mp_mask.astype(np.float32) * 0.76)
        if onnx_mask is not None:
            fused = np.maximum(fused, onnx_mask.astype(np.float32) * 0.88)
        mask_small = np.clip(fused, 0, 255).astype(np.uint8)
        mask_small = cv2.morphologyEx(
            mask_small,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )

    # Temporal smoothing for stable masks across frames.
    if prev_mask_small is not None and prev_mask_small.shape == mask_small.shape:
        a = float(SUBJECT_MASK_TEMPORAL_ALPHA)
        mask_small = cv2.addWeighted(mask_small, 1.0 - a, prev_mask_small, a, 0)

    mask_small = cv2.GaussianBlur(mask_small, (0, 0), sigmaX=2.2, sigmaY=2.2)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask, gray, mask_small


def _apply_mask_roto_effect(
    clip: Any,
    center_x: float = 0.5,
    center_y: float = 0.42,
    intensity: int = 2,
) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    refresh_hz = 2.3
    mask_cache: dict[int, np.ndarray] = {}
    prev_gray_small: np.ndarray | None = None
    prev_mask_small: np.ndarray | None = None

    def _mask(get_frame: Any, t: float) -> np.ndarray:
        nonlocal prev_gray_small, prev_mask_small
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        cache_key = int(max(0.0, t) * refresh_hz)
        subj_mask = mask_cache.get(cache_key)
        if subj_mask is None:
            subj_mask, prev_gray_small, prev_mask_small = _extract_subject_soft_mask(
                frame,
                center_x=center_x,
                center_y=center_y,
                prev_gray_small=prev_gray_small,
                prev_mask_small=prev_mask_small,
            )
            mask_cache[cache_key] = subj_mask
            if len(mask_cache) > 12:
                mask_cache.pop(min(mask_cache.keys()), None)

        alpha = np.clip(subj_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        bg_blur = cv2.GaussianBlur(
            frame,
            (0, 0),
            sigmaX=2.6 + (0.8 * intensity),
            sigmaY=2.6 + (0.8 * intensity),
        )
        bg_dim = np.clip(frame.astype(np.float32) * (0.6 - (0.06 * intensity)), 0, 255).astype(np.uint8)
        bg_mix = cv2.addWeighted(bg_blur, 0.68, bg_dim, 0.32, 0)
        out = np.clip((frame.astype(np.float32) * alpha) + (bg_mix.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)
        return out

    return clip.fl(_mask)


def _apply_green_screen_chroma_key_effect(clip: Any, intensity: int = 2) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    prev_gray_small: np.ndarray | None = None
    prev_mask_small: np.ndarray | None = None

    def _chroma(get_frame: Any, t: float) -> np.ndarray:
        nonlocal prev_gray_small, prev_mask_small
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        subject_mask, prev_gray_small, prev_mask_small = _extract_subject_soft_mask(
            frame,
            center_x=0.5,
            center_y=0.42,
            prev_gray_small=prev_gray_small,
            prev_mask_small=prev_mask_small,
        )
        subject_alpha = np.clip(subject_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([34, 36, 22], dtype=np.uint8)
        upper1 = np.array([95, 255, 255], dtype=np.uint8)
        lower2 = np.array([25, 26, 20], dtype=np.uint8)
        upper2 = np.array([112, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)
        dom = (g - np.maximum(r, b)) > (14 + (3 * intensity))
        sat_gate = hsv[:, :, 1] > (32 + (3 * intensity))
        val_gate = hsv[:, :, 2] > 24
        mask_dom = ((dom & sat_gate & val_gate).astype(np.uint8) * 255)
        mask_bg = cv2.bitwise_or(mask_hsv, mask_dom)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

        protect = cv2.dilate(subject_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        mask_bg = cv2.bitwise_and(mask_bg, cv2.bitwise_not(protect))
        green_ratio = float(np.mean(mask_bg > 0))

        # If no reliable green screen is present, still provide a professional subject-background split.
        if green_ratio < 0.012:
            bg_blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=7.2 + (1.8 * intensity), sigmaY=7.2 + (1.8 * intensity))
            shift_x = int(3.0 * math.sin((2.0 * math.pi * t * 0.45) + 0.7))
            shift_y = int(2.0 * math.cos((2.0 * math.pi * t * 0.37) + 0.2))
            bg_blur = np.roll(np.roll(bg_blur, shift_y, axis=0), shift_x, axis=1)
            vignette = _build_vignette(h, w, strength=0.2 + (0.05 * intensity))
            bg_blur = np.clip(bg_blur.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
            split_alpha = np.clip(cv2.GaussianBlur(subject_mask, (0, 0), sigmaX=2.2, sigmaY=2.2).astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
            out = np.clip(
                (frame.astype(np.float32) * split_alpha) + (bg_blur.astype(np.float32) * (1.0 - split_alpha)),
                0,
                255,
            ).astype(np.uint8)
            edge = cv2.morphologyEx(subject_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8), iterations=1)
            edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.0, sigmaY=1.0)
            edge_col = np.zeros_like(frame, dtype=np.uint8)
            edge_col[:, :, 0] = np.clip(edge.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
            edge_col[:, :, 2] = np.clip(edge.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            return cv2.addWeighted(out, 1.0, edge_col, 0.17 + (0.03 * intensity), 0)

        alpha_bg = cv2.GaussianBlur(mask_bg, (0, 0), sigmaX=2.0 + (0.55 * intensity), sigmaY=2.0 + (0.55 * intensity))
        alpha_bg = np.clip(alpha_bg.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        replace_alpha = np.clip(alpha_bg * (1.0 - (subject_alpha * 0.94)), 0.0, 1.0)

        bg = cv2.GaussianBlur(frame, (0, 0), sigmaX=8.2 + (2.2 * intensity), sigmaY=8.2 + (2.2 * intensity))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = np.clip((xx / max(1.0, float(w - 1))), 0.0, 1.0)
        tint = np.zeros_like(frame, dtype=np.uint8)
        tint[:, :, 0] = np.clip(28 + (grad * 42), 0, 255).astype(np.uint8)
        tint[:, :, 1] = np.clip(18 + (grad * 48), 0, 255).astype(np.uint8)
        tint[:, :, 2] = np.clip(16 + (grad * 56), 0, 255).astype(np.uint8)
        bg = cv2.addWeighted(bg, 0.76, tint, 0.24, 0)

        out_f = np.clip(
            (frame.astype(np.float32) * (1.0 - replace_alpha))
            + (bg.astype(np.float32) * replace_alpha),
            0,
            255,
        )

        edge = cv2.morphologyEx(subject_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8), iterations=1)
        edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.1, sigmaY=1.1).astype(np.float32) / 255.0
        spill = np.clip((g.astype(np.float32) - np.maximum(r.astype(np.float32), b.astype(np.float32))) / 255.0, 0.0, 1.0)
        despill_alpha = np.clip(edge * spill * (0.5 + (0.08 * intensity)), 0.0, 0.65)
        out_g = out_f[:, :, 1]
        out_rb = (out_f[:, :, 0] + out_f[:, :, 2]) * 0.5
        out_f[:, :, 1] = (out_g * (1.0 - despill_alpha)) + (out_rb * despill_alpha)

        # Slight foreground preservation for cleaner subject detail and less matte halo.
        keep_alpha = np.clip(subject_alpha * 0.18, 0.0, 0.22)
        out_f = np.clip(
            (out_f * (1.0 - keep_alpha)) + (frame.astype(np.float32) * keep_alpha),
            0,
            255,
        )
        return out_f.astype(np.uint8)

    return clip.fl(_chroma)


def _apply_difference_matte_effect(clip: Any, intensity: int = 2) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    ref_gray_small: np.ndarray | None = None
    prev_gray_small: np.ndarray | None = None
    prev_mask_small: np.ndarray | None = None

    def _matte(get_frame: Any, t: float) -> np.ndarray:
        nonlocal ref_gray_small, prev_gray_small, prev_mask_small
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        sw = max(224, w // 3)
        sh = max(126, h // 3)
        subject_mask, prev_gray_small, prev_mask_small = _extract_subject_soft_mask(
            frame,
            center_x=0.5,
            center_y=0.42,
            prev_gray_small=prev_gray_small,
            prev_mask_small=prev_mask_small,
        )
        gray_small = cv2.cvtColor(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if ref_gray_small is None:
            ref_gray_small = gray_small.copy()
            return frame

        diff = cv2.absdiff(gray_small, ref_gray_small)
        thr = int(np.clip(np.percentile(diff, 74.0) + (3 * intensity), 18, 80))
        _, mask_small = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, np.ones((4, 4), dtype=np.uint8), iterations=1)

        diff_mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        combined_mask = np.clip((diff_mask.astype(np.float32) * 0.58) + (subject_mask.astype(np.float32) * 0.82), 0, 255).astype(np.uint8)
        if float(np.mean(combined_mask)) < 3.0:
            return frame

        alpha = cv2.GaussianBlur(combined_mask, (0, 0), sigmaX=2.0 + (0.5 * intensity), sigmaY=2.0 + (0.5 * intensity))
        alpha = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]

        bg = np.clip(frame.astype(np.float32) * (0.56 - (0.04 * intensity)), 0, 255).astype(np.uint8)
        out = np.clip((frame.astype(np.float32) * alpha) + (bg.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)

        edge = cv2.morphologyEx(combined_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8), iterations=1)
        edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.0, sigmaY=1.0)
        edge_color = np.zeros_like(frame, dtype=np.uint8)
        edge_color[:, :, 0] = np.clip(edge.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
        edge_color[:, :, 1] = np.clip(edge.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        out = cv2.addWeighted(out, 1.0, edge_color, 0.18 + (0.05 * intensity), 0)
        return out

    return clip.fl(_matte)


def _apply_object_inpaint_background_effect(
    clip: Any,
    center_x: float = 0.5,
    center_y: float = 0.42,
    intensity: int = 2,
) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    refresh_hz = 2.0
    mask_cache: dict[int, np.ndarray] = {}
    prev_gray_small: np.ndarray | None = None
    prev_mask_small: np.ndarray | None = None

    def _inpaint_bg(get_frame: Any, t: float) -> np.ndarray:
        nonlocal prev_gray_small, prev_mask_small
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        cache_key = int(max(0.0, t) * refresh_hz)
        subj_mask = mask_cache.get(cache_key)
        if subj_mask is None:
            subj_mask, prev_gray_small, prev_mask_small = _extract_subject_soft_mask(
                frame,
                center_x=center_x,
                center_y=center_y,
                prev_gray_small=prev_gray_small,
                prev_mask_small=prev_mask_small,
            )
            mask_cache[cache_key] = subj_mask
            if len(mask_cache) > 12:
                mask_cache.pop(min(mask_cache.keys()), None)

        area_ratio = float(np.mean(subj_mask > 0))
        if area_ratio < 0.015 or area_ratio > 0.8:
            return frame

        inpaint_mask = cv2.dilate(subj_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        bg_est = cv2.inpaint(frame, inpaint_mask, inpaintRadius=2 + intensity, flags=cv2.INPAINT_TELEA)
        bg_est = cv2.GaussianBlur(bg_est, (0, 0), sigmaX=2.2 + (0.8 * intensity), sigmaY=2.2 + (0.8 * intensity))

        alpha = cv2.GaussianBlur(subj_mask, (0, 0), sigmaX=1.6, sigmaY=1.6)
        alpha = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        out = np.clip((frame.astype(np.float32) * alpha) + (bg_est.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)
        return out

    return clip.fl(_inpaint_bg)


def _apply_shake_effect(clip: Any, amount_px: int = 8, frequency: float = 10.0) -> Any:
    def _shake(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        dx = int(amount_px * math.sin(2.0 * math.pi * frequency * t))
        dy = int(amount_px * math.cos(2.0 * math.pi * (frequency * 0.85) * t))
        return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)

    return clip.fl(_shake)


def _apply_rgb_split_effect(clip: Any, pixels: int = 4) -> Any:
    def _rgb_split(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        if len(frame.shape) < 3 or frame.shape[2] < 3:
            return frame
        out = frame.copy()
        out[:, :, 0] = np.roll(frame[:, :, 0], pixels, axis=1)
        out[:, :, 2] = np.roll(frame[:, :, 2], -pixels, axis=1)
        return out

    return clip.fl(_rgb_split)


def _apply_zoom_punch_effect(clip: Any, strength: float = 0.1) -> Any:
    def _zoom(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        factor = 1.0 + (strength * math.exp(-6.0 * t))
        if factor <= 1.001:
            return frame

        scaled_w = max(w + 2, int(w * factor))
        scaled_h = max(h + 2, int(h * factor))
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        x1 = max(0, (scaled_w - w) // 2)
        y1 = max(0, (scaled_h - h) // 2)
        return resized[y1 : y1 + h, x1 : x1 + w]

    return clip.fl(_zoom)


def _apply_motion_blur_effect(clip: Any, blur_px: int = 7) -> Any:
    kernel_size = max(3, blur_px if blur_px % 2 == 1 else blur_px + 1)

    def _motion_blur(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        blurred = cv2.GaussianBlur(frame, (kernel_size, 1), 0)
        return cv2.addWeighted(frame, 0.78, blurred, 0.22, 0)

    return clip.fl(_motion_blur)


def _apply_glow_effect(clip: Any, strength: float = 0.25) -> Any:
    sigma = 2.2

    def _glow(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return cv2.addWeighted(frame, 1.0, blurred, strength, 0)

    return clip.fl(_glow)


def _apply_film_grain_effect(clip: Any, amount: float = 7.0) -> Any:
    noise_cache: dict[tuple[int, int], np.ndarray] = {}

    def _grain(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t)).astype(np.float32)
        h, w = frame.shape[:2]
        key = (h, w)
        if key not in noise_cache:
            rng = np.random.default_rng(1337 + h + w)
            base_noise = rng.normal(0, amount, (h, w, 1)).astype(np.float32)
            noise_cache[key] = np.repeat(base_noise, 3, axis=2)

        pulse = 0.85 + (0.15 * math.sin(2.0 * math.pi * 3.0 * t))
        out = frame + (noise_cache[key] * pulse)
        return np.clip(out, 0, 255).astype(np.uint8)

    return clip.fl(_grain)


def _build_vignette(h: int, w: int, strength: float = 0.35) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, max(2, int(w)))
    y = np.linspace(-1.0, 1.0, max(2, int(h)))
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx * xx + yy * yy)
    mask = 1.0 - np.clip((dist - 0.2) / 0.8, 0.0, 1.0) * float(np.clip(strength, 0.0, 0.9))
    return mask.astype(np.float32)


def _apply_vignette_effect(clip: Any, strength: float = 0.35) -> Any:
    mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def _vignette(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t)).astype(np.float32)
        h, w = frame.shape[:2]
        key = (h, w)
        if key not in mask_cache:
            mask_cache[key] = _build_vignette(h, w, strength=strength)[..., None]
        out = frame * mask_cache[key]
        return np.clip(out, 0, 255).astype(np.uint8)

    return clip.fl(_vignette)


def _apply_edge_glow_effect(clip: Any, strength: float = 0.35) -> Any:
    edge_cache: dict[int, np.ndarray] = {}

    def _edge(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        key = int(max(0.0, t) * EDGE_GLOW_REFRESH_HZ)
        edge_color = edge_cache.get(key)
        if edge_color is None:
            small_w = max(160, w // 2)
            small_h = max(90, h // 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            edges_small = cv2.Canny(gray_small, 80, 180)
            edges = cv2.resize(edges_small, (w, h), interpolation=cv2.INTER_LINEAR)
            edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edge_cache[key] = edge_color
            if len(edge_cache) > 8:
                edge_cache.pop(min(edge_cache.keys()), None)
        edge_color = cv2.GaussianBlur(edge_color, (0, 0), sigmaX=1.2, sigmaY=1.2)
        return cv2.addWeighted(frame, 1.0, edge_color, strength, 0)

    return clip.fl(_edge)


def _apply_ken_burns_animation(clip: Any, zoom_strength: float = 0.08, pan_strength: float = 0.12) -> Any:
    duration = max(float(clip.duration or 0.001), 0.001)

    def _ken_burns(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        progress = min(max(t / duration, 0.0), 1.0)

        scale = 1.0 + (zoom_strength * progress)
        scaled_w = max(w + 2, int(w * scale))
        scaled_h = max(h + 2, int(h * scale))
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        max_x = max(0, scaled_w - w)
        max_y = max(0, scaled_h - h)
        x1 = int(max_x * (0.25 + pan_strength * progress))
        y1 = int(max_y * (0.35 + 0.2 * math.sin(progress * math.pi)))
        x1 = int(np.clip(x1, 0, max_x))
        y1 = int(np.clip(y1, 0, max_y))
        return resized[y1 : y1 + h, x1 : x1 + w]

    return clip.fl(_ken_burns)


def _apply_focus_zoom_animation(
    clip: Any,
    center_x: float = 0.5,
    center_y: float = 0.42,
    max_zoom: float = 0.14,
    pulse_hz: float = 2.1,
) -> Any:
    duration = max(float(clip.duration or 0.001), 0.001)
    center_x = float(np.clip(center_x, 0.15, 0.85))
    center_y = float(np.clip(center_y, 0.15, 0.85))
    max_zoom = float(np.clip(max_zoom, 0.04, 0.24))

    def _focus(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        progress = min(max(t / duration, 0.0), 1.0)

        # Smooth entry + rhythmic pulse to match edit style.
        entry = math.sin(progress * math.pi)
        pulse = 0.84 + (0.16 * math.sin(2.0 * math.pi * pulse_hz * t))
        factor = 1.0 + (max_zoom * entry * pulse)
        if factor <= 1.001:
            return frame

        scaled_w = max(w + 2, int(w * factor))
        scaled_h = max(h + 2, int(h * factor))
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        cx = center_x * scaled_w
        cy = center_y * scaled_h
        x1 = int(np.clip(cx - (w * 0.5), 0, max(0, scaled_w - w)))
        y1 = int(np.clip(cy - (h * 0.5), 0, max(0, scaled_h - h)))
        return resized[y1 : y1 + h, x1 : x1 + w]

    return clip.fl(_focus)


def _apply_fire_lines_effect(
    clip: Any,
    center_x: float = 0.5,
    center_y: float = 0.42,
    face_area: float = 0.0,
    intensity: int = 2,
) -> Any:
    center_x = float(np.clip(center_x, 0.12, 0.88))
    center_y = float(np.clip(center_y, 0.12, 0.88))
    face_area = float(np.clip(face_area, 0.0, 1.0))
    intensity = int(np.clip(intensity, 1, 3))
    refresh_hz = FIRE_BOUNDARY_REFRESH_HZ
    boundary_cache: dict[int, np.ndarray] = {}

    def _extract_boundary_map(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw = max(192, w // 3)
        sh = max(108, h // 3)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        cx = int(center_x * sw)
        cy = int(center_y * sh)
        rx = int(np.clip(sw * (0.14 + (0.42 * face_area)), sw * 0.14, sw * 0.45))
        ry = int(np.clip(sh * (0.2 + (0.5 * face_area)), sh * 0.2, sh * 0.58))
        x1 = int(np.clip(cx - rx, 0, sw - 1))
        y1 = int(np.clip(cy - ry, 0, sh - 1))
        x2 = int(np.clip(cx + rx, x1 + 1, sw))
        y2 = int(np.clip(cy + ry, y1 + 1, sh))

        boundary = np.zeros((sh, sw), dtype=np.uint8)
        roi = small[y1:y2, x1:x2]
        if roi.size <= 0:
            return boundary

        roi_h, roi_w = roi.shape[:2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges_roi = cv2.Canny(gray_roi, 70, 160)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        px = max(3, roi_w // 10)
        py = max(3, roi_h // 10)
        cx_r = roi_w // 2
        cy_r = roi_h // 2
        sx1 = max(0, cx_r - px)
        sy1 = max(0, cy_r - py)
        sx2 = min(roi_w, cx_r + px)
        sy2 = min(roi_h, cy_r + py)
        patch = hsv_roi[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            patch = hsv_roi
        med_h = float(np.median(patch[:, :, 0]))
        med_s = float(np.median(patch[:, :, 1]))
        med_v = float(np.median(patch[:, :, 2]))

        hue = hsv_roi[:, :, 0].astype(np.float32)
        sat = hsv_roi[:, :, 1].astype(np.float32)
        val = hsv_roi[:, :, 2].astype(np.float32)
        dh = np.abs(hue - med_h)
        dh = np.minimum(dh, 180.0 - dh) / 90.0
        ds = np.abs(sat - med_s) / 255.0
        dv = np.abs(val - med_v) / 255.0
        color_dist = (0.62 * dh) + (0.23 * ds) + (0.15 * dv)
        thr = float(np.clip(0.34 - (0.03 * intensity), 0.22, 0.36))
        color_mask = (color_dist < thr).astype(np.uint8) * 255
        color_boundary = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        combined = cv2.bitwise_or(edges_roi, color_boundary)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
        combined = cv2.dilate(combined, np.ones((2, 2), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, roi_w * roi_h))
        best = None
        best_score = -1.0
        max_dist = math.sqrt((roi_w * roi_w) + (roi_h * roi_h)) + 1e-6
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < (roi_area * 0.01):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            touch_border = (bx <= 1) or (by <= 1) or ((bx + bw) >= (roi_w - 1)) or ((by + bh) >= (roi_h - 1))
            extent = area / float(max(1, bw * bh))
            if touch_border and area > (roi_area * 0.08):
                continue
            if area > (roi_area * 0.76) and extent > 0.7:
                continue

            m = cv2.moments(cnt)
            if m["m00"] > 1e-6:
                ccx = float(m["m10"] / m["m00"])
                ccy = float(m["m01"] / m["m00"])
            else:
                ccx, ccy = float(cx_r), float(cy_r)
            cdist = math.sqrt(((ccx - cx_r) ** 2) + ((ccy - cy_r) ** 2)) / max_dist
            center_weight = max(0.2, 1.0 - (cdist * 1.6))
            score = area * center_weight
            if score > best_score:
                best_score = score
                best = cnt

        if best is not None:
            thick = max(1, 1 + (intensity // 2))
            roi_boundary = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.drawContours(roi_boundary, [best], -1, 255, thickness=thick)
            boundary[y1:y2, x1:x2] = roi_boundary
        else:
            # Fallback: global contour search to avoid drawing center boxes.
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            global_edges = cv2.Canny(gray_small, 76, 168)
            global_edges = cv2.morphologyEx(
                global_edges,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            global_edges = cv2.dilate(global_edges, np.ones((2, 2), dtype=np.uint8), iterations=1)

            g_contours, _ = cv2.findContours(global_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_global = None
            best_global_score = -1.0
            full_area = float(max(1, sw * sh))
            max_dist_g = math.sqrt((sw * sw) + (sh * sh)) + 1e-6
            for cnt_g in g_contours:
                area_g = float(cv2.contourArea(cnt_g))
                if area_g < (full_area * 0.01):
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt_g)
                extent_g = area_g / float(max(1, bw * bh))
                if area_g > (full_area * 0.86) and extent_g > 0.72:
                    continue
                m_g = cv2.moments(cnt_g)
                if m_g["m00"] > 1e-6:
                    gcx = float(m_g["m10"] / m_g["m00"])
                    gcy = float(m_g["m01"] / m_g["m00"])
                else:
                    gcx, gcy = float(cx), float(cy)
                cdist_g = math.sqrt(((gcx - cx) ** 2) + ((gcy - cy) ** 2)) / max_dist_g
                center_weight_g = max(0.2, 1.0 - (cdist_g * 1.45))
                score_g = area_g * center_weight_g
                if score_g > best_global_score:
                    best_global_score = score_g
                    best_global = cnt_g

            if best_global is not None:
                cv2.drawContours(boundary, [best_global], -1, 255, thickness=max(1, 1 + (intensity // 2)))

        return boundary

    def _fire(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]

        cache_key = int(max(0.0, t) * refresh_hz)
        boundary = boundary_cache.get(cache_key)
        if boundary is None:
            boundary = _extract_boundary_map(frame)
            boundary_cache[cache_key] = boundary
            if len(boundary_cache) > 10:
                oldest = min(boundary_cache.keys())
                boundary_cache.pop(oldest, None)

        pulse = 0.74 + (0.26 * math.sin((2.0 * math.pi * 3.4 * t) + 0.6))
        hot_line = np.clip(boundary.astype(np.float32) * pulse, 0, 255).astype(np.uint8)
        outer = cv2.GaussianBlur(
            hot_line,
            (0, 0),
            sigmaX=1.25 + (0.45 * intensity),
            sigmaY=1.25 + (0.45 * intensity),
        )
        inner = cv2.dilate(hot_line, np.ones((2, 2), dtype=np.uint8), iterations=1)

        fire_layer_small = np.zeros((boundary.shape[0], boundary.shape[1], 3), dtype=np.uint8)
        fire_layer_small[:, :, 2] = np.clip(outer.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
        fire_layer_small[:, :, 1] = np.clip(
            (outer.astype(np.float32) * (0.56 + (0.14 * intensity))) + (inner.astype(np.float32) * 0.22),
            0,
            255,
        ).astype(np.uint8)
        fire_layer_small[:, :, 0] = np.clip(inner.astype(np.float32) * 0.12, 0, 255).astype(np.uint8)

        spark_pulse = 0.5 + (0.5 * math.sin((2.0 * math.pi * (4.6 + (0.5 * intensity)) * t) + 1.15))
        electric_core = cv2.GaussianBlur(inner, (0, 0), sigmaX=0.7 + (0.2 * intensity), sigmaY=0.7 + (0.2 * intensity))
        electric_strength = float(0.24 + (0.16 * spark_pulse))
        fire_layer_small[:, :, 0] = np.clip(
            fire_layer_small[:, :, 0].astype(np.float32) + (electric_core.astype(np.float32) * (0.85 * electric_strength)),
            0,
            255,
        ).astype(np.uint8)
        fire_layer_small[:, :, 1] = np.clip(
            fire_layer_small[:, :, 1].astype(np.float32) + (electric_core.astype(np.float32) * (0.35 * electric_strength)),
            0,
            255,
        ).astype(np.uint8)
        fire_layer_small[:, :, 2] = np.clip(
            fire_layer_small[:, :, 2].astype(np.float32) + (electric_core.astype(np.float32) * (0.18 * electric_strength)),
            0,
            255,
        ).astype(np.uint8)

        white_core = cv2.GaussianBlur(
            cv2.erode(inner, np.ones((2, 2), dtype=np.uint8), iterations=1),
            (0, 0),
            sigmaX=0.5 + (0.12 * intensity),
            sigmaY=0.5 + (0.12 * intensity),
        ).astype(np.float32)
        fire_layer_small[:, :, 0] = np.clip(fire_layer_small[:, :, 0].astype(np.float32) + (white_core * 0.68), 0, 255).astype(np.uint8)
        fire_layer_small[:, :, 1] = np.clip(fire_layer_small[:, :, 1].astype(np.float32) + (white_core * 0.78), 0, 255).astype(np.uint8)
        fire_layer_small[:, :, 2] = np.clip(fire_layer_small[:, :, 2].astype(np.float32) + (white_core * 0.9), 0, 255).astype(np.uint8)

        fire_layer = cv2.resize(fire_layer_small, (w, h), interpolation=cv2.INTER_LINEAR)
        glow_strength = 0.24 + (0.1 * intensity)
        out = cv2.addWeighted(frame, 1.0, fire_layer, glow_strength, 0)
        return out

    return clip.fl(_fire)


def _apply_black_white_pulse_effect(clip: Any, intensity: int = 2, pulse_hz: float = 2.6) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    base_strength = 0.68 + (0.1 * intensity)

    def _bw(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pulse = 0.82 + (0.18 * math.sin((2.0 * math.pi * pulse_hz * t) + 0.4))
        strength = float(np.clip(base_strength * pulse, 0.0, 1.0))
        out = cv2.addWeighted(frame, 1.0 - strength, gray3, strength, 0)
        return out

    return clip.fl(_bw)


def _apply_anime_tone_effect(clip: Any, strength: float = 0.35) -> Any:
    clip_limit = float(np.clip(0.015 + (0.03 * strength), 0.01, 0.07))
    clahe = cv2.createCLAHE(clipLimit=max(1.5, 1.5 + (strength * 2.0)), tileGridSize=(8, 8))

    def _anime_tone(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0].astype(np.float32) / 255.0

        # `equalize_adapthist` is visually good but expensive per-frame.
        if (not USE_FAST_ANIME_TONE) and SKIMAGE_AVAILABLE and exposure is not None:
            y_eq = exposure.equalize_adapthist(y, clip_limit=clip_limit)
            y_mix = np.clip((1.0 - strength) * y + (strength * y_eq), 0.0, 1.0)
            yuv[:, :, 0] = (y_mix * 255.0).astype(np.uint8)
        else:
            y_u8 = (y * 255.0).astype(np.uint8)
            y_clahe = clahe.apply(y_u8).astype(np.float32) / 255.0
            y_mix = np.clip((1.0 - strength) * y + (strength * y_clahe), 0.0, 1.0)
            yuv[:, :, 0] = (y_mix * 255.0).astype(np.uint8)

        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return out

    return clip.fl(_anime_tone)


def _apply_chroma_pulse_effect(clip: Any, max_shift: int = 4, hz: float = 4.0) -> Any:
    max_shift = max(1, int(max_shift))

    def _pulse(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        shift = int(round(((math.sin(2.0 * math.pi * hz * t) + 1.0) * 0.5) * max_shift))
        out = frame.copy()
        if out.shape[2] >= 3:
            out[:, :, 1] = np.roll(frame[:, :, 1], shift, axis=0)
            out[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)
        return out

    return clip.fl(_pulse)


def _blend_layers(base: np.ndarray, layer: np.ndarray, alpha: float, mode: str = "screen") -> np.ndarray:
    a = np.clip(base.astype(np.float32) / 255.0, 0.0, 1.0)
    b = np.clip(layer.astype(np.float32) / 255.0, 0.0, 1.0)
    if mode == "add":
        mixed = np.clip(a + (b * alpha), 0.0, 1.0)
    elif mode == "overlay":
        over = np.where(a < 0.5, 2.0 * a * b, 1.0 - (2.0 * (1.0 - a) * (1.0 - b)))
        mixed = np.clip((a * (1.0 - alpha)) + (over * alpha), 0.0, 1.0)
    else:
        # screen blend
        screened = 1.0 - ((1.0 - a) * (1.0 - b))
        mixed = np.clip((a * (1.0 - alpha)) + (screened * alpha), 0.0, 1.0)
    return (mixed * 255.0).astype(np.uint8)


def _apply_particle_atmosphere_effect(clip: Any, mood: str, intensity: int = 2) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    mode = {
        "sad": "snow",
        "chill": "smoke",
        "dark": "rain",
        "dramatic": "sparks",
        "hype": "sparks",
    }.get(mood, "sparks")
    if mood == "dark" and intensity >= 2:
        mode = "rain"

    seed = 137 + (intensity * 17) + sum(ord(c) for c in mode)
    rng = np.random.default_rng(seed)
    count = 40 + (25 * intensity)
    base_x = rng.random(count)
    base_y = rng.random(count)
    speed = rng.uniform(0.18, 0.65, size=count) * (1.0 + (0.2 * intensity))
    drift = rng.uniform(-0.22, 0.22, size=count)
    phase = rng.uniform(0.0, math.pi * 2.0, size=count)
    size = rng.uniform(1.0, 3.2 + (0.6 * intensity), size=count)
    flicker = rng.uniform(0.6, 1.25, size=count)

    palette = {
        "snow": (235, 235, 255),
        "rain": (170, 210, 255),
        "sparks": (95, 190, 255),
        "smoke": (165, 165, 178),
    }
    color = palette.get(mode, (200, 220, 255))
    alpha = 0.16 + (0.06 * intensity)

    def _particles(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        x_vals = ((base_x + (drift * np.sin((0.9 * t) + phase))) * w) % w
        y_vals = ((base_y + (speed * t)) * h) % h
        pulse = 0.72 + (0.28 * np.sin((2.0 * math.pi * 2.8 * t) + phase))

        for i in range(count):
            xi = int(x_vals[i])
            yi = int(y_vals[i])
            brightness = float(np.clip(0.45 + (0.55 * pulse[i] * flicker[i]), 0.25, 1.0))
            c = (
                int(np.clip(color[0] * brightness, 0, 255)),
                int(np.clip(color[1] * brightness, 0, 255)),
                int(np.clip(color[2] * brightness, 0, 255)),
            )
            if mode == "rain":
                ln = int(np.clip(size[i] * 2.5, 2, 7))
                cv2.line(overlay, (xi, yi), (xi + int(drift[i] * 7), yi + ln), c, 1, lineType=cv2.LINE_AA)
            elif mode == "smoke":
                r = int(np.clip(size[i] * 1.7, 1, 5))
                cv2.circle(overlay, (xi, yi), r, c, thickness=-1, lineType=cv2.LINE_AA)
            else:
                r = int(np.clip(size[i], 1, 4))
                cv2.circle(overlay, (xi, yi), r, c, thickness=-1, lineType=cv2.LINE_AA)

        blur_sigma = 0.8 + (0.3 * intensity)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        return _blend_layers(frame, overlay, alpha=alpha, mode="screen")

    return clip.fl(_particles)


def _apply_energy_aura_effect(
    clip: Any,
    center_x: float = 0.5,
    center_y: float = 0.42,
    intensity: int = 2,
) -> Any:
    center_x = float(np.clip(center_x, 0.1, 0.9))
    center_y = float(np.clip(center_y, 0.1, 0.9))
    intensity = int(np.clip(intensity, 1, 3))
    refresh_hz = 2.2
    aura_cache: dict[int, np.ndarray] = {}

    def _build_subject_mask(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw = max(224, w // 3)
        sh = max(126, h // 3)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
        edges = cv2.Canny(blur, 70, 165)
        edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

        cx = int(center_x * sw)
        cy = int(center_y * sh)
        dist_y = np.abs(np.arange(sh, dtype=np.float32)[:, None] - cy) / max(1.0, sh * 0.55)
        dist_x = np.abs(np.arange(sw, dtype=np.float32)[None, :] - cx) / max(1.0, sw * 0.55)
        gate = ((dist_x + dist_y) < 1.24).astype(np.uint8) * 255
        gated = cv2.bitwise_and(edges, gate)
        contours, _ = cv2.findContours(gated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((sh, sw), dtype=np.uint8)
        if contours:
            areas = [(cv2.contourArea(c), c) for c in contours]
            areas.sort(key=lambda x: x[0], reverse=True)
            keep = [c for area, c in areas[:4] if area > 36]
            if keep:
                cv2.drawContours(mask, keep, -1, 255, thickness=cv2.FILLED)
        if int(mask.sum()) <= 0:
            cv2.circle(mask, (cx, cy), int(min(sw, sh) * 0.2), 255, thickness=-1, lineType=cv2.LINE_AA)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), dtype=np.uint8), iterations=1)
        return mask

    def _aura(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        key = int(max(0.0, t) * refresh_hz)
        mask = aura_cache.get(key)
        if mask is None:
            mask = _build_subject_mask(frame)
            aura_cache[key] = mask
            if len(aura_cache) > 12:
                aura_cache.pop(min(aura_cache.keys()), None)

        mask_big = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        core = cv2.GaussianBlur(mask_big, (0, 0), sigmaX=2.5 + (0.7 * intensity), sigmaY=2.5 + (0.7 * intensity))
        glow = cv2.GaussianBlur(mask_big, (0, 0), sigmaX=8.0 + (2.2 * intensity), sigmaY=8.0 + (2.2 * intensity))
        pulse = 0.7 + (0.3 * math.sin((2.0 * math.pi * (2.8 + (0.4 * intensity)) * t) + 0.9))
        aura = np.zeros((h, w, 3), dtype=np.uint8)
        aura[:, :, 0] = np.clip((glow.astype(np.float32) * (0.8 + (0.2 * pulse))), 0, 255).astype(np.uint8)
        aura[:, :, 1] = np.clip((glow.astype(np.float32) * (0.48 + (0.18 * pulse))), 0, 255).astype(np.uint8)
        aura[:, :, 2] = np.clip((core.astype(np.float32) * (0.24 + (0.12 * pulse))), 0, 255).astype(np.uint8)
        out = _blend_layers(frame, aura, alpha=0.34 + (0.08 * intensity), mode="screen")
        return out

    return clip.fl(_aura)


def _apply_turbulent_displace_effect(clip: Any, intensity: int = 2) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    map_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    amp = 1.3 + (1.1 * intensity)
    freq_x = 0.016 + (0.004 * intensity)
    freq_y = 0.018 + (0.004 * intensity)
    speed = 1.2 + (0.35 * intensity)

    def _warp(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        cache_key = (w, h, int(t * 9))
        maps = map_cache.get(cache_key)
        if maps is None:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dx = amp * np.sin((yy * freq_y) + (t * speed * 1.4)) + (amp * 0.7 * np.cos((xx * (freq_x * 0.6)) - (t * speed)))
            dy = amp * np.cos((xx * freq_x) + (t * speed * 1.1)) + (amp * 0.65 * np.sin((yy * (freq_y * 0.7)) + (t * speed * 0.8)))
            map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
            map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
            maps = (map_x, map_y)
            map_cache[cache_key] = maps
            if len(map_cache) > 8:
                map_cache.pop(min(map_cache.keys(), key=lambda k: k[2]), None)
        warped = cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return warped

    return clip.fl(_warp)


def _apply_posterize_time_effect(clip: Any, target_fps: int = 15) -> Any:
    target_fps = int(np.clip(target_fps, 8, 24))
    step = 1.0 / float(target_fps)

    def _posterize(get_frame: Any, t: float) -> np.ndarray:
        qt = math.floor(max(0.0, float(t)) / step) * step
        return _ensure_uint8_frame(get_frame(qt))

    return clip.fl(_posterize)


def _apply_light_leak_overlay_effect(clip: Any, mood: str, intensity: int = 2) -> Any:
    intensity = int(np.clip(intensity, 1, 3))
    mode = "screen"
    if mood == "hype":
        mode = "add"
    elif mood == "dark":
        mode = "overlay"
    alpha = 0.07 + (0.03 * intensity)

    # Cinematic palettes by mood, avoiding flat yellow circles.
    leak_palette = {
        "hype": ((60, 170, 255), (255, 130, 80)),
        "dark": ((75, 95, 175), (155, 90, 165)),
        "dramatic": ((90, 145, 235), (210, 120, 120)),
        "sad": ((95, 115, 170), (165, 140, 175)),
        "chill": ((85, 160, 215), (145, 170, 190)),
    }.get(mood, ((80, 150, 220), (200, 120, 120)))

    def _leak(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t))
        h, w = frame.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        # Two off-screen moving sources create natural light leaks.
        side = -1.0 if math.sin(0.42 * t) >= 0 else 1.0
        cx1 = ((-0.18 if side < 0 else 1.18) + (0.05 * math.sin(0.7 * t))) * w
        cy1 = (0.22 + (0.28 * (0.5 + (0.5 * math.sin(0.34 * t + 0.6))))) * h
        cx2 = ((1.16 if side < 0 else -0.16) + (0.04 * math.cos(0.63 * t + 1.2))) * w
        cy2 = (0.58 + (0.26 * (0.5 + (0.5 * math.cos(0.29 * t + 0.9))))) * h

        d1 = np.sqrt(((xx - cx1) ** 2) + ((yy - cy1) ** 2))
        d2 = np.sqrt(((xx - cx2) ** 2) + ((yy - cy2) ** 2))
        r1 = max(1.0, 0.72 * max(w, h))
        r2 = max(1.0, 0.66 * max(w, h))
        m1 = np.clip(1.0 - (d1 / r1), 0.0, 1.0)
        m2 = np.clip(1.0 - (d2 / r2), 0.0, 1.0)

        # Add directional streak shape to mimic lens leak strips.
        angle = 0.32 + (0.18 * math.sin(0.41 * t))
        strip = np.clip(1.0 - np.abs((xx * math.cos(angle) + yy * math.sin(angle)) - (0.58 * w)) / (0.38 * w), 0.0, 1.0)
        strip = cv2.GaussianBlur(strip.astype(np.float32), (0, 0), sigmaX=11.0, sigmaY=11.0)

        pulse = 0.8 + (0.2 * math.sin((2.0 * math.pi * 0.75 * t) + 0.2))
        a = np.clip(((m1 * 0.72) + (m2 * 0.55) + (strip * 0.45)) * pulse, 0.0, 1.0)

        c1 = leak_palette[0]
        c2 = leak_palette[1]
        leak = np.zeros((h, w, 3), dtype=np.uint8)
        leak[:, :, 0] = np.clip((m1 * c1[0]) + (m2 * c2[0]) + (strip * 35), 0, 255).astype(np.uint8)
        leak[:, :, 1] = np.clip((m1 * c1[1]) + (m2 * c2[1]) + (strip * 50), 0, 255).astype(np.uint8)
        leak[:, :, 2] = np.clip((m1 * c1[2]) + (m2 * c2[2]) + (strip * 70), 0, 255).astype(np.uint8)
        leak = cv2.GaussianBlur(leak, (0, 0), sigmaX=3.8, sigmaY=3.8)
        leak = np.clip(leak.astype(np.float32) * a[..., None], 0, 255).astype(np.uint8)
        return _blend_layers(frame, leak, alpha=alpha, mode=mode)

    return clip.fl(_leak)


def _apply_color_grade(
    clip: Any,
    mood: str,
    intensity: int,
    scene_energy: float = 0.5,
    brightness: float = 0.5,
    contrast: float = 0.5,
) -> Any:
    mood_params = {
        "hype": {"temp": 0.007, "tint": 0.004, "sat": 1.14, "cont": 1.04},
        "dark": {"temp": -0.006, "tint": -0.003, "sat": 0.99, "cont": 1.05},
        "dramatic": {"temp": 0.005, "tint": 0.001, "sat": 1.12, "cont": 1.045},
        "sad": {"temp": -0.005, "tint": -0.002, "sat": 0.98, "cont": 0.995},
        "chill": {"temp": 0.002, "tint": 0.0, "sat": 1.06, "cont": 1.01},
    }.get(mood, {"temp": 0.0, "tint": 0.0, "sat": 1.04, "cont": 1.0})

    strength = float(np.clip(0.35 + (0.08 * intensity) + (0.07 * scene_energy), 0.28, 0.7))
    sat_mult = float(np.clip(1.0 + ((mood_params["sat"] - 1.0) * strength), 0.96, 1.18))
    contrast_soften = max(0.0, contrast - 0.62) * 0.08
    cont_gain = float(np.clip(1.0 + ((mood_params["cont"] - 1.0) * strength) - contrast_soften, 0.96, 1.09))
    temp_shift = float(np.clip(mood_params["temp"] * strength, -0.012, 0.012))
    tint_shift = float(np.clip(mood_params["tint"] * strength, -0.01, 0.01))
    shadow_lift = float(np.clip((0.34 - contrast) * 0.018, 0.0, 0.016))

    def _grade(get_frame: Any, t: float) -> np.ndarray:
        frame = _ensure_uint8_frame(get_frame(t)).astype(np.float32) / 255.0

        frame = np.clip((frame - 0.5) * cont_gain + 0.5, 0.0, 1.0)

        if shadow_lift > 0:
            frame = np.clip(frame + shadow_lift * (1.0 - frame), 0.0, 1.0)

        luma = (0.114 * frame[:, :, 0]) + (0.587 * frame[:, :, 1]) + (0.299 * frame[:, :, 2])

        # Mild white balance in BGR domain.
        b_gain = 1.0 - temp_shift + (tint_shift * 0.6)
        g_gain = 1.0 - (abs(tint_shift) * 0.35)
        r_gain = 1.0 + temp_shift + tint_shift
        frame[:, :, 0] *= b_gain
        frame[:, :, 1] *= g_gain
        frame[:, :, 2] *= r_gain
        frame = np.clip(frame, 0.0, 1.0)

        # Saturation via luma mix (cleaner than HSV round-trip for this pipeline).
        frame = np.clip(luma[..., None] + ((frame - luma[..., None]) * sat_mult), 0.0, 1.0)

        # Filmic highlight rolloff to avoid dirty clipping.
        hi = np.clip((luma - 0.74) / 0.26, 0.0, 1.0)[..., None]
        frame = np.clip(frame - (hi * 0.08 * frame), 0.0, 1.0)

        out = (frame * 255.0).astype(np.uint8)
        return out

    return clip.fl(_grade)


def _make_transition_clip(size: tuple[int, int], fps: int, mood: str, index: int, style: dict[str, Any]) -> Any:
    intensity = int(style.get("intensity", 1))
    transition_hint = str(style.get("transition_hint", "cut"))
    wants_flash = bool(style.get("flash", False))
    wants_glitch = bool(style.get("glitch", False))
    wants_overlap = bool(style.get("overlap_transitions", False))
    wants_mask_transition = bool(style.get("mask_transition", False))
    wants_pro_pack = bool(style.get("pro_edit_pack", False))

    if mood == "hype":
        palette = [(255, 255, 255), (165, 235, 255), (255, 220, 170)]
        color = palette[index % len(palette)]
        duration = 0.028 + (0.006 * intensity)
        opacity = 0.78 + (0.05 * intensity)
    elif mood == "dark":
        palette = [(25, 0, 45), (12, 12, 12), (80, 10, 40)]
        color = palette[index % len(palette)]
        duration = 0.03 + (0.008 * intensity)
        opacity = 0.58 + (0.06 * intensity)
    elif mood == "dramatic":
        palette = [(250, 250, 250), (0, 0, 0), (255, 246, 220)]
        color = palette[index % len(palette)]
        duration = 0.04 + (0.008 * intensity)
        opacity = 0.52 + (0.05 * intensity)
    elif mood == "sad":
        color = (240, 240, 255)
        duration = 0.04 + (0.006 * intensity)
        opacity = 0.24 + (0.04 * intensity)
    else:
        if not wants_flash:
            return None
        color = (255, 255, 255)
        duration = 0.022 + (0.005 * intensity)
        opacity = 0.2 + (0.04 * intensity)

    if transition_hint == "flash":
        color = (255, 255, 255)
        opacity = min(0.94, opacity + 0.12)
        duration = min(0.075, duration + 0.008)
    elif transition_hint == "glitch":
        wants_glitch = True
        opacity = min(0.9, opacity + 0.08)
    elif transition_hint.startswith("wipe_"):
        wants_mask_transition = True
        wants_overlap = True
        opacity = min(0.8, opacity + 0.04)
    elif transition_hint == "soft_push":
        wants_overlap = True
        duration = min(0.11, duration + 0.02)
        opacity = min(0.62, opacity)

    if wants_glitch and index % 2 == 1:
        color = (color[2], color[0], color[1])

    if wants_overlap:
        duration = min(0.095, duration + (0.012 + (0.004 * max(0, intensity - 1))))
        opacity = min(0.92, opacity + 0.04)

    if wants_mask_transition:
        try:
            w, h = int(size[0]), int(size[1])
            color_rgb = np.array([color[0], color[1], color[2]], dtype=np.float32)
            band = max(48, int(0.14 * w))
            edge_band = max(7, int(0.03 * w))
            inv = -1 if index % 2 else 1
            variant_count = 3 if wants_pro_pack or transition_hint.startswith("wipe_") else 1
            variant = (index + intensity + (1 if mood == "dark" else 0)) % variant_count
            if transition_hint == "wipe_left" or transition_hint == "wipe_right":
                variant = 0

            def _make_frame(t: float) -> np.ndarray:
                p = float(np.clip(t / max(0.001, duration), 0.0, 1.0))
                x = np.arange(w, dtype=np.float32)[None, :]
                y = np.arange(h, dtype=np.float32)[:, None]
                if variant == 1:
                    cx = (0.5 + (0.12 * math.sin(p * math.pi * 2.0))) * w
                    cy = (0.5 + (0.1 * math.cos(p * math.pi * 1.6))) * h
                    dx = x - cx
                    dy = y - cy
                    dist = np.sqrt((dx * dx) + (dy * dy))
                    max_r = math.sqrt((w * w) + (h * h)) * 0.6
                    radius = max(1.0, max_r * p)
                    fill = np.clip(1.0 - (dist / radius), 0.0, 1.0).astype(np.float32)
                    edge = np.clip(1.0 - (np.abs(dist - radius) / float(edge_band * 1.4)), 0.0, 1.0).astype(np.float32)
                elif variant == 2:
                    center = int(((p * (h + (2 * band))) - band))
                    if inv < 0:
                        center = h - center
                    wave = 0.08 * h * np.sin((x / max(1.0, w)) * math.pi * 2.0 + (p * 6.0))
                    dist = np.abs((y + wave) - center)
                    fill = np.clip(1.0 - (dist / float(band)), 0.0, 1.0).astype(np.float32)
                    edge = np.clip(1.0 - (dist / float(edge_band)), 0.0, 1.0).astype(np.float32)
                else:
                    center = int(((p * (w + (2 * band))) - band))
                    if transition_hint == "wipe_left":
                        center = w - center
                    elif transition_hint == "wipe_right":
                        center = center
                    elif inv < 0:
                        center = w - center
                    diag = x + (0.34 * y)
                    dist = np.abs(diag - center)
                    fill = np.clip(1.0 - (dist / float(band)), 0.0, 1.0).astype(np.float32)
                    edge = np.clip(1.0 - (dist / float(edge_band)), 0.0, 1.0).astype(np.float32)
                fill = cv2.GaussianBlur(fill, (0, 0), sigmaX=2.2, sigmaY=2.2)
                edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.2, sigmaY=1.2)

                frame_f = np.zeros((h, w, 3), dtype=np.float32)
                frame_f += fill[..., None] * (color_rgb[None, None, :] * 0.25)
                edge_color = np.array([245.0, 242.0, 255.0], dtype=np.float32)
                if mood == "dark":
                    edge_color = np.array([170.0, 130.0, 245.0], dtype=np.float32)
                frame_f += edge[..., None] * edge_color[None, None, :] * 0.92
                # Slight accent streak gives a more "AE masking" feel.
                diag_for_streak = x + (0.34 * y)
                streak = np.clip(1.0 - (np.abs(diag_for_streak - (center + (edge_band * 1.6))) / float(edge_band * 2.1)), 0.0, 1.0)
                frame_f[:, :, 1] += streak * 20.0
                frame_f[:, :, 2] += streak * 30.0
                frame = np.clip(frame_f, 0, 255).astype(np.uint8)
                return frame

            clip_obj = (
                VideoClip(make_frame=_make_frame, duration=duration)
                .set_fps(fps)
                .set_opacity(min(0.62, (0.24 + (0.07 * intensity) + (0.06 if wants_overlap else 0.0))))
            )
            clip_obj.size = size
            return clip_obj
        except Exception:
            # Safety fallback for environments where dynamic transition clips fail.
            return (
                ColorClip(size=size, color=color)
                .set_duration(duration)
                .set_fps(fps)
                .set_opacity(opacity)
            )

    return (
        ColorClip(size=size, color=color)
        .set_duration(duration)
        .set_fps(fps)
        .set_opacity(opacity)
    )


def _apply_mood_effects(
    segment: Any,
    mood: str,
    index: int,
    style: dict[str, Any],
    intelligence: dict[str, Any] | None = None,
    effect_hints: dict[str, Any] | None = None,
    offload_to_ffmpeg: bool = False,
) -> tuple[Any, dict[str, Any]]:
    intensity = int(style.get("intensity", 1))
    segment_duration = float(segment.duration or 0.0)
    is_short_segment = segment_duration < HEAVY_EFFECT_MIN_SEG_SEC
    intelligence = intelligence or {}
    effect_hints = effect_hints or {}
    scene_energy = float(intelligence.get("scene_energy", 0.5))
    motion = float(intelligence.get("motion", 0.5))
    flow_mag = float(intelligence.get("flow_mag", 0.0))
    scene_change = float(intelligence.get("scene_change", 0.0))
    contrast = float(intelligence.get("contrast", 0.5))
    brightness = float(intelligence.get("brightness", 0.5))
    saturation = float(intelligence.get("saturation", 0.5))
    face_count = int(intelligence.get("face_count", 0))
    face_detected = bool(intelligence.get("face_detected", False))
    subject_conf = float(intelligence.get("subject_conf", 0.0))
    face_center_x = float(intelligence.get("face_center_x", 0.5))
    face_center_y = float(intelligence.get("face_center_y", 0.42))
    face_area = float(intelligence.get("face_area", 0.0))
    archetype = str(intelligence.get("archetype", "neutral"))
    edit_priority = float(intelligence.get("edit_priority", 0.5))
    cut_priority = float(intelligence.get("cut_priority", 0.5))
    hype_score = float(intelligence.get("hype_score", 0.5))
    emotion_weight = float(intelligence.get("emotion_weight", 0.0))
    shot_type = str(intelligence.get("shot_type", "wide"))
    motion_class = str(intelligence.get("motion_class", "low"))
    motion_direction = str(intelligence.get("motion_direction", "still"))
    motion_intensity_scale = float(style.get("motion_intensity", 1.0))
    impact_rank = float(intelligence.get("impact_rank", 0.5))
    is_peak_beat = bool(intelligence.get("is_peak_beat", False))
    global_phase = str(effect_hints.get("global_phase", "flow"))
    global_allow_heavy = bool(effect_hints.get("global_allow_heavy", True))
    global_energy_rank = float(effect_hints.get("global_energy_rank", 0.5))
    signature_pack = effect_hints.get("signature_pack", None)

    is_quiet = scene_energy < 0.28 and hype_score < 0.45
    is_impact = scene_energy > 0.58 or scene_change > 0.23 or motion > 0.15 or cut_priority > 0.74
    if not is_peak_beat:
        is_peak_beat = bool(is_impact and (edit_priority > 0.55 or impact_rank > 0.62))

    wants_shake = bool(style.get("shake", False)) or bool(effect_hints.get("shake_applied", False))
    wants_zoom = bool(style.get("zoom", False)) or bool(effect_hints.get("zoom_applied", False))
    wants_glow = bool(style.get("glow", False)) or bool(effect_hints.get("glow_applied", False))
    wants_flash = bool(style.get("flash", False)) or bool(effect_hints.get("flash_applied", False))
    wants_glitch = bool(style.get("glitch", False)) or bool(effect_hints.get("glitch_applied", False))
    wants_blur = bool(style.get("motion_blur", False))
    wants_grain = bool(style.get("grain", False))
    wants_vignette = bool(style.get("vignette", False))
    wants_edge = bool(style.get("edge_glow", False))
    explicit_ramp_requested = bool(style.get("speed_ramp", False))
    wants_ramp = explicit_ramp_requested
    wants_slow_motion = bool(style.get("slow_motion", False))
    wants_fast_motion = bool(style.get("fast_motion", False))
    wants_reverse = bool(style.get("reverse", False))
    wants_boomerang = bool(style.get("boomerang", False))
    wants_pro_pack = bool(style.get("pro_edit_pack", False))
    auto_timefx = bool(style.get("auto_timefx", False)) or wants_pro_pack
    explicit_effects = set(style.get("explicit_effects", []) or [])
    avoid_effects = set(effect_hints.get("avoid_effects", []) or [])

    def allow_effect(effect_key: str) -> bool:
        return (effect_key not in avoid_effects) or (effect_key in explicit_effects)
    wants_anime_tone = bool(style.get("anime_tone", False)) or bool(effect_hints.get("anime_tone_applied", False))
    wants_ken_burns = bool(style.get("ken_burns", False)) or bool(effect_hints.get("ken_burns_applied", False))
    wants_chroma_pulse = bool(style.get("chroma_pulse", False)) or bool(effect_hints.get("chroma_pulse_applied", False))
    wants_color_grade = bool(style.get("color_grade", False))
    wants_black_white = bool(style.get("black_white", False))
    wants_no_color_grade = bool(style.get("no_color_grade", False))
    wants_fire_lines = bool(style.get("fire_lines", False))
    wants_particles = bool(style.get("particles", False))
    wants_energy_aura = bool(style.get("energy_aura", False))
    wants_turbulent = bool(style.get("turbulent_displace", False))
    wants_posterize_time = bool(style.get("posterize_time", False))
    wants_blend_overlay = bool(style.get("blend_overlay", False))
    wants_mask_roto = bool(style.get("mask_roto", False))
    wants_chroma_key = bool(style.get("chroma_key", False))
    wants_difference_matte = bool(style.get("difference_matte", False))
    wants_object_inpaint = bool(style.get("object_inpaint", False))
    simple_mode = bool(style.get("simple_mode", False))
    selective_beats = bool(style.get("selective_beats", True)) and (not bool(style.get("all_beats", False)))
    fast_render = bool(style.get("fast_render", False))
    quality_priority = bool(style.get("quality_priority", False))
    wants_character_focus = bool(style.get("character_focus", False)) or archetype == "character_focus"
    if (face_detected or subject_conf >= 0.56 or face_count > 0) and edit_priority > 0.4:
        wants_character_focus = True

    archetype_rules = ANIME_EDIT_ARCHETYPES.get(archetype, ANIME_EDIT_ARCHETYPES["neutral"])
    if not simple_mode:
        wants_shake = wants_shake or bool(archetype_rules.get("auto_shake", False))
        wants_flash = wants_flash or bool(archetype_rules.get("auto_flash", False))
    if archetype == "character_focus":
        wants_shake = False
        wants_glitch = False
        wants_flash = False
        wants_blur = False
        if not simple_mode:
            wants_zoom = True
        wants_character_focus = True
    elif archetype == "buildup":
        wants_flash = False
        wants_shake = False
        wants_ken_burns = True
    elif archetype == "dark_tension":
        wants_glitch = bool(style.get("glitch", False)) and (scene_change > 0.12)
        wants_vignette = True
    elif archetype in {"action", "hard_cut"}:
        wants_chroma_pulse = wants_chroma_pulse or mood in {"hype", "dramatic"}

    if simple_mode:
        # Obey "simple" intent strictly: no auto punch/shake unless explicitly requested.
        wants_shake = bool(style.get("shake", False))
        wants_zoom = bool(style.get("zoom", False))
        wants_flash = bool(style.get("flash", False))
        wants_glitch = bool(style.get("glitch", False))
        wants_blur = bool(style.get("motion_blur", False))
        wants_chroma_pulse = bool(style.get("chroma_pulse", False))
        wants_ramp = bool(style.get("speed_ramp", False))
        wants_ken_burns = bool(style.get("ken_burns", False))
        wants_anime_tone = bool(style.get("anime_tone", False))
        wants_slow_motion = bool(style.get("slow_motion", False))
        wants_fast_motion = bool(style.get("fast_motion", False))
        wants_reverse = bool(style.get("reverse", False))
        wants_boomerang = bool(style.get("boomerang", False))

    local_intensity = intensity
    if is_quiet:
        local_intensity = max(1, intensity - 1)
    elif is_impact and intensity < 3:
        local_intensity = intensity + 1
    if archetype == "action":
        local_intensity = min(3, local_intensity + 1)
    if archetype == "character_focus":
        local_intensity = max(1, local_intensity - 1)
    if hype_score >= 0.78 and mood in {"hype", "dark", "dramatic"}:
        local_intensity = min(3, local_intensity + 1)
    if emotion_weight >= 0.72 and shot_type in {"close_up", "extreme_close_up"}:
        local_intensity = max(1, local_intensity - 1)

    if wants_no_color_grade:
        wants_anime_tone = False
    if wants_black_white:
        wants_no_color_grade = True
        wants_anime_tone = False
        wants_color_grade = False
    if wants_no_color_grade:
        wants_color_grade = False
    preserve_original_color = wants_no_color_grade and (not wants_color_grade) and (not wants_black_white)
    if preserve_original_color:
        wants_anime_tone = False
        wants_glow = False
        wants_chroma_pulse = False
        if not bool(style.get("blend_overlay", False)):
            wants_blend_overlay = False

    tool_intelligence = bool(style.get("tool_intelligence", True)) or wants_pro_pack
    if tool_intelligence and (not simple_mode):
        rng = random.Random((index + 1) * 17 + int(scene_energy * 1000))
        impact_score = (scene_change * 0.42) + (motion * 0.28) + (impact_rank * 0.3)
        cinematic_score = (0.22 * (1.0 - abs(brightness - 0.52))) + (contrast * 0.22) + (0.28 if face_count > 0 else 0.12)
        aura_score = (0.25 if wants_character_focus else 0.05) + (saturation * 0.2) + (0.2 if subject_conf >= 0.45 else 0.05)
        glitch_score = (0.28 if mood == "dark" else 0.08) + (scene_change * 0.2) + (contrast * 0.12) + (0.18 if brightness < 0.38 else 0.02)
        flow_score = (0.28 if archetype in {"buildup", "character_focus"} else 0.08) + (0.22 if motion < 0.1 else 0.05)

        packs = [
            ("impact", impact_score),
            ("cinematic", cinematic_score),
            ("aura", aura_score),
            ("glitch", glitch_score),
            ("flow", flow_score),
        ]
        if isinstance(signature_pack, (list, tuple, set)) and signature_pack:
            sig_set = {str(x) for x in signature_pack}
            packs = [p for p in packs if p[0] in sig_set] or packs
        packs.sort(key=lambda item: item[1], reverse=True)
        chosen = [packs[0][0]]
        if len(packs) > 1 and (packs[1][1] + rng.random() * 0.05) > 0.2:
            chosen.append(packs[1][0])

        if "impact" in chosen:
            wants_shake = wants_shake or (scene_change > 0.12 or motion > 0.1)
            wants_zoom = wants_zoom or (scene_energy > 0.55 and impact_rank > 0.55)
            wants_flash = wants_flash or (scene_change > 0.18 or impact_rank > 0.7)
            if mood in {"hype", "dramatic"}:
                wants_ramp = wants_ramp or (impact_rank > 0.7)
                wants_chroma_pulse = wants_chroma_pulse or (scene_change > 0.16)
            if archetype in {"action", "hard_cut"}:
                wants_posterize_time = wants_posterize_time or (impact_rank > 0.75)

        if "cinematic" in chosen:
            wants_ken_burns = wants_ken_burns or (archetype in {"buildup", "character_focus"} or scene_energy < 0.55)
            wants_glow = wants_glow or (contrast > 0.25 or mood in {"dramatic", "chill"})
            wants_vignette = wants_vignette or (mood in {"dark", "dramatic"})
            if not preserve_original_color:
                wants_blend_overlay = wants_blend_overlay or (contrast > 0.2)
                wants_color_grade = wants_color_grade or (mood in {"dramatic", "dark"} and contrast > 0.22)

        if "aura" in chosen:
            wants_fire_lines = wants_fire_lines or (wants_character_focus and (impact_rank > 0.55 or scene_change > 0.12))
            wants_energy_aura = wants_energy_aura or (wants_character_focus and mood in {"dramatic", "hype"})
            wants_particles = wants_particles or (scene_energy < 0.55 or mood in {"sad", "dark"})
            if not preserve_original_color:
                wants_mask_roto = wants_mask_roto or (subject_conf >= 0.5 and scene_change > 0.12)

        if "glitch" in chosen:
            wants_glitch = wants_glitch or (scene_change > 0.2 or contrast > 0.3)
            wants_turbulent = wants_turbulent or (scene_change > 0.16 or motion > 0.12)

        if "flow" in chosen:
            wants_slow_motion = wants_slow_motion or (motion < 0.09 and archetype in {"buildup", "character_focus"})
            wants_ken_burns = wants_ken_burns or (scene_energy < 0.5)
            if not preserve_original_color:
                wants_blend_overlay = wants_blend_overlay or (mood in {"chill", "sad"})

    effect_focus_gate = (not selective_beats) or is_peak_beat or impact_rank >= 0.72
    auto_ramp_gate = (
        (not explicit_ramp_requested)
        and (not simple_mode)
        and (not is_short_segment)
        and (not wants_character_focus)
        and (mood in {"hype", "dramatic"})
        and effect_focus_gate
        and (
            (impact_rank >= 0.84 and scene_change >= 0.16)
            or (is_peak_beat and scene_energy >= 0.68 and motion >= 0.11)
            or (archetype in {"action", "hard_cut"} and scene_change >= 0.14 and motion >= 0.1)
        )
    )
    if auto_ramp_gate:
        wants_ramp = True

    if offload_to_ffmpeg:
        # Offload full-frame expensive looks to ffmpeg post-pass.
        wants_grain = False
        wants_vignette = False
        wants_edge = False
        wants_color_grade = False

    applied = {
        "segment_index": index,
        "scene_energy": round(scene_energy, 4),
        "motion": round(motion, 4),
        "flow_mag": round(flow_mag, 4),
        "contrast": round(contrast, 4),
        "brightness": round(brightness, 4),
        "saturation": round(float(intelligence.get("saturation", 0.5)), 4),
        "scene_change": round(scene_change, 4),
        "face_count": face_count,
        "face_detected": bool(face_detected),
        "subject_conf": round(subject_conf, 4),
        "face_center_x": round(face_center_x, 4),
        "face_center_y": round(face_center_y, 4),
        "face_area": round(face_area, 4),
        "archetype": archetype,
        "edit_priority": round(edit_priority, 4),
        "cut_priority": round(cut_priority, 4),
        "hype_score": round(hype_score, 4),
        "emotion_weight": round(emotion_weight, 4),
        "shot_type": shot_type,
        "motion_class": motion_class,
        "motion_direction": motion_direction,
        "zoom_applied": False,
        "shake_applied": False,
        "glitch_applied": False,
        "glow_applied": False,
        "flash_applied": False,
        "grade_applied": False,
        "ken_burns_applied": False,
        "anime_tone_applied": False,
        "chroma_pulse_applied": False,
        "speed_ramp_applied": False,
        "slow_motion_applied": False,
        "fast_motion_applied": False,
        "reverse_applied": False,
        "boomerang_applied": False,
        "black_white_applied": False,
        "fire_lines_applied": False,
        "focus_zoom_applied": False,
        "particles_applied": False,
        "energy_aura_applied": False,
        "turbulent_applied": False,
        "posterize_time_applied": False,
        "blend_overlay_applied": False,
        "mask_roto_applied": False,
        "chroma_key_applied": False,
        "difference_matte_applied": False,
        "object_inpaint_applied": False,
        "clip_repeat_applied": False,
        "clip_repeat_count": 1,
    }
    if offload_to_ffmpeg:
        if bool(style.get("color_grade", False)):
            applied["grade_applied"] = True
    effect_budget = 4 if local_intensity <= 1 else (5 if local_intensity == 2 else 6)
    if archetype in {"action", "hard_cut"}:
        effect_budget += 1
    if edit_priority > 0.72:
        effect_budget += 1
    if wants_pro_pack:
        effect_budget += 1
    if selective_beats:
        effect_budget -= 1
    if fast_render:
        effect_budget -= 1
    if quality_priority and (not fast_render):
        effect_budget += 1
    if global_phase in {"intro", "buildup", "outro"}:
        effect_budget -= 1
    if global_phase == "impact":
        effect_budget += 1
    clean_gate = (
        global_phase in {"intro", "buildup", "outro", "flow"}
        and (not is_peak_beat)
        and edit_priority < 0.55
        and (not bool(style.get("pro_edit_pack", False)))
        and (not bool(style.get("all_beats", False)))
    )
    effect_budget = int(np.clip(effect_budget, 3, 7))
    effects_used = 0
    heavy_effects_used = 0
    heavy_budget = 1 if is_short_segment else 2
    if local_intensity >= 3 and (not is_short_segment):
        heavy_budget = 3
    if wants_fire_lines and (not is_short_segment):
        heavy_budget = max(2, heavy_budget)
    if hype_score > 0.78 and mood in {"hype", "dark", "dramatic"}:
        effect_budget += 1
        heavy_budget = min(4, heavy_budget + 1)
    if emotion_weight > 0.72 and shot_type in {"close_up", "extreme_close_up"}:
        heavy_budget = max(0, heavy_budget - 1)
        wants_shake = False
        wants_flash = False
    if selective_beats:
        heavy_budget = max(1, heavy_budget - 1)
    if fast_render:
        heavy_budget = max(1, heavy_budget - 1)
    if quality_priority and (not fast_render):
        heavy_budget = min(4, heavy_budget + 1)
    if global_phase in {"intro", "buildup", "outro"}:
        heavy_budget = max(0, heavy_budget - 1)
    if global_phase == "impact":
        heavy_budget = min(4, heavy_budget + 1)
    if not global_allow_heavy:
        heavy_budget = max(0, heavy_budget - 1)
    if clean_gate and not explicit_effects:
        effect_budget = 0
        heavy_budget = 0
        wants_shake = False
        wants_zoom = False
        wants_flash = False
        wants_glitch = False
        wants_chroma_pulse = False
        wants_particles = False
        wants_energy_aura = False
        wants_fire_lines = False
        wants_turbulent = False
        wants_posterize_time = False
        wants_blend_overlay = False

    if motion_class == "explosive" and effect_focus_gate and mood in {"hype", "dark", "dramatic"} and not simple_mode:
        wants_shake = wants_shake or hype_score > 0.62
        wants_ramp = wants_ramp or cut_priority > 0.68
    if shot_type in {"close_up", "extreme_close_up"} and emotion_weight > 0.58 and not simple_mode:
        wants_character_focus = True
        wants_zoom = True
        wants_ken_burns = wants_ken_burns or mood in {"dramatic", "sad", "chill"}

    def can_apply() -> bool:
        return effects_used < effect_budget

    def can_apply_heavy() -> bool:
        return can_apply() and (heavy_effects_used < heavy_budget)

    def mark_applied() -> None:
        nonlocal effects_used
        effects_used += 1

    def mark_heavy_applied() -> None:
        nonlocal heavy_effects_used
        mark_applied()
        heavy_effects_used += 1

    fire_priority_gate = wants_fire_lines and (effect_focus_gate or wants_character_focus) and (
        is_impact
        or scene_change > 0.12
        or motion > 0.09
        or face_detected
        or subject_conf >= 0.42
        or face_count > 0
        or archetype in {"character_focus", "action", "hard_cut"}
        or (index % 3 == 0)
    )
    if fire_priority_gate and can_apply_heavy():
        segment = _apply_fire_lines_effect(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            face_area=face_area,
            intensity=local_intensity,
        )
        applied["fire_lines_applied"] = True
        mark_heavy_applied()

    focus_gate = (
        wants_character_focus
        and wants_zoom
        and (effect_focus_gate or archetype == "character_focus")
        and (face_detected or subject_conf >= 0.56 or archetype == "character_focus")
        and (edit_priority > 0.4 or archetype == "character_focus")
    )
    if focus_gate and can_apply_heavy():
        zoom_strength = float(np.clip((0.06 + (0.026 * local_intensity) + (0.24 * face_area)) * max(0.75, motion_intensity_scale), 0.06, 0.24))
        pulse_hz = float(np.clip(1.9 + (0.42 * local_intensity), 1.6, 3.5))
        segment = _apply_focus_zoom_animation(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            max_zoom=zoom_strength,
            pulse_hz=pulse_hz,
        )
        applied["zoom_applied"] = True
        applied["focus_zoom_applied"] = True
        mark_heavy_applied()

    if wants_chroma_key and can_apply_heavy():
        segment = _apply_green_screen_chroma_key_effect(segment, intensity=local_intensity)
        applied["chroma_key_applied"] = True
        mark_heavy_applied()

    if wants_object_inpaint and (not applied["chroma_key_applied"]) and can_apply_heavy():
        segment = _apply_object_inpaint_background_effect(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            intensity=local_intensity,
        )
        applied["object_inpaint_applied"] = True
        mark_heavy_applied()

    if wants_mask_roto and (not applied["chroma_key_applied"]) and can_apply_heavy():
        segment = _apply_mask_roto_effect(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            intensity=local_intensity,
        )
        applied["mask_roto_applied"] = True
        mark_heavy_applied()

    if wants_difference_matte and (effect_focus_gate or wants_mask_roto) and can_apply():
        segment = _apply_difference_matte_effect(segment, intensity=local_intensity)
        applied["difference_matte_applied"] = True
        mark_applied()

    # Speed and reverse effects (explicitly requested).
    if wants_reverse and (not wants_boomerang) and can_apply():
        segment = segment.fx(vfx.time_mirror)
        applied["reverse_applied"] = True
        mark_applied()
    if wants_boomerang and can_apply():
        segment = segment.fx(vfx.time_symmetrize)
        applied["boomerang_applied"] = True
        mark_applied()

    if wants_slow_motion and (not wants_fast_motion) and can_apply():
        segment = segment.fx(vfx.speedx, 0.85 if not is_short_segment else 0.92)
        applied["slow_motion_applied"] = True
        mark_applied()
    if wants_fast_motion and can_apply():
        segment = segment.fx(vfx.speedx, 1.12 if not is_short_segment else 1.08)
        applied["fast_motion_applied"] = True
        mark_applied()

    if auto_timefx and (not wants_slow_motion) and (not wants_fast_motion) and (not wants_reverse) and (not wants_boomerang):
        time_gate = effect_focus_gate and (not is_short_segment) and (not wants_character_focus)
        if time_gate and archetype in {"action", "hard_cut"} and (is_impact or scene_energy > 0.6) and can_apply():
            segment = segment.fx(vfx.speedx, 1.08)
            applied["fast_motion_applied"] = True
            mark_applied()
        elif time_gate and archetype in {"buildup", "character_focus"} and (scene_energy < 0.5 or motion < 0.08) and can_apply():
            segment = segment.fx(vfx.speedx, 0.92)
            applied["slow_motion_applied"] = True
            mark_applied()

    if wants_posterize_time and effect_focus_gate and (is_impact or archetype in {"hard_cut", "action"}) and can_apply():
        segment = _apply_posterize_time_effect(segment, target_fps=(12 if local_intensity >= 2 else 15))
        applied["posterize_time_applied"] = True
        mark_applied()

    if wants_turbulent and effect_focus_gate and (is_impact or scene_change > 0.14 or archetype in {"action", "hard_cut"}) and can_apply_heavy():
        segment = _apply_turbulent_displace_effect(segment, intensity=local_intensity)
        applied["turbulent_applied"] = True
        mark_heavy_applied()

    if mood == "hype":
        if is_impact and (not preserve_original_color):
            segment = segment.fx(vfx.lum_contrast, 0, 8 + (3 * local_intensity), 128)
        ramp_gate = wants_ramp and effect_focus_gate and (not is_short_segment) and (not wants_character_focus) and (
            (impact_rank >= 0.78 and (scene_change > 0.15 or motion > 0.1))
            or (is_peak_beat and scene_energy >= 0.62)
        )
        if ramp_gate and ((impact_rank >= 0.9) or (index % 2 == 0)) and can_apply() and allow_effect("speed_ramp"):
            ramp_mult = float(np.clip(1.05 + (0.018 * local_intensity * motion_intensity_scale) + (0.05 * max(0.0, scene_change - 0.18)), 1.04, 1.2))
            segment = segment.fx(vfx.speedx, ramp_mult)
            applied["speed_ramp_applied"] = True
            mark_applied()
        if wants_zoom and effect_focus_gate and is_impact and (not applied["zoom_applied"]) and can_apply_heavy() and allow_effect("zoom"):
            segment = _apply_zoom_punch_effect(segment, strength=float(np.clip((0.1 + (0.02 * local_intensity)) * motion_intensity_scale, 0.06, 0.2)))
            applied["zoom_applied"] = True
            mark_heavy_applied()
        if wants_shake and effect_focus_gate and (motion > 0.11 or scene_change > 0.2) and can_apply() and allow_effect("shake"):
            segment = _apply_shake_effect(segment, amount_px=int(np.clip((5 + (2 * local_intensity)) * motion_intensity_scale, 2, 14)), frequency=9.0 + (local_intensity * motion_intensity_scale))
            applied["shake_applied"] = True
            mark_applied()
    elif mood == "dark":
        if not preserve_original_color:
            segment = segment.fx(vfx.lum_contrast, -6, 18 + (4 * local_intensity), 128)
        if wants_glitch and effect_focus_gate and (scene_change > 0.18 or is_impact) and can_apply() and allow_effect("glitch"):
            segment = _apply_rgb_split_effect(segment, pixels=3 + local_intensity)
            applied["glitch_applied"] = True
            mark_applied()
        if wants_shake and effect_focus_gate and motion > 0.14 and can_apply() and allow_effect("shake"):
            segment = _apply_shake_effect(segment, amount_px=int(np.clip((3 + local_intensity) * motion_intensity_scale, 2, 10)), frequency=7.0 + (local_intensity * motion_intensity_scale))
            applied["shake_applied"] = True
            mark_applied()
    elif mood == "dramatic":
        if not preserve_original_color:
            segment = segment.fx(vfx.lum_contrast, 2, 12 + (4 * local_intensity), 128)
        if wants_zoom and effect_focus_gate and (scene_change > 0.2 or is_impact) and (not applied["zoom_applied"]) and can_apply_heavy() and allow_effect("zoom"):
            segment = _apply_zoom_punch_effect(segment, strength=float(np.clip((0.08 + (0.015 * local_intensity)) * motion_intensity_scale, 0.05, 0.17)))
            applied["zoom_applied"] = True
            mark_heavy_applied()
        if wants_glow and (contrast > 0.27 or is_impact) and can_apply() and allow_effect("glow"):
            segment = _apply_glow_effect(segment, strength=0.1 + (0.06 * local_intensity))
            applied["glow_applied"] = True
            mark_applied()
        ramp_gate = wants_ramp and effect_focus_gate and (not is_short_segment) and (
            (impact_rank >= 0.8 and scene_change > 0.16)
            or (is_peak_beat and scene_energy > 0.62)
        )
        if ramp_gate and can_apply() and allow_effect("speed_ramp"):
            ramp_mult = 0.95 if (scene_change > 0.24 and motion < 0.12) else 1.06
            segment = segment.fx(vfx.speedx, ramp_mult)
            applied["speed_ramp_applied"] = True
            mark_applied()
    elif mood == "sad":
        if not preserve_original_color:
            segment = segment.fx(vfx.lum_contrast, 0, -4, 128)
        ramp_gate = explicit_ramp_requested and wants_ramp and (not is_quiet) and (not is_short_segment) and (
            (scene_change > 0.12 and motion > 0.06) or is_peak_beat
        )
        if ramp_gate and (index % 3 == 0) and can_apply():
            segment = segment.fx(vfx.speedx, 0.92)
            applied["speed_ramp_applied"] = True
            mark_applied()
    else:
        if not preserve_original_color:
            segment = segment.fx(vfx.lum_contrast, 0, 2, 128)

    if wants_glitch and effect_focus_gate and not applied["glitch_applied"] and (is_impact or scene_change > 0.16) and can_apply() and allow_effect("glitch"):
        segment = _apply_rgb_split_effect(segment, pixels=2 + local_intensity)
        applied["glitch_applied"] = True
        mark_applied()

    bw_gate = wants_black_white and effect_focus_gate and (is_impact or scene_change > 0.16 or index % 2 == 0)
    if bw_gate and can_apply() and allow_effect("black_white"):
        segment = _apply_black_white_pulse_effect(
            segment,
            intensity=local_intensity,
            pulse_hz=2.1 + (0.4 * local_intensity),
        )
        applied["black_white_applied"] = True
        mark_applied()

    fire_gate = (not applied["fire_lines_applied"]) and wants_fire_lines and (effect_focus_gate or wants_character_focus) and (
        ((face_detected or subject_conf >= 0.5 or face_count > 0) and (is_impact or edit_priority > 0.55))
        or (wants_character_focus and (is_impact or scene_change > 0.16))
        or (archetype in {"character_focus", "action", "hard_cut"} and (is_impact or scene_change > 0.14))
        or (scene_change > 0.22)
        or (index % 2 == 0)
    )
    if fire_gate and can_apply_heavy() and allow_effect("fire_lines"):
        segment = _apply_fire_lines_effect(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            face_area=face_area,
            intensity=local_intensity,
        )
        applied["fire_lines_applied"] = True
        mark_heavy_applied()

    if wants_energy_aura and (applied["fire_lines_applied"] or face_detected or subject_conf >= 0.48 or archetype == "character_focus") and can_apply() and allow_effect("energy_aura"):
        segment = _apply_energy_aura_effect(
            segment,
            center_x=face_center_x,
            center_y=face_center_y,
            intensity=local_intensity,
        )
        applied["energy_aura_applied"] = True
        mark_applied()

    # Keep boundary extraction/effect first, then apply grade if requested.
    do_grade = (
        wants_color_grade
        and ((not is_quiet) or mood in {"dark", "dramatic"} or bool(effect_hints.get("grade_applied", False)))
        and (not wants_no_color_grade)
        and (not wants_black_white)
    )
    if do_grade:
        segment = _apply_color_grade(
            segment,
            mood,
            local_intensity,
            scene_energy=scene_energy,
            brightness=brightness,
            contrast=contrast,
        )
        applied["grade_applied"] = True

    if wants_blur and effect_focus_gate and is_impact and can_apply():
        segment = _apply_motion_blur_effect(segment, blur_px=5 + (2 * local_intensity))
        mark_applied()
    if wants_glow and not applied["glow_applied"] and (brightness < 0.6 and contrast > 0.22) and can_apply():
        segment = _apply_glow_effect(segment, strength=0.08 + (0.05 * local_intensity))
        applied["glow_applied"] = True
        mark_applied()
    if wants_grain and not is_quiet and can_apply():
        segment = _apply_film_grain_effect(segment, amount=3.6 + (1.8 * local_intensity))
        mark_applied()
    if wants_vignette and (brightness > 0.6 or mood in {"dark", "sad"}) and can_apply():
        segment = _apply_vignette_effect(segment, strength=0.2 + (0.06 * local_intensity))
        mark_applied()
    if wants_edge and effect_focus_gate and is_impact and can_apply_heavy():
        segment = _apply_edge_glow_effect(segment, strength=0.16 + (0.06 * local_intensity))
        mark_heavy_applied()
    if wants_flash and effect_focus_gate and (not preserve_original_color) and (scene_change > 0.22 or (is_impact and index % 2 == 0)) and can_apply():
        segment = segment.fx(vfx.lum_contrast, 6, 8 + (2 * local_intensity), 128)
        applied["flash_applied"] = True
        mark_applied()

    if wants_ken_burns and (is_quiet or mood in {"dramatic", "sad", "chill"}) and index % 3 == 0 and can_apply_heavy():
        segment = _apply_ken_burns_animation(
            segment,
            zoom_strength=0.04 + (0.012 * local_intensity),
            pan_strength=0.1 + (0.03 * local_intensity),
        )
        applied["ken_burns_applied"] = True
        mark_heavy_applied()

    if wants_anime_tone and (contrast > 0.2 or (mood in {"dramatic", "chill"} and index % 3 == 1)) and can_apply_heavy():
        segment = _apply_anime_tone_effect(segment, strength=0.2 + (0.08 * local_intensity))
        applied["anime_tone_applied"] = True
        mark_heavy_applied()

    if wants_chroma_pulse and effect_focus_gate and is_impact and index % 2 == 0 and can_apply() and allow_effect("chroma_pulse"):
        segment = _apply_chroma_pulse_effect(segment, max_shift=2 + local_intensity, hz=3.5 + local_intensity)
        applied["chroma_pulse_applied"] = True
        mark_applied()

    if wants_particles and (effect_focus_gate or mood in {"dark", "sad"}) and (is_impact or scene_change > 0.1 or local_intensity >= 2) and can_apply() and allow_effect("particles"):
        segment = _apply_particle_atmosphere_effect(segment, mood=mood, intensity=local_intensity)
        applied["particles_applied"] = True
        mark_applied()

    if wants_blend_overlay and (not preserve_original_color) and can_apply() and allow_effect("blend_overlay"):
        segment = _apply_light_leak_overlay_effect(segment, mood=mood, intensity=local_intensity)
        applied["blend_overlay_applied"] = True
        mark_applied()

    return segment, applied


def _pick_repeat_segment_span(
    intelligence_rows: list[dict[str, Any]],
    *,
    min_len: int = 2,
    max_len: int = 4,
) -> tuple[int, int, float] | None:
    if not intelligence_rows:
        return None
    n = len(intelligence_rows)
    if n < min_len:
        return None

    scores: list[float] = []
    for row in intelligence_rows:
        impact = float(np.clip(float(row.get("impact_rank", 0.0)), 0.0, 1.0))
        energy = float(np.clip(float(row.get("scene_energy", 0.0)), 0.0, 1.0))
        change = float(np.clip(float(row.get("scene_change", 0.0)), 0.0, 1.0))
        motion = float(np.clip(float(row.get("motion", 0.0)), 0.0, 1.0))
        peak = 1.0 if bool(row.get("is_peak_beat", False)) else 0.0
        archetype = str(row.get("archetype", "neutral"))
        arche_bonus = 0.06 if archetype in {"action", "hard_cut"} else (0.03 if archetype == "character_focus" else 0.0)
        score = (impact * 0.44) + (energy * 0.27) + (change * 0.16) + (motion * 0.07) + (peak * 0.06) + arche_bonus
        scores.append(float(np.clip(score, 0.0, 1.0)))

    best: tuple[int, int, float] | None = None
    max_len = max(min_len, min(max_len, 6))
    for span in range(min_len, max_len + 1):
        for start in range(0, n - span + 1):
            end = start + span
            window = scores[start:end]
            avg = float(np.mean(window))
            # Require at least one strong beat in looped run.
            has_peak = any(bool(intelligence_rows[i].get("is_peak_beat", False)) for i in range(start, end))
            if not has_peak:
                continue
            if (best is None) or (avg > best[2]):
                best = (start, end, avg)

    if best is None:
        return None
    # Avoid repeating low-impact regions.
    if best[2] < 0.58:
        return None
    return best


def _pick_tail_repeat_span(
    renderable_indices: list[int],
    intelligence_rows: list[dict[str, Any]],
    *,
    min_len: int = 2,
    max_len: int = 3,
) -> list[int]:
    if not renderable_indices or not intelligence_rows:
        return []

    tail_window_len = max(2, int(round(len(renderable_indices) * 0.28)))
    tail_window = renderable_indices[-tail_window_len:]
    if len(tail_window) < min_len:
        return []

    best: tuple[int, int, float] | None = None
    max_len = max(min_len, min(max_len, len(tail_window)))
    for span in range(min_len, max_len + 1):
        for start in range(0, len(tail_window) - span + 1):
            span_indices = tail_window[start : start + span]
            scores: list[float] = []
            has_peak = False
            for idx in span_indices:
                if idx >= len(intelligence_rows):
                    continue
                row = intelligence_rows[idx]
                impact = float(np.clip(float(row.get("impact_rank", 0.0)), 0.0, 1.0))
                energy = float(np.clip(float(row.get("scene_energy", 0.0)), 0.0, 1.0))
                change = float(np.clip(float(row.get("scene_change", 0.0)), 0.0, 1.0))
                motion = float(np.clip(float(row.get("motion", 0.0)), 0.0, 1.0))
                peak = 1.0 if bool(row.get("is_peak_beat", False)) else 0.0
                archetype = str(row.get("archetype", "neutral"))
                arche_bonus = 0.06 if archetype in {"action", "hard_cut"} else (0.03 if archetype == "character_focus" else 0.0)
                score = (impact * 0.44) + (energy * 0.27) + (change * 0.16) + (motion * 0.07) + (peak * 0.06) + arche_bonus
                scores.append(float(np.clip(score, 0.0, 1.0)))
                if peak > 0:
                    has_peak = True
            if not scores:
                continue
            avg = float(np.mean(scores))
            if not has_peak and avg < 0.6:
                continue
            if (best is None) or (avg > best[2]):
                best = (start, start + span, avg)

    if best is None or best[2] < 0.56:
        return []
    return tail_window[best[0] : best[1]]


# =============================================================================
# STRONG EFFECTS FIX — Visible Parameters
# =============================================================================

def _strong_glow(frame: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Visible bloom glow. Old was 0.08 — invisible. New is 0.5-1.5."""
    f32 = frame.astype(np.float32) / 255.0
    h, w = f32.shape[:2]
    luma = 0.299*f32[:,:,2] + 0.587*f32[:,:,1] + 0.114*f32[:,:,0]
    bright_mask = np.clip((luma - 0.45) / 0.55, 0, 1)
    bright = f32 * bright_mask[..., None]
    b1 = cv2.GaussianBlur(bright, (0,0), 4.0)
    b2 = cv2.GaussianBlur(bright, (0,0), 12.0)
    b3 = cv2.GaussianBlur(bright, (0,0), 28.0)
    bloom = b1*0.45 + b2*0.35 + b3*0.20
    bloom *= strength * 1.8
    result = 1.0 - (1.0 - f32) * (1.0 - np.clip(bloom, 0, 1))
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _strong_shake(frame: np.ndarray, t: float, amount: int = 14, freq: float = 12.0) -> np.ndarray:
    """Visible shake. Old was 2-5px — invisible. New is 10-18px."""
    dx = int(amount * math.sin(2 * math.pi * freq * t))
    dy = int(amount * 0.7 * math.cos(2 * math.pi * (freq * 0.8) * t + 0.4))
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)


def _strong_zoom(frame: np.ndarray, factor: float = 1.18) -> np.ndarray:
    """Visible zoom. Old was 1.04-1.08. New is 1.12-1.22."""
    h, w = frame.shape[:2]
    new_w = int(w * factor)
    new_h = int(h * factor)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x1 = (new_w - w) // 2
    y1 = (new_h - h) // 2
    return resized[y1:y1+h, x1:x1+w]


def _strong_rgb_split(frame: np.ndarray, pixels: int = 10) -> np.ndarray:
    """Visible chromatic aberration. Old was 2-3px. New is 8-14px."""
    out = frame.copy()
    if pixels < 1:
        return out
    out[:, :, 2] = np.roll(frame[:, :, 2],  pixels, axis=1)
    out[:, :, 0] = np.roll(frame[:, :, 0], -pixels, axis=1)
    return out


def _strong_vignette(frame: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Visible vignette. Old was 0.2. New is 0.55-0.85."""
    h, w = frame.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    mask = 1.0 - np.clip((dist - 0.3) / 0.7, 0, 1) * strength
    return np.clip(frame.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def _strong_grain(frame: np.ndarray, amount: float = 18.0, t: float = 0.0) -> np.ndarray:
    """Visible grain. Old was 3-5. New is 14-22."""
    h, w = frame.shape[:2]
    rng = np.random.default_rng(int(t * 10000) & 0x7FFFFFFF)
    noise = rng.normal(0, amount, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (3, 3), 0.8)
    pulse = 0.85 + 0.15 * math.sin(2 * math.pi * 3.5 * t)
    out = frame.astype(np.float32)
    out[:,:,0] = np.clip(out[:,:,0] + noise * pulse, 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] + noise * pulse, 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] + noise * pulse, 0, 255)
    return out.astype(np.uint8)


def _strong_color_grade(frame: np.ndarray, mood: str, intensity: int = 3) -> np.ndarray:
    """Visible color grading. Old contrast=+4-8. New is +18-30."""
    out = frame.astype(np.float32)
    contrast_gain = 1.0 + (0.12 * intensity)
    out = np.clip((out - 128) * contrast_gain + 128, 0, 255)
    if mood == "hype":
        out[:,:,2] = np.clip(out[:,:,2] * 1.08 + 8,  0, 255)
        out[:,:,1] = np.clip(out[:,:,1] * 1.02 + 4,  0, 255)
        out[:,:,0] = np.clip(out[:,:,0] * 0.90 - 5,  0, 255)
    elif mood == "dark":
        out[:,:,2] = np.clip(out[:,:,2] * 0.88 - 10, 0, 255)
        out[:,:,0] = np.clip(out[:,:,0] * 1.08 + 12, 0, 255)
        out = np.clip(out * 0.88, 0, 255)
    elif mood == "sad":
        out[:,:,2] = np.clip(out[:,:,2] * 0.85, 0, 255)
        out[:,:,0] = np.clip(out[:,:,0] * 1.12 + 15, 0, 255)
    elif mood == "dramatic":
        out[:,:,2] = np.clip(out[:,:,2] * 1.10 + 10, 0, 255)
        out[:,:,0] = np.clip(out[:,:,0] * 1.06 + 8,  0, 255)
        out[:,:,1] = np.clip(out[:,:,1] * 0.94,      0, 255)
    elif mood == "chill":
        out[:,:,2] = np.clip(out[:,:,2] * 1.05 + 6,  0, 255)
        out[:,:,1] = np.clip(out[:,:,1] * 1.03 + 4,  0, 255)
    f32 = out / 255.0
    luma = 0.299*f32[:,:,2] + 0.587*f32[:,:,1] + 0.114*f32[:,:,0]
    hi   = np.clip((luma - 0.78) / 0.22, 0, 1)[..., None]
    f32  = np.clip(f32 - hi * 0.12 * f32, 0, 1)
    return (f32 * 255).astype(np.uint8)


def _strong_flash(frame: np.ndarray, intensity: float = 0.85) -> np.ndarray:
    """White flash transition."""
    white = np.ones_like(frame, dtype=np.float32) * 255
    f32   = frame.astype(np.float32)
    return np.clip(f32 * (1 - intensity) + white * intensity, 0, 255).astype(np.uint8)


def _strong_energy_aura(frame: np.ndarray, t: float, intensity: float = 1.0) -> np.ndarray:
    """Visible energy aura effect."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (0,0), 1.5)
    edges = cv2.Canny(blur, 40, 120)
    cv2.dilate(edges, None, edges, iterations=2)
    pulse = 0.65 + 0.35 * math.sin(2 * math.pi * 2.8 * t + 0.9)
    spark = 0.5  + 0.5  * math.sin(2 * math.pi * 4.6 * t + 1.15)
    glow = np.zeros_like(frame, dtype=np.float32)
    for sigma, weight in [(3, 0.35), (7, 0.30), (14, 0.20), (25, 0.15)]:
        bl = cv2.GaussianBlur(edges.astype(np.float32), (0,0), sigma)
        glow[:,:,0] += bl * weight * pulse * 0.85
        glow[:,:,1] += bl * weight * pulse * 0.55
        glow[:,:,2] += bl * weight * pulse * 0.20
    core = cv2.GaussianBlur(edges.astype(np.float32), (3,3), 0.8) * spark * 0.6
    glow[:,:,0] += core * 0.7
    glow[:,:,1] += core * 0.8
    glow[:,:,2] += core * 0.9
    glow *= intensity * 1.1
    f32    = frame.astype(np.float32) / 255.0
    g32    = np.clip(glow / 255.0, 0, 1)
    result = 1.0 - (1.0 - f32) * (1.0 - g32)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _strong_fire_lines(frame: np.ndarray, t: float, intensity: float = 1.0) -> np.ndarray:
    """Visible fire/electric outline effect."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 1.2)
    edges = cv2.Canny(blur, 50, 130)
    cv2.dilate(edges, None, edges, iterations=1)
    pulse = 0.72 + 0.28 * math.sin(2 * math.pi * 3.4 * t + 0.6)
    spark = 0.50 + 0.50 * math.sin(2 * math.pi * 4.6 * t + 1.15)
    fire  = np.zeros_like(frame, dtype=np.float32)
    o1 = cv2.GaussianBlur(edges.astype(np.float32), (0,0), 6.0)
    fire[:,:,2] += o1 * pulse * 1.1
    fire[:,:,1] += o1 * pulse * 0.55
    o2 = cv2.GaussianBlur(edges.astype(np.float32), (0,0), 3.0)
    fire[:,:,2] += o2 * pulse * 0.7
    fire[:,:,1] += o2 * pulse * 0.65
    o3 = cv2.GaussianBlur(edges.astype(np.float32), (3,3), 0.8)
    fire[:,:,2] += o3 * spark * 0.85
    fire[:,:,1] += o3 * spark * 0.88
    fire[:,:,0] += o3 * spark * 0.75
    fire *= intensity * 0.9
    f32    = frame.astype(np.float32) / 255.0
    g32    = np.clip(fire / 255.0, 0, 1)
    result = np.clip(f32 + g32 * 0.85, 0, 1)
    return (result * 255).astype(np.uint8)


def _strong_glitch(frame: np.ndarray, t: float, strength: float = 1.0, rng_seed: int = 0) -> np.ndarray:
    """Visible digital glitch effect."""
    result = frame.copy()
    h, w   = frame.shape[:2]
    rng    = random.Random(int(t * 1000) ^ rng_seed)
    n_bands = int(3 + strength * 5)
    for _ in range(n_bands):
        y    = rng.randint(0, h - 1)
        bh   = rng.randint(2, max(3, int(h * 0.06)))
        shift = int((rng.random() - 0.5) * strength * 35)
        y2   = min(y + bh, h)
        if abs(shift) > 0:
            result[y:y2] = np.roll(frame[y:y2], shift, axis=1)
    px = int(strength * 12)
    if px > 0 and px < w:
        result = _strong_rgb_split(result, px)
    if rng.random() > 0.5:
        y    = rng.randint(0, h - 5)
        bh   = rng.randint(1, 4)
        result[y:y+bh] = np.clip(
            result[y:y+bh].astype(np.float32) * (1.5 + strength * 0.8), 0, 255
        ).astype(np.uint8)
    return result


def _strong_speed_lines(frame: np.ndarray, t: float, density: float = 1.0) -> np.ndarray:
    """Visible manga-style speed lines from center."""
    h, w   = frame.shape[:2]
    overlay = np.zeros((h, w), dtype=np.uint8)
    cx, cy  = w // 2, h // 2
    diag    = math.sqrt(w*w + h*h)
    rng     = random.Random(int(t * 100))
    n       = int(60 + density * 80)
    for _ in range(n):
        angle  = rng.uniform(0, 2 * math.pi)
        r_in   = rng.uniform(0.05, 0.15) * min(w, h)
        r_out  = rng.uniform(0.45, 0.80) * diag
        lw     = rng.randint(1, max(2, int(density * 4)))
        alpha  = rng.randint(90, 220)
        x1 = int(cx + r_in  * math.cos(angle))
        y1 = int(cy + r_in  * math.sin(angle))
        x2 = int(cx + r_out * math.cos(angle))
        y2 = int(cy + r_out * math.sin(angle))
        cv2.line(overlay, (x1,y1), (x2,y2), alpha, lw, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (3,3), 0.8)
    ol3     = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR).astype(np.float32)
    f32     = frame.astype(np.float32) / 255.0
    g32     = ol3 / 255.0
    result  = 1.0 - (1.0 - f32) * (1.0 - g32)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _mood_freq_map(mood: str) -> float:
    """Get shake frequency for mood."""
    return {"hype": 4.0, "dark": 2.0, "dramatic": 2.5, "sad": 1.0, "chill": 1.0}.get(mood, 2.0)


def apply_strong_effects_to_segment(
    segment: Any,
    mood: str,
    index: int,
    style: dict[str, Any],
    intelligence: dict[str, Any] | None = None,
    effect_hints: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Drop-in replacement for _apply_mood_effects() with VISIBLE parameters."""
    if MOVIEPY_DISABLED:
        return segment, {}

    intel     = intelligence or {}
    hints     = effect_hints or {}
    intensity = int(style.get("intensity", 3))

    scene_energy  = float(intel.get("scene_energy",  0.5))
    motion        = float(intel.get("motion",         0.5))
    scene_change  = float(intel.get("scene_change",   0.0))
    contrast      = float(intel.get("contrast",       0.5))
    brightness    = float(intel.get("brightness",     0.5))
    face_cx       = float(intel.get("face_center_x",  0.5))
    face_cy       = float(intel.get("face_center_y",  0.42))
    is_peak       = bool(intel.get("is_peak_beat",    False))
    archetype     = str(intel.get("archetype",        "neutral"))
    impact_rank   = float(intel.get("impact_rank",    0.5))

    is_impact = scene_energy > 0.55 or scene_change > 0.20 or is_peak

    wants = lambda k: bool(style.get(k, False)) or bool(hints.get(f"{k}_applied", False))

    # REDUCED AUTO-EFFECTS: Brain render already decides effects per-segment
    # Only apply auto-effects if explicitly requested via style
    brain_render_used = bool(hints.get("brain_render_used", False))
    
    # If brain render was used, trust its decisions - don't auto-apply aggressive effects
    if brain_render_used:
        auto_shake   = False  # Brain render handles animations
        auto_zoom    = False  # Brain render handles zoom
        auto_glitch  = False  # Brain render handles glitch
        auto_flash   = False  # Brain render handles flash
        auto_fire    = False  # Brain render handles fire/aura
        auto_glow    = intensity >= 2 and not style.get("no_color_grade", False)  # Subtle global glow OK
        auto_grain   = intensity >= 2  # Subtle grain OK
        auto_vignette = intensity >= 2 and not style.get("no_color_grade", False)  # Subtle vignette OK
        auto_grade   = not style.get("no_color_grade", False)  # Always apply subtle grade
    else:
        # Fallback pipeline: more conservative auto-effects
        auto_shake   = intensity >= 3 and mood in ("hype", "dark", "dramatic") and is_impact and index % 3 == 0
        auto_zoom    = intensity >= 2 and is_impact and impact_rank > 0.55 and index % 2 == 0
        auto_glow    = intensity >= 2 and not style.get("no_color_grade", False)
        auto_grain   = intensity >= 2
        auto_vignette = intensity >= 2 and not style.get("no_color_grade", False)
        auto_grade   = not style.get("no_color_grade", False)
        auto_flash   = intensity >= 3 and is_impact and scene_change > 0.22 and index % 4 == 0  # Rare
        auto_glitch  = mood == "dark" and intensity >= 3 and is_impact and index % 5 == 0  # Very rare
        auto_fire    = style.get("fire_lines", False) or style.get("energy_aura", False)

    do_grade      = auto_grade  or wants("color_grade")
    do_glow       = auto_glow   or wants("glow")
    do_fire       = auto_fire   or wants("fire_lines") or wants("energy_aura")
    do_glitch     = auto_glitch or wants("glitch")
    do_chromatic  = wants("chroma_pulse") or (auto_glitch and wants("glitch"))  # Only with explicit glitch
    do_shake      = auto_shake  or wants("shake")
    do_zoom       = auto_zoom   or wants("zoom")
    do_speed_lines = wants("speed_lines") or wants("turbulent_displace")
    do_vignette   = auto_vignette or wants("vignette")
    do_grain      = auto_grain  or wants("grain")
    do_flash      = auto_flash  or wants("flash")
    do_bw         = wants("black_white")

    if style.get("simple_mode", False) or intensity == 1:
        do_fire = do_glitch = do_chromatic = do_speed_lines = False
        do_flash = do_shake = False

    applied: dict[str, bool] = {
        "zoom_applied": False, "shake_applied": False,
        "glitch_applied": False, "glow_applied": False,
        "flash_applied": False, "grade_applied": False,
        "fire_lines_applied": False, "energy_aura_applied": False,
        "chroma_pulse_applied": False, "speed_lines_applied": False,
        "vignette_applied": False, "grain_applied": False,
        "black_white_applied": False,
    }
    applied.update({
        "scene_energy": scene_energy, "motion": motion,
        "contrast": contrast, "brightness": brightness,
        "scene_change": scene_change, "archetype": archetype,
        "impact_rank": impact_rank, "is_peak_beat": is_peak,
    })

    energy_boost = 0.7 + scene_energy * 0.6
    int_scale    = 0.6 + intensity * 0.25

    GLOW_STRENGTH    = min(1.5, 0.65 * energy_boost * int_scale)
    SHAKE_AMOUNT     = int(min(22, (8 + scene_energy * 12) * int_scale))
    SHAKE_FREQ       = 10.0 + _mood_freq_map(mood)
    ZOOM_FACTOR      = min(1.28, 1.10 + impact_rank * 0.14 * int_scale)
    RGB_SPLIT_PX     = int(min(18, (7 + scene_change * 10) * int_scale))
    GRAIN_AMOUNT     = min(26, 14 + intensity * 4)
    VIGNETTE_STR     = min(0.88, 0.55 + (1 - brightness) * 0.25)
    FIRE_INTENSITY   = min(1.4, 0.85 * energy_boost * int_scale)
    AURA_INTENSITY   = min(1.4, 0.85 * energy_boost * int_scale)
    GLITCH_STR       = min(1.5, 0.75 * energy_boost * int_scale)
    SPEED_LINES_D    = min(1.5, 0.80 * energy_boost)
    CONTRAST_BOOST   = 14 + int(scene_energy * 16) + intensity * 4
    FLASH_STRENGTH   = min(0.95, 0.72 + impact_rank * 0.20)

    def _ensure_uint8(f):
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0, posinf=255, neginf=0)
                if arr.max() <= 1.5:
                    arr = arr * 255
            return np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    if do_grade and not do_bw:
        if mood == "hype":
            segment = segment.fx(_vfx.lum_contrast, 0, CONTRAST_BOOST, 128)
        elif mood == "dark":
            segment = segment.fx(_vfx.lum_contrast, -8, CONTRAST_BOOST + 8, 128)
        elif mood == "sad":
            segment = segment.fx(_vfx.lum_contrast, -4, CONTRAST_BOOST - 4, 128)
        elif mood == "dramatic":
            segment = segment.fx(_vfx.lum_contrast, 2, CONTRAST_BOOST, 128)
        else:
            segment = segment.fx(_vfx.lum_contrast, 0, CONTRAST_BOOST - 4, 128)
        applied["grade_applied"] = True

    def _frame_effects(get_frame, t):
        frame = _ensure_uint8(get_frame(t))
        if do_grade and not do_bw:
            frame = _strong_color_grade(frame, mood, intensity)
        if do_bw:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if do_glow:
            frame = _strong_glow(frame, GLOW_STRENGTH)
        if do_fire:
            if style.get("fire_lines", False) or style.get("energy_aura", False):
                frame = _strong_fire_lines(frame, t, FIRE_INTENSITY)
                frame = _strong_energy_aura(frame, t, AURA_INTENSITY * 0.7)
            elif is_impact and scene_energy > 0.65:
                frame = _strong_fire_lines(frame, t, FIRE_INTENSITY * 0.6)
        if do_glitch:
            frame = _strong_glitch(frame, t, GLITCH_STR, index)
        if do_chromatic and not do_glitch:
            frame = _strong_rgb_split(frame, RGB_SPLIT_PX)
        if do_speed_lines and is_impact:
            frame = _strong_speed_lines(frame, t, SPEED_LINES_D)
        if do_shake:
            frame = _strong_shake(frame, t, SHAKE_AMOUNT, SHAKE_FREQ)
        if do_vignette:
            frame = _strong_vignette(frame, VIGNETTE_STR)
        if do_grain:
            frame = _strong_grain(frame, GRAIN_AMOUNT, t)
        if do_flash and t < 0.08:
            fade = 1.0 - (t / 0.08)
            frame = _strong_flash(frame, FLASH_STRENGTH * fade)
        return frame

    needs_per_frame = any([
        do_grade, do_bw, do_glow, do_fire, do_glitch,
        do_chromatic, do_speed_lines, do_shake,
        do_vignette, do_grain, do_flash,
    ])
    if needs_per_frame:
        segment = segment.fl(_frame_effects)

    if do_zoom and not do_shake:
        def _zoom_frame(get_frame, t):
            frame = _ensure_uint8(get_frame(t))
            z = ZOOM_FACTOR - (ZOOM_FACTOR - 1.0) * min(1.0, t / max(0.001, segment.duration or 1))
            z = max(1.001, z)
            return _strong_zoom(frame, z)
        segment = segment.fl(_zoom_frame)
        applied["zoom_applied"] = True

    applied.update({
        "glow_applied":       do_glow,
        "grade_applied":      do_grade,
        "fire_lines_applied": do_fire and style.get("fire_lines", False),
        "energy_aura_applied":do_fire and style.get("energy_aura", False),
        "glitch_applied":     do_glitch,
        "chroma_pulse_applied":do_chromatic,
        "speed_lines_applied":do_speed_lines,
        "shake_applied":      do_shake,
        "vignette_applied":   do_vignette,
        "grain_applied":      do_grain,
        "flash_applied":      do_flash,
        "black_white_applied":do_bw,
    })

    return segment, applied


def render_video_with_pyav_pipeline(
    video_path: Path,
    output_path: Path,
    beat_markers: list[float],
    mood: str,
    music_path: Path | None = None,
    prompt_text: str = "",
    variation_seed: int | None = None,
    drop_times: list[float] | None = None,
    visual_vibe: dict[str, Any] | None = None,
    audio_features: dict[str, Any] | None = None,
) -> float:
    style = build_style_profile(prompt_text, mood)
    style = _enforce_high_energy_style(prompt_text, mood, style)
    style = _apply_performance_defaults(prompt_text, style)
    intensity = int(style.get("intensity", 1))
    disable_fast_renderer = bool(style.get("disable_fast_renderer", False))
    ffmpeg_available = shutil.which("ffmpeg") is not None
    style_compatible = archive_fast_render_is_compatible(style, bool(music_path)) if archive_fast_render_is_compatible else False
    can_try_archive_fast = bool(
        FAST_RENDERER_AVAILABLE
        and run_archive_fast_render
        and archive_fast_render_is_compatible
        and ffmpeg_available
        and not disable_fast_renderer
        and style_compatible
    )

    if can_try_archive_fast:
        try:
            app.logger.info("[ARCHIVE] Attempting FFmpeg fast renderer for %s", output_path.name)
            fast_duration = float(
                run_archive_fast_render(
                    video_path=video_path,
                    output_path=output_path,
                    beat_markers=beat_markers,
                    mood=mood,
                    music_path=music_path,
                    music_start=0.0,
                    prompt_text=prompt_text,
                    style=style,
                    intensity=intensity,
                    drop_times=drop_times,
                )
                or 0.0
            )
            if fast_duration > 0 and output_path.exists() and output_path.stat().st_size > 0:
                app.logger.info("[ARCHIVE] Fast FFmpeg renderer succeeded for %s", output_path.name)
                return fast_duration
            app.logger.warning("[ARCHIVE] FFmpeg renderer returned invalid output; switching to PyAV")
        except Exception as fast_exc:
            app.logger.warning("Fast FFmpeg renderer failed; switching to PyAV: %s", fast_exc, exc_info=True)
    else:
        app.logger.info("[ARCHIVE] FFmpeg fast renderer unavailable or incompatible; using PyAV")

    if PYAV_RENDERER_AVAILABLE and render_video_with_pyav:
        pyav_duration = float(
            render_video_with_pyav(
                video_path=str(video_path),
                output_path=str(output_path),
                beat_markers=beat_markers,
                mood=mood,
                music_path=str(music_path) if music_path else None,
                prompt_text=prompt_text,
                variation_seed=variation_seed,
                drop_times=drop_times,
                visual_vibe=visual_vibe,
                audio_features=audio_features,
                style_profile=style,
            )
            or 0.0
        )
        if pyav_duration > 0 and output_path.exists() and output_path.stat().st_size > 0:
            app.logger.info("[ARCHIVE] PyAV renderer succeeded for %s", output_path.name)
            return pyav_duration
        app.logger.warning("[ARCHIVE] PyAV renderer returned invalid output; using FFmpeg safe fallback")

    return render_video_safe_fallback(video_path, output_path, music_path=music_path)


def render_video_with_moviepy(
    video_path: Path,
    output_path: Path,
    beat_markers: list[float],
    mood: str,
    music_path: Path | None = None,
    prompt_text: str = "",
    variation_seed: int | None = None,
    drop_times: list[float] | None = None,
    visual_vibe: dict[str, Any] | None = None,
    audio_features: dict[str, Any] | None = None,
) -> float:
    raise RuntimeError("MoviePy renderer is disabled; use render_video_with_pyav_pipeline instead.")

    with VideoFileClip(str(video_path)) as clip:
        duration = min(float(clip.duration or 0.0), 45.0)
        if duration <= 0:
            raise ValueError("Uploaded video duration is invalid.")

        source_fps = int(clip.fps) if clip.fps else 30
        fps_cap = 30
        if duration >= 30.0 and (not bool(style.get("quality_priority", False))):
            fps_cap = 24
        target_fps = max(24, min(source_fps, fps_cap))

        source_w, source_h = int(clip.size[0]), int(clip.size[1])
        source_pixels = source_w * source_h
        max_pixels = _max_render_pixels(style)
        working_clip = clip
        if source_pixels > max_pixels:
            scale = math.sqrt(max_pixels / float(source_pixels))
            new_w = max(640, int((source_w * scale) // 2 * 2))
            new_h = max(360, int((source_h * scale) // 2 * 2))
            working_clip = clip.resize(newsize=(new_w, new_h))

        cut_density = float(style.get("cut_density", 1.0))
        beat_step = {
            "hype": 2,
            "dark": 2,
            "dramatic": 2,
            "sad": 3,
            "chill": 3,
        }.get(mood, 2)
        rng = random.Random(int(variation_seed or 0))
        beat_step = max(1, beat_step - (1 if intensity >= 2 else 0))
        if intensity >= 3 and duration <= 20:
            beat_step = 1
        elif intensity >= 3:
            beat_step = 2
        if selective_beats and (not all_beats):
            beat_step += 1
        if fast_render:
            beat_step += 1
        if cut_density >= 1.25:
            beat_step = max(1, beat_step - 1)
        elif cut_density <= 0.75:
            beat_step += 1
        beat_phase = rng.randrange(beat_step) if beat_step > 1 else 0

        cut_points = [0.0]
        for idx, t in enumerate(beat_markers):
            if 0 < t < duration and ((idx + beat_phase) % beat_step == 0):
                cut_points.append(float(t))

        if len(cut_points) <= 1:
            interval = 0.8 if mood in {"hype", "dark", "dramatic"} else 1.2
            t = interval
            while t < duration:
                cut_points.append(round(t, 3))
                t += interval

        cut_points = sorted(set(cut_points))
        if cut_points[-1] < duration:
            cut_points.append(duration)
        min_seg = 0.16 if intensity >= 3 else 0.2
        max_segments = 54 if intensity >= 3 else 40
        if selective_beats and (not all_beats):
            max_segments = max(24, max_segments - 10)
        if fast_render:
            max_segments = max(20, max_segments - 8)
        if quality_priority and (not fast_render):
            max_segments = min(64, max_segments + 6)
        cut_points = _compress_cut_points(cut_points, min_gap=min_seg, max_points=max_segments)
        segment_intelligence = analyze_segment_intelligence(video_path, cut_points, source_fps=source_fps)
        segment_intelligence = annotate_cut_scores(
            segment_intelligence,
            cut_points,
            audio_features=audio_features,
            drop_times=drop_times,
            mood=mood,
            style=style,
        )
        segment_intelligence = _annotate_peak_beats(segment_intelligence, style)
        global_plan = build_global_edit_plan(segment_intelligence, mood, style)
        signature_pack = select_signature_pack(
            mood,
            (visual_vibe if isinstance(visual_vibe, dict) else None),
            style,
        )
        if drop_times:
            for drop_t in drop_times:
                for idx in range(len(cut_points) - 1):
                    if cut_points[idx] <= drop_t < cut_points[idx + 1]:
                        for j in range(max(0, idx - 1), min(len(segment_intelligence), idx + 2)):
                            if global_plan.get("phase"):
                                global_plan["phase"][j] = "impact"
                            if global_plan.get("heavy_allow"):
                                global_plan["heavy_allow"][j] = True
                            if global_plan.get("transition_after") and j == idx:
                                global_plan["transition_after"][j] = True
                        break
        effect_hints = predict_effect_hints_for_segments(segment_intelligence, mood, style)
        for idx, row in enumerate(effect_hints):
            if idx < len(segment_intelligence):
                row["global_phase"] = global_plan.get("phase", [])[idx] if global_plan.get("phase") else "flow"
                row["global_allow_heavy"] = bool(global_plan.get("heavy_allow", [True])[idx] if global_plan.get("heavy_allow") else True)
                row["global_energy_rank"] = float(global_plan.get("energy_rank", [0.5])[idx] if global_plan.get("energy_rank") else 0.5)
                row["global_transition_after"] = bool(global_plan.get("transition_after", [False])[idx] if global_plan.get("transition_after") else False)
                transition_hints = global_plan.get("transition_hint", [])
                row["global_transition_hint"] = str(transition_hints[idx] if idx < len(transition_hints) else "cut")
                row["signature_pack"] = signature_pack

        total_segments = max(0, len(cut_points) - 1)
        renderable_indices = [
            i
            for i in range(total_segments)
            if (cut_points[i + 1] - cut_points[i]) >= min_seg
        ]

        wants_loop_clips = bool(style.get("loop_clips", False))
        wants_loop_ending = bool(style.get("loop_ending", False))
        loop_count = int(np.clip(int(style.get("loop_count", 2)), 2, 4))
        repeat_source_indices: list[int] = []
        if wants_loop_clips and len(renderable_indices) >= 2:
            max_repeat_len = 4 if (not fast_render) else 3
            repeat_candidate = _pick_repeat_segment_span(
                [segment_intelligence[i] for i in renderable_indices if i < len(segment_intelligence)],
                min_len=2,
                max_len=max_repeat_len,
            )
            if repeat_candidate is not None:
                span_start, span_end, _span_score = repeat_candidate
                repeat_source_indices = renderable_indices[span_start:span_end]

        if repeat_source_indices:
            repeat_before = renderable_indices.index(repeat_source_indices[-1]) + 1
            segment_plan = (
                renderable_indices[:repeat_before]
                + (repeat_source_indices * (loop_count - 1))
                + renderable_indices[repeat_before:]
            )
        else:
            segment_plan = list(renderable_indices)
        repeat_source_set = set(repeat_source_indices)
        tail_repeat_indices: list[int] = []
        if wants_loop_ending:
            tail_repeat_indices = _pick_tail_repeat_span(
                renderable_indices,
                segment_intelligence,
                min_len=2,
                max_len=3 if intensity <= 2 else 4,
            )

        timeline: list[Any] = []
        label_rows: list[dict[str, Any]] = []
        segment_cache: dict[int, tuple[Any, dict[str, Any]]] = {}
        recent_effects: deque[str] = deque(maxlen=2)
        effect_cooldowns: dict[str, int] = {}
        effect_cooldown_len = {
            "fire_lines": 3,
            "energy_aura": 3,
            "turbulent": 2,
            "glitch": 2,
            "chroma_pulse": 2,
            "speed_ramp": 2,
            "zoom": 2,
            "shake": 1,
            "particles": 2,
            "blend_overlay": 2,
            "mask_roto": 2,
            "chroma_key": 2,
            "difference_matte": 2,
            "object_inpaint": 2,
            "posterize_time": 2,
        }
        transition_size = (int(working_clip.size[0]), int(working_clip.size[1]))
        transition_every = 4 if (intensity >= 3 and duration <= 20) else 5
        if wants_overlap_transitions:
            transition_every = max(2, transition_every - 2)
        if wants_mask_transition:
            transition_every = max(2, transition_every - 1)
        if selective_beats and (not all_beats):
            transition_every = min(7, transition_every + 1)
        if fast_render:
            transition_every = min(8, transition_every + 1)
        if quality_priority and (not fast_render):
            transition_every = max(2, transition_every - 1)
        transition_phase = rng.randrange(transition_every) if transition_every > 1 else 0
        for out_idx, src_idx in enumerate(segment_plan):
            start = cut_points[src_idx]
            end = cut_points[src_idx + 1]

            if src_idx in segment_cache:
                cached_segment, cached_applied = segment_cache[src_idx]
                try:
                    segment = cached_segment.copy()
                except Exception:
                    segment = cached_segment
                applied = dict(cached_applied)
            else:
                seg_len = float(end - start)
                segment = working_clip.subclip(start, end)
                fade_sec = 0.012 if intensity >= 3 else (0.02 if mood in {"hype", "dark"} else 0.04)
                # For overlap transitions, use minimal fades (overlap does the blending)
                # For normal cuts, use stronger fades on longer clips
                if wants_overlap_transitions:
                    fade_sec = 0.004  # Ultra-minimal fade with overlap
                    segment = segment.fx(vfx.fadein, fade_sec).fx(vfx.fadeout, fade_sec)
                elif seg_len > 0.36:
                    fade_sec = max(fade_sec, 0.015)
                    segment = segment.fx(vfx.fadein, fade_sec).fx(vfx.fadeout, fade_sec)
                intel = segment_intelligence[src_idx] if src_idx < len(segment_intelligence) else None
                hints = effect_hints[src_idx] if src_idx < len(effect_hints) else None
                hints = dict(hints or {})
                if recent_effects:
                    hints["avoid_effects"] = list(recent_effects)
                cooldown_block = [k for k, v in effect_cooldowns.items() if v > 0]
                if cooldown_block:
                    if "avoid_effects" in hints:
                        hints["avoid_effects"] = list(set(hints["avoid_effects"] + cooldown_block))
                    else:
                        hints["avoid_effects"] = cooldown_block
                segment, applied = apply_strong_effects_to_segment(
                    segment,
                    mood,
                    src_idx,
                    style,
                    intel,
                    hints,
                )
                segment_cache[src_idx] = (segment, dict(applied))

            if repeat_source_set and (src_idx in repeat_source_set):
                applied["clip_repeat_applied"] = True
                applied["clip_repeat_count"] = loop_count
            else:
                applied["clip_repeat_applied"] = False
                applied["clip_repeat_count"] = 1

            label_rows.append(applied)
            timeline.append(segment)
            applied_order = [
                "zoom",
                "shake",
                "glow",
                "flash",
                "glitch",
                "chroma_pulse",
                "speed_ramp",
                "slow_motion",
                "fast_motion",
                "reverse",
                "boomerang",
                "particles",
                "energy_aura",
                "turbulent",
                "posterize_time",
                "blend_overlay",
                "mask_roto",
                "chroma_key",
                "difference_matte",
                "object_inpaint",
                "fire_lines",
                "black_white",
            ]
            # Cooldown tick
            for k in list(effect_cooldowns.keys()):
                effect_cooldowns[k] = max(0, effect_cooldowns[k] - 1)
            for key in applied_order:
                if applied.get(f"{key}_applied", False) and key not in recent_effects:
                    recent_effects.append(key)
                if applied.get(f"{key}_applied", False):
                    effect_cooldowns[key] = effect_cooldown_len.get(key, 2)

            transition_slot = False
            transition_hint = "cut"
            if src_idx < len(effect_hints):
                transition_slot = bool(effect_hints[src_idx].get("global_transition_after", False))
                transition_hint = str(effect_hints[src_idx].get("global_transition_hint", "cut"))
            if out_idx < len(segment_plan) - 1 and (transition_slot or ((out_idx + transition_phase) % transition_every == 0)):
                transition_style = dict(style)
                transition_style["transition_hint"] = transition_hint
                transition = _make_transition_clip(transition_size, target_fps, mood, src_idx, transition_style)
                if transition is not None:
                    timeline.append(transition)

        if tail_repeat_indices:
            for src_idx in tail_repeat_indices:
                if src_idx in segment_cache:
                    cached_segment, cached_applied = segment_cache[src_idx]
                    try:
                        segment = cached_segment.copy()
                    except Exception:
                        segment = cached_segment
                    applied = dict(cached_applied)
                else:
                    start = cut_points[src_idx]
                    end = cut_points[src_idx + 1]
                    segment = working_clip.subclip(start, end)
                    fade_sec = 0.012 if intensity >= 3 else (0.02 if mood in {"hype", "dark"} else 0.04)
                    # For overlap transitions, use minimal fades (overlap does the blending)
                    # For normal cuts, use stronger fades on longer clips
                    if wants_overlap_transitions:
                        fade_sec = 0.004  # Ultra-minimal fade with overlap
                        segment = segment.fx(vfx.fadein, fade_sec).fx(vfx.fadeout, fade_sec)
                    elif float(end - start) > 0.36:
                        fade_sec = max(fade_sec, 0.015)
                        segment = segment.fx(vfx.fadein, fade_sec).fx(vfx.fadeout, fade_sec)
                    intel = segment_intelligence[src_idx] if src_idx < len(segment_intelligence) else None
                    hints = effect_hints[src_idx] if src_idx < len(effect_hints) else None
                    segment, applied = apply_strong_effects_to_segment(
                        segment,
                        mood,
                        src_idx,
                        style,
                        intel,
                        hints,
                    )
                applied["clip_repeat_applied"] = True
                applied["clip_repeat_count"] = 2
                label_rows.append(applied)
                timeline.append(segment)

        if not timeline:
            timeline = [working_clip.subclip(0, duration)]

        # Force minimum contrast on timeline before concatenation
        contrast_boost = 12 + (intensity * 6)
        timeline = [
            clip.fx(vfx.lum_contrast, 0, contrast_boost, 128)
            for clip in timeline
        ]

        if wants_overlap_transitions:
            overlap_base = 0.035 + (0.01 * intensity)
            overlap_jitter = (rng.random() - 0.5) * 0.012
            overlap_sec = float(np.clip(overlap_base + overlap_jitter, 0.03, 0.085))
            final_clip = concatenate_videoclips(timeline, method="compose", padding=-overlap_sec)
        else:
            final_clip = concatenate_videoclips(timeline, method="chain")
        music_clip = None
        if music_path and music_path.exists():
            try:
                music_clip = AudioFileClip(str(music_path))
                final_duration = float(final_clip.duration or 0.0)
                if float(music_clip.duration or 0.0) < final_duration:
                    music_clip = _audio_loop_clip(music_clip, duration=float(final_clip.duration or 0.0))
                else:
                    if MUSIC_UPGRADE_AVAILABLE and select_music_excerpt_for_video_v2:
                        music_start_sec, _music_info = select_music_excerpt_for_video_v2(
                            music_path,
                            target_duration_sec=final_duration,
                            mood=mood,
                            variation_seed=variation_seed,
                        )
                    else:
                        music_start_sec, _music_info = select_music_excerpt_for_video(
                            music_path,
                            target_duration_sec=final_duration,
                            mood=mood,
                            variation_seed=variation_seed,
                        )
                    music_end_sec = min(float(music_clip.duration or 0.0), music_start_sec + final_duration)
                    if music_end_sec - music_start_sec < 0.2:
                        music_start_sec = 0.0
                        music_end_sec = min(float(music_clip.duration or 0.0), final_duration)
                    music_clip = music_clip.subclip(music_start_sec, music_end_sec)
                    if float(music_clip.duration or 0.0) < final_duration:
                        music_clip = _audio_loop_clip(music_clip, duration=final_duration)
                final_clip = final_clip.set_audio(music_clip)
            except Exception as music_exc:
                try:
                    if music_clip:
                        music_clip.close()
                except Exception:
                    pass
                music_clip = None
                app.logger.warning(
                    "Music track load failed (%s). Falling back to source audio.",
                    music_exc,
                )

        cpu_threads = os.cpu_count() or 4
        encode_threads = max(4, min(16, cpu_threads))
        if bool(style.get("quality_priority", False)):
            default_cpu_preset = "veryfast"
        else:
            if duration >= 30.0:
                default_cpu_preset = "superfast"
            elif duration >= 18.0:
                default_cpu_preset = "veryfast"
            else:
                default_cpu_preset = "veryfast" if intensity <= 2 else "superfast"
        preferred_codec = _pick_best_working_video_codec()

        ffmpeg_params = ["-movflags", "+faststart", "-pix_fmt", "yuv420p"]
        write_kwargs = {
            "audio_codec": "aac",
            "fps": target_fps,
            "threads": encode_threads,
            "temp_audiofile": str(UPLOAD_DIR / f"temp_audio_{uuid.uuid4().hex}.m4a"),
            "remove_temp": True,
            "ffmpeg_params": ffmpeg_params,
            "logger": "bar",
        }
        if preferred_codec == "libx264":
            write_kwargs["preset"] = default_cpu_preset
            ffmpeg_params.extend(["-crf", "18"])
        elif preferred_codec == "h264_nvenc":
            ffmpeg_params.extend(["-preset", "p4", "-rc", "vbr", "-cq", "19"])

        moviepy_output_path = output_path
        temp_post_path: Path | None = None
        if offload_to_ffmpeg:
            temp_post_path = output_path.with_name(f"{output_path.stem}_mpy.mp4")
            moviepy_output_path = temp_post_path

        try:
            final_clip.write_videofile(
                str(moviepy_output_path),
                codec=preferred_codec,
                **write_kwargs,
            )
        except Exception:
            # Turbo Mode: Use ultrafast to slash assembly time
            fallback_kwargs = dict(write_kwargs)
            fallback_kwargs["preset"] = "ultrafast"
            fallback_kwargs["ffmpeg_params"] = ["-movflags", "+faststart", "-pix_fmt", "yuv420p", "-crf", "21"]
            final_clip.write_videofile(
                str(moviepy_output_path),
                codec="libx264",
                **fallback_kwargs,
            )

        if offload_to_ffmpeg and temp_post_path is not None and temp_post_path.exists():
            ok_post = _run_ffmpeg_postprocess(
                temp_post_path,
                output_path,
                ffmpeg_post_filter,
                threads=encode_threads,
            )
            if not ok_post:
                try:
                    if output_path.exists():
                        output_path.unlink(missing_ok=True)
                    temp_post_path.replace(output_path)
                except Exception:
                    shutil.copy2(temp_post_path, output_path)
            temp_post_path.unlink(missing_ok=True)

        _apply_external_brain_polish(
            output_path,
            mood,
            style,
            intensity,
            logger_obj=logger,
            request_id=getattr(g, "request_id", ""),
        )

        out_duration = float(final_clip.duration or 0.0)
        final_clip.close()
        if music_clip:
            music_clip.close()
        append_segment_labels_csv(
            csv_path=output_path.parent / "segment_labels.csv",
            render_name=output_path.name,
            mood=mood,
            style=style,
            rows=label_rows,
        )
        return out_duration


def render_video_safe_fallback(
    video_path: Path,
    output_path: Path,
    music_path: Path | None = None,
) -> float:
    """
    Minimal, resilient render path used only when advanced pipeline fails.
    Keeps API responsive instead of returning HTTP 500.
    """
    meta = probe_video_metadata(video_path)
    duration = min(float(meta.get("duration") or 0.0), 45.0)
    if duration <= 0:
        raise ValueError("Uploaded video duration is invalid.")

    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        app.logger.warning("[ARCHIVE] FFmpeg not found; using passthrough fallback")
        shutil.copy2(video_path, output_path)
        return duration

    music_exists = bool(music_path) and Path(str(music_path)).exists()
    app.logger.info("[ARCHIVE] Safe fallback render: music=%s, duration=%.1fs", music_path if music_exists else "NONE", duration)

    tmp_video = output_path.with_name(f"{output_path.stem}_safe_video.mp4")
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *ffmpeg_hwaccel_input_args(),
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(tmp_video),
    ]
    proc = run_subprocess(cmd, capture_output=True, text=True, timeout=max(120, int(duration * 8)))
    if not proc.ok:
        shutil.copy2(video_path, output_path)
        return duration
    try:
        if music_path and music_path.exists():
            mux_cmd = [
                ffmpeg_cmd,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_video),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        else:
            mux_cmd = [
                ffmpeg_cmd,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_video),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        proc_mux = run_subprocess(mux_cmd, capture_output=True, text=True, timeout=max(120, int(duration * 8)))
        if not proc_mux.ok:
            app.logger.warning("[ARCHIVE] Music muxing failed (stderr=%s); using video-only fallback", proc_mux.stderr[:200] if proc_mux.stderr else "none")
            shutil.copy2(tmp_video, output_path)
        else:
            app.logger.info("[ARCHIVE] Safe fallback render complete with music")
        return duration
    finally:
        tmp_video.unlink(missing_ok=True)


def _passthrough_video_fallback(input_path: Path, output_path: Path) -> float:
    """
    Absolute last-resort fallback: return the uploaded video as output.
    Prevents user-facing 500s when local codec/environment fails.
    """
    shutil.copy2(input_path, output_path)
    return float(probe_video_metadata(output_path).get("duration") or 0.0)


def build_edit_actions(beat_markers: list[float], mood: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    transition = {
        "hype": "flash_cut",
        "dark": "glitch_dissolve",
        "sad": "crossfade",
        "dramatic": "whip_pan",
        "chill": "soft_blur_fade",
    }.get(mood, "cut")

    effect = {
        "hype": "velocity_ramp",
        "dark": "rgb_split",
        "sad": "slow_motion",
        "dramatic": "zoom_punch",
        "chill": "film_grain",
    }.get(mood, "none")

    for idx, t in enumerate(beat_markers):
        if idx == 0:
            continue

        if idx % 4 == 0:
            actions.append(
                {
                    "time": t,
                    "type": "transition",
                    "name": transition,
                    "strength": "high",
                }
            )
        elif idx % 2 == 0:
            actions.append(
                {
                    "time": t,
                    "type": "effect",
                    "name": effect,
                    "strength": "medium",
                }
            )

    return actions


@app.get("/api/health")
def health() -> Any:
    """Enhanced health check with live diagnostics."""
    # Live checks (not cached at startup).
    live_checks: dict[str, Any] = {}
    try:
        upload_ok = _can_write_to_dir(UPLOAD_DIR)
        output_ok = _can_write_to_dir(OUTPUT_DIR)
        live_checks["upload_dir_writable"] = upload_ok
        live_checks["output_dir_writable"] = output_ok
    except Exception:
        live_checks["upload_dir_writable"] = False
        live_checks["output_dir_writable"] = False
    # Disk space check.
    try:
        disk_usage = shutil.disk_usage(str(RUNTIME_DIR))
        live_checks["disk_free_gb"] = round(disk_usage.free / (1024 ** 3), 2)
        live_checks["disk_total_gb"] = round(disk_usage.total / (1024 ** 3), 2)
        live_checks["disk_ok"] = disk_usage.free > (512 * 1024 * 1024)  # > 512 MB free
    except Exception:
        live_checks["disk_ok"] = True  # Don't fail health on unsupported OS.
    # Render queue status.
    with _RENDER_STATUS_LOCK:
        active_renders = sum(
            1 for p in _RENDER_STATUS.values()
            if not p.get("done", True)
        )
    live_checks["active_renders"] = active_renders
    live_checks["max_upload_mb"] = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
    live_checks["auth_enabled"] = bool(ARCHIVE_API_KEY)
    live_checks["rate_limiting_enabled"] = LIMITER_AVAILABLE
    all_ok = bool(
        live_checks.get("upload_dir_writable", False)
        and live_checks.get("output_dir_writable", False)
        and live_checks.get("disk_ok", True)
    )
    return jsonify({"ok": all_ok, **STARTUP_DIAGNOSTICS, "live": live_checks})


@app.get("/api/render-status/<render_id>")
def render_status(render_id: str) -> Any:
    rid = str(render_id or "").strip()
    if not rid:
        return jsonify({"error": "render_id is required"}), 400
    payload = _get_render_status(rid)
    if not payload:
        return jsonify({"ok": False, "render_id": rid, "stage": "unknown", "message": "Render id not found."}), 404
    return jsonify({"ok": True, **payload})


@app.post("/api/cancel/<render_id>")
def cancel_render(render_id: str) -> Any:
    """Cancel an ongoing render by render_id."""
    rid = str(render_id or "").strip()
    if not rid:
        return jsonify({"error": "render_id is required"}), 400
    
    # Mark the render as cancelled
    _mark_render_cancelled(rid)
    
    # Update the status to show it was cancelled
    _set_render_status(
        rid,
        stage="cancelled",
        progress_pct=100.0,
        message="Render cancelled by user.",
        done=True,
        ok=False,
        extras={"reason": "user_cancelled"}
    )
    
    app.logger.info("[ARCHIVE] Render %s cancelled by user", rid)
    return jsonify({"ok": True, "render_id": rid, "message": "Render cancellation requested."})


@app.post("/api/plan")
def plan() -> Any:
    body = request.get_json(silent=True) or {}

    prompt = str(body.get("prompt", "")).strip()
    video_duration = float(body.get("video_duration_sec", 0) or 0)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    mood = infer_mood(prompt)
    style_profile = build_style_profile(prompt, mood)
    style_profile = _enforce_high_energy_style(prompt, mood, style_profile)
    query = f"{MOOD_TO_QUERY.get(mood, 'phonk')} anime edit"
    deny_music = bool(style_profile.get("no_music", False))
    deny_phonk = bool(style_profile.get("no_phonk", False))
    req_music_mood, req_music_strict = _infer_prompt_music_mood(prompt, mood)
    ptxt = prompt.lower()
    romantic_intent = any(t in ptxt for t in ("romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"))
    happy_intent = any(t in ptxt for t in ("happy", "joy", "cheerful", "uplifting", "smile", "positive"))
    sad_or_dark_intent = any(t in ptxt for t in ("sad", "heartbreak", "cry", "tears", "dark", "villain", "shadow"))
    force_non_phonk_music = bool(
        req_music_strict and req_music_mood == "chill" and (romantic_intent or happy_intent) and (not sad_or_dark_intent)
    )

    warnings: list[str] = []
    if deny_music:
        tracks = []
        warnings.append("Prompt requests original audio only (no added music).")
    else:
        tracks = build_local_tracks_for_mood(mood, limit=5, allow_phonk=(not (deny_phonk or force_non_phonk_music)))
        if tracks:
            warnings.append("Using local music library.")
        else:
            tracks = build_fallback_tracks(mood) if (not deny_phonk and not force_non_phonk_music) else []
            if tracks:
                warnings.append("No local music files found. Using built-in fallback tracks.")
            else:
                warnings.append("No allowed local music found; will keep source audio.")

    chosen_track = tracks[0] if tracks else None
    bpm = MOOD_TO_BPM.get(mood, 130)

    if chosen_track and chosen_track.duration_sec and video_duration <= 0:
        video_duration = float(chosen_track.duration_sec)

    beat_markers = generate_beat_markers(video_duration, bpm)
    edit_actions = build_edit_actions(beat_markers, mood)

    return jsonify(
        {
            "mood": mood,
            "music_query": query,
            "chosen_track": None
            if not chosen_track
            else {
                "id": chosen_track.id,
                "title": chosen_track.title,
                "tags": chosen_track.tags,
                "duration_sec": chosen_track.duration_sec,
                "preview_url": chosen_track.preview_url,
                "audio_url": chosen_track.audio_url,
                "user": chosen_track.user,
                "source": chosen_track.source,
            },
            "track_options": [
                {
                    "id": t.id,
                    "title": t.title,
                    "duration_sec": t.duration_sec,
                    "audio_url": t.audio_url,
                    "user": t.user,
                    "source": t.source,
                }
                for t in tracks[:5]
            ],
            "estimated_bpm": bpm,
            "style_profile": style_profile,
            "video_duration_sec": round(video_duration, 2),
            "beat_markers": beat_markers[:300],
            "edit_actions": edit_actions[:200],
            "warning": " ".join(warnings) if warnings else None,
        }
    )


@app.post("/api/render")
@_rate_limit("5 per minute")
def render() -> Any:
    global LAST_SELECTED_TRACK_KEY
    render_id = str(request.form.get("render_id", "")).strip() or uuid.uuid4().hex
    prompt = str(request.form.get("prompt", "")).strip()
    video_file = request.files.get("video")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if not video_file or not video_file.filename:
        return jsonify({"error": "video file is required"}), 400
    if not is_allowed_video(video_file.filename):
        return jsonify({"error": "unsupported video format"}), 400

    mood = infer_mood(prompt)
    style_profile = build_style_profile(prompt, mood)
    deny_music = bool(style_profile.get("no_music", False))
    deny_phonk = bool(style_profile.get("no_phonk", False))
    req_music_mood, req_music_strict = _infer_prompt_music_mood(prompt, mood)
    ptxt = prompt.lower()
    romantic_intent = any(t in ptxt for t in ("romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"))
    happy_intent = any(t in ptxt for t in ("happy", "joy", "cheerful", "uplifting", "smile", "positive"))
    sad_or_dark_intent = any(t in ptxt for t in ("sad", "heartbreak", "cry", "tears", "dark", "villain", "shadow"))
    force_non_phonk_music = bool(
        req_music_strict and req_music_mood == "chill" and (romantic_intent or happy_intent) and (not sad_or_dark_intent)
    )
    prompt_l = prompt.lower()
    deterministic_requested = any(
        token in prompt_l for token in ("same output", "deterministic", "repeatable", "exact same", "fixed seed")
    )
    variation_seed: int | None = None
    if ARCHIVE_VARIATION_MODE and not deterministic_requested:
        variation_seed = int((time.time_ns() ^ uuid.uuid4().int ^ (os.getpid() << 12)) & 0x7FFFFFFF)
    edit_brain_loaded = _load_edit_brain_artifact() is not None
    encoder_codec = _pick_best_working_video_codec()
    safe_name = secure_filename(video_file.filename)
    upload_ext = Path(safe_name).suffix.lower()
    if upload_ext not in ALLOWED_VIDEO_EXTENSIONS:
        upload_ext = ".mp4"
    # Keep upload path short on Windows to avoid Errno 22 from deep/long paths.
    upload_name = f"{uuid.uuid4().hex}{upload_ext}"
    input_path = UPLOAD_DIR / upload_name
    video_file.save(str(input_path))
    _cleanup_old_render_outputs()

    warnings: list[str] = []
    anime_check: dict[str, Any] | None = None
    content_check: dict[str, Any] | None = None
    unified_visual: dict[str, Any] | None = None
    audio_analysis: dict[str, Any] | None = None
    started_at = time.perf_counter()
    _set_render_status(
        render_id,
        stage="accepted",
        progress_pct=2.0,
        message="Render request accepted. Preparing analysis pipeline...",
        done=False,
        ok=None,
    )
    try:
        _set_render_status(
            render_id,
            stage="analyze_video",
            progress_pct=8.0,
            message="Analyzing uploaded video content...",
            done=False,
        )
        app.logger.info("[ARCHIVE] Stage 0/6: unified video analysis")
        unified = analyze_video_unified(input_path)
        unified_visual = unified.get("visual") if isinstance(unified, dict) else None
        if ARCHIVE_CONTENT_SAFETY_MODE:
            content_check = unified.get("content_check") if isinstance(unified, dict) else None
            if not isinstance(content_check, dict):
                content_check = {"is_explicit": False, "confidence": 0.0}
        if ARCHIVE_ANIME_ONLY_MODE:
            anime_check = unified.get("anime_check") if isinstance(unified, dict) else None
            if not isinstance(anime_check, dict):
                anime_check = {"is_anime": True, "anime_confidence": 0.5, "real_confidence": 0.5}

        if ARCHIVE_CONTENT_SAFETY_MODE:
            if bool(content_check.get("is_explicit", False)):
                return (
                    jsonify(
                        {
                            "error": "Cannot edit explicit or pornographic content.",
                            "reason": "explicit_content_detected",
                            "note": "Romantic scenes like kissing are allowed.",
                            "content_check": content_check,
                            "build": BACKEND_BUILD,
                        }
                    ),
                    400,
                )

        if ARCHIVE_ANIME_ONLY_MODE:
            if not bool(anime_check.get("is_anime", False)):
                return (
                    jsonify(
                        {
                            "error": "Can only edit anime videos for now.",
                            "reason": "non_anime_video_detected",
                            "anime_check": anime_check,
                            "build": BACKEND_BUILD,
                        }
                    ),
                    400,
                )

        app.logger.info("[ARCHIVE] Stage 1/5: selecting music track")
        _set_render_status(
            render_id,
            stage="music_select",
            progress_pct=22.0,
            message="Selecting best music track for your vibe...",
            done=False,
        )
        t1 = time.perf_counter()
        skip_music_search = False
        if deny_music:
            local_track = None
            track_selection = {"reason": "prompt_disabled_music", "no_music": True, "no_phonk": deny_phonk}
            warnings.append("Prompt requests original audio only (no added music/phonk).")
        else:
            if MUSIC_UPGRADE_AVAILABLE and pick_local_phonk_track_v2:
                def _vibe_override(_p: Path) -> dict[str, Any]:
                    if isinstance(unified_visual, dict):
                        return unified_visual
                    return analyze_video_vibe(_p)

                local_track, track_selection = pick_local_phonk_track_v2(
                    mood=mood,
                    video_path=input_path,
                    prompt_text=prompt,
                    allow_phonk=(not (deny_phonk or force_non_phonk_music)),
                    variation_seed=variation_seed,
                    recent_tracks=RECENT_TRACKS,
                    last_selected_key=LAST_SELECTED_TRACK_KEY,
                    analyze_video_vibe_fn=_vibe_override,
                    analyze_track_audio_vibe_fn=analyze_track_audio_vibe,
                    build_style_profile_fn=build_style_profile,
                    infer_prompt_music_profile_fn=infer_prompt_music_profile,
                    list_local_phonk_files_fn=list_local_phonk_files,
                    relative_track_key_fn=_relative_track_key,
                    infer_track_mood_from_name_fn=infer_track_mood_from_name,
                    mood_to_bpm=MOOD_TO_BPM,
                    mood_keywords=MOOD_KEYWORDS,
                    infer_prompt_music_mood_fn=_infer_prompt_music_mood,
                    is_phonk_like_fn=_is_phonk_like_track,
                    track_matches_folder_hints_fn=_track_matches_folder_hints,
                    infer_prompt_music_folder_hints_fn=_infer_prompt_music_folder_hints,
                )
            else:
                local_track, track_selection = pick_local_phonk_track(
                    mood,
                    video_path=input_path,
                    prompt_text=prompt,
                    allow_phonk=(not (deny_phonk or force_non_phonk_music)),
                    variation_seed=variation_seed,
                    visual_override=unified_visual,
                )
            if isinstance(track_selection, dict) and track_selection.get("reason") in {"no_folder_match", "no_non_phonk_for_chill"}:
                skip_music_search = True
                if track_selection.get("reason") == "no_non_phonk_for_chill":
                    warnings.append("No non-phonk happy/romantic tracks found. Using source audio only.")
                else:
                    warnings.append(
                        "Requested folder music (happy/romantic) was not found in local library. Using source audio only."
                    )
        t_select = time.perf_counter() - t1
        source_meta = probe_video_metadata(input_path)
        source_duration = float(source_meta.get("duration") or 0.0)
        source_fps = float(source_meta.get("fps") or 30.0)

        if deny_music or skip_music_search:
            t_proxy_rank = 0.0
        else:
            app.logger.info("[ARCHIVE] Stage 2/5: proxy ranking top music candidates")
            _set_render_status(
                render_id,
                stage="music_proxy_rank",
                progress_pct=34.0,
                message="Ranking top music candidates (proxy analysis)...",
                done=False,
            )
            t_proxy = time.perf_counter()
            proxy_track, proxy_eval, proxy_warnings = evaluate_top_phonk_candidates(
                video_path=input_path,
                mood=mood,
                prompt_text=prompt,
                source_duration=source_duration,
                source_fps=source_fps,
                track_selection=track_selection,
                allow_phonk=(not (deny_phonk or force_non_phonk_music)),
                variation_seed=variation_seed,
            )
            t_proxy_rank = time.perf_counter() - t_proxy
            warnings.extend(proxy_warnings[:2])
            if isinstance(track_selection, dict):
                track_selection["proxy_evaluation"] = proxy_eval
            if proxy_track is not None:
                local_track = proxy_track
                warnings.append(f"Proxy ranking selected track: {proxy_track.name}")
            if local_track is None:
                fallback_tracks = list_local_phonk_files(allow_phonk=(not (deny_phonk or force_non_phonk_music)))
                if fallback_tracks:
                    if variation_seed is not None and len(fallback_tracks) > 1:
                        rng = random.Random(int(variation_seed) + 89)
                        local_track = rng.choice(fallback_tracks[: min(5, len(fallback_tracks))])
                    else:
                        local_track = fallback_tracks[0]
                    warnings.append(f"Track fallback selected: {local_track.name}")

        if local_track is not None and deny_phonk and _is_phonk_like_track(local_track):
            warnings.append("Filtered phonk track due to prompt. Using source audio.")
            local_track = None
        if local_track is not None and force_non_phonk_music and _is_phonk_like_track(local_track):
            warnings.append("Filtered phonk track due to happy/romantic prompt. Using source audio.")
            local_track = None

        if local_track is not None:
            LAST_SELECTED_TRACK_KEY = _relative_track_key(local_track)
            try:
                audio_analysis = analyze_audio_features(local_track, duration_sec=min(60.0, max(12.0, source_duration)))
            except Exception as audio_exc:
                audio_analysis = {"ok": False, "reason": f"audio_feature_failed:{audio_exc}"}

        # Check if render was cancelled
        if _is_render_cancelled(render_id):
            app.logger.info("[ARCHIVE] Render %s was cancelled by user", render_id)
            return jsonify({"ok": False, "reason": "user_cancelled", "render_id": render_id, "build": BACKEND_BUILD}), 200

        app.logger.info("[ARCHIVE] Stage 3/5: beat detection")
        _set_render_status(
            render_id,
            stage="beat_detect",
            progress_pct=48.0,
            message="Detecting beats and timing map...",
            done=False,
        )
        t2 = time.perf_counter()
        if local_track:
            try:
                detected_bpm, beat_markers, beat_warnings = detect_beats_from_audio_file(
                    local_track, source_duration, mood
                )
            except Exception as beat_exc:
                detected_bpm = int(MOOD_TO_BPM.get(mood, 130))
                beat_markers = generate_beat_markers(source_duration, detected_bpm)
                beat_warnings = [f"Beat detect fallback used ({beat_exc})."]
            warnings.append(
                "Selected local audio: "
                f"{_relative_track_key(local_track)} | visual mood={track_selection.get('visual_vibe', {}).get('visual_mood')} "
                f"| target tempo={track_selection.get('target_tempo')}"
            )
        else:
            try:
                detected_bpm, _, beat_markers, beat_warnings = detect_beats_with_librosa(input_path, mood)
                if audio_analysis is None:
                    try:
                        audio_analysis = analyze_audio_features(input_path, duration_sec=min(60.0, max(12.0, source_duration)))
                    except Exception:
                        audio_analysis = None
            except Exception as beat_exc:
                detected_bpm = int(MOOD_TO_BPM.get(mood, 130))
                safe_duration = float(probe_video_metadata(input_path).get("duration") or source_duration or 0.0)
                beat_markers = generate_beat_markers(safe_duration, detected_bpm)
                beat_warnings = [f"Video-audio beat fallback used ({beat_exc})."]
            warnings.append("No local music track selected; synced to video audio.")
        t_beats = time.perf_counter() - t2
        warnings.extend(beat_warnings)

        edit_actions = build_edit_actions(beat_markers, mood)
        drop_times: list[float] = []
        if local_track:
            drop_times = detect_music_drops(local_track, source_duration)
        if isinstance(audio_analysis, dict) and isinstance(audio_analysis.get("drops"), list):
            merged_drops = list(drop_times)
            for drop_t in audio_analysis.get("drops", []):
                try:
                    f_drop = float(drop_t)
                except Exception:
                    continue
                if 0.0 < f_drop < max(source_duration, 0.1) and all(abs(f_drop - old) > 1.5 for old in merged_drops):
                    merged_drops.append(round(f_drop, 3))
            drop_times = sorted(merged_drops)[:5]

        app.logger.info("[ARCHIVE] Stage 4/5: render timeline + encode")
        _set_render_status(
            render_id,
            stage="render_encode",
            progress_pct=62.0,
            message="Rendering timeline and encoding video...",
            done=False,
        )
        
        # Check if render was cancelled before starting heavy encoding
        if _is_render_cancelled(render_id):
            app.logger.info("[ARCHIVE] Render %s was cancelled by user (before encoding)", render_id)
            return jsonify({"ok": False, "reason": "user_cancelled", "render_id": render_id, "build": BACKEND_BUILD}), 200
        
        t3 = time.perf_counter()
        output_name = f"render_{uuid.uuid4().hex}.mp4"
        output_path = OUTPUT_DIR / output_name
        brain_meta: dict[str, Any] | None = None
        use_brain = bool(style_profile.get("brain_render", True)) and BRAIN_RENDER_AVAILABLE
        if use_brain and not shutil.which("ffmpeg"):
            use_brain = False
            warnings.append("Brain renderer requires FFmpeg; falling back to PyAV/FFmpeg pipeline.")
        
        # Log music file status
        if local_track and Path(str(local_track)).exists():
            app.logger.info("[ARCHIVE] Music file ready: %s (%d bytes)", local_track, Path(str(local_track)).stat().st_size)
        else:
            app.logger.warning("[ARCHIVE] Music file missing or invalid: %s", local_track)
        
        app.logger.info("[ARCHIVE] Render config: use_brain=%s, intensity=%s, brain_available=%s", 
                       use_brain, style_profile.get("intensity", 2), BRAIN_RENDER_AVAILABLE)
        try:
            if use_brain and run_brain_render:
                app.logger.info("[ARCHIVE] Stage 4/5: brain render pipeline")
                _set_render_status(
                    render_id,
                    stage="brain_render",
                    progress_pct=70.0,
                    message="Applying brain render decisions and effects...",
                    done=False,
                )
                brain_result = _run_with_timeout(
                    "brain_render_pipeline",
                    lambda: run_brain_render(
                        video_path=input_path,
                        output_path=output_path,
                        mood=mood,
                        prompt=prompt,
                        intensity=int(style_profile.get("intensity", 2)),
                        music_path=local_track,
                        beat_markers=beat_markers,
                        style_profile=style_profile,
                        variation_seed=variation_seed,
                        target_fps=30,
                        tmp_dir=RUNTIME_DIR / "brain_tmp",
                        cleanup_tmp=True,
                    ),
                    ARCHIVE_BRAIN_TIMEOUT_SEC,
                    app.logger,
                )
                if isinstance(brain_result, dict) and brain_result.get("ok"):
                    render_duration = float(brain_result.get("total_time_sec", 0.0))
                    render_meta = brain_result.get("render_meta", {}) if isinstance(brain_result, dict) else {}
                    brain_meta = {
                        "brain_quality_score": brain_result.get("quality_score"),
                        "music_drop_time": brain_result.get("music_drop_time"),
                        "arc_phases": brain_result.get("arc_phases"),
                        "render_time_min": brain_result.get("total_time_min"),
                        "narrative_director_connected": bool(render_meta.get("narrative_director_connected", False)),
                        "cinematic_director_segments": render_meta.get("cinematic_director_segments"),
                        "explainability_log": render_meta.get("director_explainability_log", []),
                        "narrative_summary": render_meta.get("director_narrative_summary", []),
                    }

                    _set_render_status(
                        render_id,
                        stage="brain_render_polish",
                        progress_pct=85.0,
                        message="Applying heavy visual effects and color grading...",
                        done=False,
                    )

                    _apply_external_brain_polish(
                        output_path,
                        mood,
                        style_profile,
                        int(style_profile.get("intensity", 2)),
                        logger_obj=app.logger,
                        request_id=getattr(g, "request_id", ""),
                    )
                else:
                    reason = brain_result.get("reason") if isinstance(brain_result, dict) else "unknown"
                    render_meta = brain_result.get("render_meta", {}) if isinstance(brain_result, dict) else {}
                    app.logger.warning(
                        "[ARCHIVE] Brain render failed: reason=%s, segments=%s/%s, timed_out=%s", 
                        reason,
                        render_meta.get("segments_rendered", "?"),
                        render_meta.get("segments_requested", "?"),
                        render_meta.get("timed_out", False),
                    )
                    warnings.append(f"Brain renderer failed ({reason}); falling back to PyAV/FFmpeg pipeline.")
                    try:
                        output_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _set_render_status(
                        render_id,
                        stage="pyav_fallback",
                        progress_pct=72.0,
                        message="Brain render fallback triggered, switching to PyAV/FFmpeg pipeline...",
                        done=False,
                    )
                    render_duration = render_video_with_pyav_pipeline(
                        input_path,
                        output_path,
                        beat_markers,
                        mood,
                        music_path=local_track,
                        prompt_text=prompt,
                        variation_seed=variation_seed,
                        drop_times=drop_times,
                        visual_vibe=unified_visual,
                        audio_features=audio_analysis,
                    )
            else:
                render_duration = render_video_with_pyav_pipeline(
                    input_path,
                    output_path,
                    beat_markers,
                    mood,
                    music_path=local_track,
                    prompt_text=prompt,
                    variation_seed=variation_seed,
                    drop_times=drop_times,
                    visual_vibe=unified_visual,
                    audio_features=audio_analysis,
                )
        except OSError as render_os_exc:
            app.logger.error(
                "Primary render pipeline failed with OSError. Using safe fallback.\n%s",
                traceback.format_exc(),
            )
            warnings.append(f"Primary renderer fallback used ({render_os_exc}).")
            render_duration = render_video_safe_fallback(
                input_path,
                output_path,
                music_path=local_track,
            )
        t_render = time.perf_counter() - t3
        app.logger.info("[ARCHIVE] Stage 5/5: done")
        _set_render_status(
            render_id,
            stage="finalizing",
            progress_pct=95.0,
            message="Finalizing output file...",
            done=False,
        )
        _cleanup_old_render_outputs(keep_names={output_name})
        prune_runtime_temp(RUNTIME_DIR)
        EDITOR_MEMORY.remember_render(
            mood,
            style_profile,
            transitions=[transition_name_for_mood(mood, int(style_profile.get("intensity", 1)))],
        )
        EDITOR_MEMORY.remember_feedback(
            str(style_profile.get("editor_style", "anime")),
            {
                "render_success": 1.0,
                "processing_time_sec": float(t_render),
                "beat_count": float(len(beat_markers)),
                "drop_count": float(len(drop_times)),
                "audio_beat_confidence": float((audio_analysis or {}).get("beat_confidence", 0.0)) if isinstance(audio_analysis, dict) else 0.0,
                "transition_density": float(style_profile.get("transition_density", 1.0)),
                "cut_density": float(style_profile.get("cut_density", 1.0)),
            },
        )
        save_editor_memory(EDITOR_MEMORY_PATH, EDITOR_MEMORY)
        total_processing = round(time.perf_counter() - started_at, 2)
        _set_render_status(
            render_id,
            stage="done",
            progress_pct=100.0,
            message="Render completed.",
            done=True,
            ok=True,
            extras={
                "processing_time_sec": total_processing,
                "render_url": f"/api/output/{output_name}",
                "download_url": f"/api/download/{output_name}",
            },
        )

        return jsonify(
            {
                "ok": True,
                "render_id": render_id,
                "mood": mood,
                "detected_bpm": detected_bpm,
                "style_profile": style_profile,
                "scene_intelligence": True,
                "edit_brain_loaded": edit_brain_loaded,
                "encoder_codec": encoder_codec,
                "source_duration_sec": round(source_duration, 2),
                "render_duration_sec": round(render_duration, 2),
                "processing_time_sec": total_processing,
                "timings_sec": {
                    "track_select": round(t_select, 2),
                    "proxy_rank": round(t_proxy_rank, 2),
                    "beat_detect": round(t_beats, 2),
                    "render_encode": round(t_render, 2),
                },
                "brain_meta": brain_meta,
                "beat_markers": beat_markers[:300],
                "edit_actions": edit_actions[:200],
                "selected_phonk": _relative_track_key(local_track) if local_track else None,
                "selected_audio": _relative_track_key(local_track) if local_track else None,
                "variation_seed": variation_seed,
                "anime_check": anime_check,
                "content_check": content_check,
                "track_selection": track_selection,
                "audio_analysis": None
                if not isinstance(audio_analysis, dict)
                else {
                    "ok": audio_analysis.get("ok"),
                    "tempo": audio_analysis.get("tempo"),
                    "beat_confidence": audio_analysis.get("beat_confidence"),
                    "rms_mean": audio_analysis.get("rms_mean"),
                    "onset_mean": audio_analysis.get("onset_mean"),
                    "spectral_contrast": audio_analysis.get("spectral_contrast"),
                    "percussive_ratio": audio_analysis.get("percussive_ratio"),
                    "drops": audio_analysis.get("drops", [])[:5] if isinstance(audio_analysis.get("drops"), list) else [],
                },
                "labels_csv": str((OUTPUT_DIR / "segment_labels.csv").name),
                "render_url": f"/api/output/{output_name}",
                "download_url": f"/api/download/{output_name}",
                "warning": " ".join(warnings) if warnings else None,
            }
        )
    except OSError as exc:
        trace = traceback.format_exc()
        _set_render_status(
            render_id,
            stage="error",
            progress_pct=100.0,
            message=f"Render failed with OSError: {type(exc).__name__}",
            done=True,
            ok=False,
        )
        app.logger.error("Render OSError failed:\n%s", trace)
        try:
            output_name = f"render_{uuid.uuid4().hex}.mp4"
            output_path = OUTPUT_DIR / output_name
            try:
                fallback_duration = render_video_safe_fallback(
                    input_path,
                    output_path,
                    music_path=None,
                )
                fallback_mode = "safe_render"
            except Exception:
                fallback_duration = _passthrough_video_fallback(input_path, output_path)
                fallback_mode = "passthrough_copy"
            _cleanup_old_render_outputs(keep_names={output_name})
            return jsonify(
                {
                    "ok": True,
                    "render_id": render_id,
                    "mood": mood,
                    "style_profile": style_profile,
                    "scene_intelligence": False,
                    "edit_brain_loaded": edit_brain_loaded,
                    "encoder_codec": encoder_codec,
                    "source_duration_sec": 0.0,
                    "render_duration_sec": round(float(fallback_duration), 2),
                    "processing_time_sec": round(time.perf_counter() - started_at, 2),
                    "timings_sec": {},
                    "beat_markers": [],
                    "edit_actions": [],
                    "selected_phonk": None,
                    "selected_audio": None,
                    "track_selection": {"method": f"oserror_{fallback_mode}"},
                    "labels_csv": str((OUTPUT_DIR / "segment_labels.csv").name),
                    "render_url": f"/api/output/{output_name}",
                    "download_url": f"/api/download/{output_name}",
                    "degraded": True,
                    "edited": False,
                    "warning": f"OSError fallback used ({fallback_mode}): {exc}",
                    **({"debug_traceback": trace[-2500:]} if API_DEBUG else {}),
                }
            )
        except Exception:
            app.logger.error("Fallback after OSError also failed:\n%s", traceback.format_exc())
            payload: dict[str, Any] = {"error": f"render failed: {type(exc).__name__}: {exc}", "build": BACKEND_BUILD}
            if API_DEBUG:
                payload["debug_traceback"] = trace[-3000:]
            return jsonify(payload), 500
    except Exception as exc:
        trace = traceback.format_exc()
        _set_render_status(
            render_id,
            stage="error",
            progress_pct=100.0,
            message=f"Render failed: {type(exc).__name__}",
            done=True,
            ok=False,
        )
        app.logger.error("Render failed:\n%s", trace)
        try:
            output_name = f"render_{uuid.uuid4().hex}.mp4"
            output_path = OUTPUT_DIR / output_name
            fallback_duration = _passthrough_video_fallback(input_path, output_path)
            _cleanup_old_render_outputs(keep_names={output_name})
            return jsonify(
                {
                    "ok": True,
                    "render_id": render_id,
                    "mood": mood,
                    "style_profile": style_profile,
                    "scene_intelligence": False,
                    "edit_brain_loaded": edit_brain_loaded,
                    "encoder_codec": encoder_codec,
                    "source_duration_sec": 0.0,
                    "render_duration_sec": round(float(fallback_duration), 2),
                    "processing_time_sec": round(time.perf_counter() - started_at, 2),
                    "timings_sec": {},
                    "beat_markers": [],
                    "edit_actions": [],
                    "selected_phonk": None,
                    "selected_audio": None,
                    "track_selection": {"method": "exception_passthrough_copy"},
                    "labels_csv": str((OUTPUT_DIR / "segment_labels.csv").name),
                    "render_url": f"/api/output/{output_name}",
                    "download_url": f"/api/download/{output_name}",
                    "degraded": True,
                    "edited": False,
                    "warning": f"Exception fallback used (passthrough_copy): {type(exc).__name__}: {exc}",
                    **({"debug_traceback": trace[-2500:]} if API_DEBUG else {}),
                }
            )
        except Exception:
            payload = {"error": f"render failed: {type(exc).__name__}: {exc}", "build": BACKEND_BUILD}
            if API_DEBUG:
                payload["debug_traceback"] = trace[-3000:]
            return jsonify(payload), 500
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/api/output/<path:filename>")
def output_file(filename: str) -> Any:
    # Outputs are always flat files: reject any path component (traversal guard).
    safe_name = Path(filename).name
    if safe_name != filename:
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(OUTPUT_DIR, safe_name, as_attachment=False)


@app.get("/api/download/<path:filename>")
def download_file(filename: str) -> Any:
    safe_name = Path(filename).name
    if safe_name != filename:
        return jsonify({"error": "invalid filename"}), 400
    target = OUTPUT_DIR / safe_name
    if not target.exists():
        return jsonify({"error": "file not found"}), 404
    try:
        payload = target.read_bytes()
    except Exception as exc:
        return jsonify({"error": f"download read failed: {exc}"}), 500
    try:
        target.unlink(missing_ok=True)
        _cleanup_old_render_outputs()
    except Exception:
        pass
    return send_file(
        io.BytesIO(payload),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=safe_name,
        max_age=0,
    )

@app.get("/api/phonks/<path:filename>")
def phonk_file(filename: str) -> Any:
    # Phonk tracks may live in subfolders: allow relative paths but verify the
    # resolved target stays inside PHONK_DIR (traversal/symlink guard).
    base = PHONK_DIR.resolve()
    try:
        target = (base / filename).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return jsonify({"error": "invalid path"}), 400
    except (OSError, ValueError):
        return jsonify({"error": "invalid path"}), 400
    return send_from_directory(PHONK_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=API_DEBUG)
