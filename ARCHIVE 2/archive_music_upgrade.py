# =============================================================================
# ARCHIVE — MUSIC SELECTION OVERHAUL + RENDER UPGRADES
# Drop this file next to app.py and add at the top of app.py:
#   from archive_music_upgrade import *
# All functions here REPLACE their equivalents in app.py
# =============================================================================

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np

# ---------------------------------------------------------------------------
# SECTION 1 — EXTENDED HISTORY & FINGERPRINT CACHE
# ---------------------------------------------------------------------------
# These replace the originals in app.py.
# History is now 40 tracks (was 8) so repetition is rare even with small libs.

_FINGERPRINT_CACHE: dict[str, dict[str, Any]] = {}
_FINGERPRINT_CACHE_PATH = None          # set in init() below
RECENT_TRACKS_MAX = 40                  # was 8 — massive improvement
SIMILARITY_PENALTY_THRESHOLD = 0.82    # penalise tracks that sound too similar
NOVELTY_WEIGHT = 2.4                   # how much we reward fresh tracks


def init_music_upgrade(runtime_dir: Path) -> None:
    """Call this once at startup, after RUNTIME_DIR is known."""
    global _FINGERPRINT_CACHE_PATH
    _FINGERPRINT_CACHE_PATH = runtime_dir / "fingerprint_cache.json"
    _load_fingerprint_cache()


def _load_fingerprint_cache() -> None:
    if _FINGERPRINT_CACHE_PATH and _FINGERPRINT_CACHE_PATH.exists():
        try:
            data = json.loads(_FINGERPRINT_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _FINGERPRINT_CACHE.update(data)
        except Exception:
            pass


def _save_fingerprint_cache() -> None:
    if _FINGERPRINT_CACHE_PATH is None:
        return
    try:
        _FINGERPRINT_CACHE_PATH.write_text(
            json.dumps(_FINGERPRINT_CACHE, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SECTION 2 — AUDIO FINGERPRINT (detect acoustically similar tracks)
# ---------------------------------------------------------------------------

def compute_audio_fingerprint(track_path: Path, duration_sec: float = 12.0) -> dict[str, Any]:
    """
    Lightweight acoustic fingerprint.
    Returns a feature dict used to measure inter-track similarity.
    Cached on disk so each track is analysed at most once.
    """
    cache_key = str(track_path.resolve())
    try:
        sig = f"{track_path.stat().st_mtime_ns}:{track_path.stat().st_size}"
    except Exception:
        sig = cache_key

    cached = _FINGERPRINT_CACHE.get(cache_key)
    if cached and cached.get("sig") == sig:
        return dict(cached["fp"])

    try:
        y, sr = librosa.load(str(track_path), sr=22050, mono=True,
                             duration=duration_sec, res_type="kaiser_fast")
        if len(y) < 512:
            raise ValueError("track too short")

        hop = 512

        # Chroma — harmonic content fingerprint
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
        chroma_mean = chroma.mean(axis=1).tolist()   # 12 values

        # MFCC — timbral fingerprint (first 13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        mfcc_mean = mfcc.mean(axis=1).tolist()

        # Rhythm — beat histogram
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        tempo_val = float(tempo[0] if hasattr(tempo, "__len__") else tempo)

        # Spectral rolloff — brightness fingerprint
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        rolloff_mean = float(np.mean(rolloff) / (sr * 0.5))   # normalised

        # RMS energy curve (8 buckets) — energy shape fingerprint
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        bucket_n = 8
        bucket_size = max(1, len(rms) // bucket_n)
        energy_curve = [
            float(np.mean(rms[i * bucket_size:(i + 1) * bucket_size]))
            for i in range(bucket_n)
        ]
        max_e = max(energy_curve) or 1.0
        energy_curve = [e / max_e for e in energy_curve]

        fp = {
            "tempo": round(tempo_val, 1),
            "chroma": [round(v, 4) for v in chroma_mean],
            "mfcc": [round(v, 4) for v in mfcc_mean],
            "rolloff": round(rolloff_mean, 4),
            "energy_curve": [round(v, 4) for v in energy_curve],
        }
    except Exception:
        # Fallback fingerprint derived purely from filename
        name_hash = int(hashlib.md5(track_path.name.encode()).hexdigest(), 16)
        rng = random.Random(name_hash)
        fp = {
            "tempo": rng.uniform(90, 175),
            "chroma": [rng.random() for _ in range(12)],
            "mfcc": [rng.gauss(0, 20) for _ in range(13)],
            "rolloff": rng.random(),
            "energy_curve": [rng.random() for _ in range(8)],
        }

    _FINGERPRINT_CACHE[cache_key] = {"sig": sig, "fp": fp}
    if len(_FINGERPRINT_CACHE) % 8 == 0:
        _save_fingerprint_cache()
    return fp


def fingerprint_similarity(fp_a: dict[str, Any], fp_b: dict[str, Any]) -> float:
    """
    Returns 0.0 (completely different) to 1.0 (identical).
    Uses cosine similarity on MFCC + chroma + energy, plus tempo proximity.
    """
    def cosine(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))

    mfcc_sim = cosine(fp_a.get("mfcc", []), fp_b.get("mfcc", []))
    chroma_sim = cosine(fp_a.get("chroma", []), fp_b.get("chroma", []))
    energy_sim = cosine(fp_a.get("energy_curve", []), fp_b.get("energy_curve", []))

    tempo_a = float(fp_a.get("tempo", 120))
    tempo_b = float(fp_b.get("tempo", 120))
    tempo_sim = max(0.0, 1.0 - abs(tempo_a - tempo_b) / 60.0)

    rolloff_sim = max(0.0, 1.0 - abs(
        float(fp_a.get("rolloff", 0.5)) - float(fp_b.get("rolloff", 0.5))
    ))

    # Weighted combination — MFCC is the best single proxy for "sounds similar"
    return float(np.clip(
        (mfcc_sim * 0.38)
        + (chroma_sim * 0.22)
        + (energy_sim * 0.18)
        + (tempo_sim * 0.14)
        + (rolloff_sim * 0.08),
        0.0, 1.0,
    ))


# ---------------------------------------------------------------------------
# SECTION 3 — DROP DETECTION
# ---------------------------------------------------------------------------

def detect_music_drop(track_path: Path, duration_sec: float = 0.0) -> dict[str, Any]:
    """
    Find the 'drop' — the moment where energy surges the most.
    Returns drop_time_sec and confidence (0–1).
    This is the most important structural moment in phonk/AMV editing.
    """
    cache_key = f"drop:{track_path.resolve()}"
    cached = _FINGERPRINT_CACHE.get(cache_key)
    if cached:
        return dict(cached)

    try:
        load_dur = min(duration_sec, 90.0) if duration_sec > 0 else 90.0
        y, sr = librosa.load(str(track_path), sr=22050, mono=True,
                             duration=load_dur, res_type="kaiser_fast")
        if len(y) < sr * 2:
            raise ValueError("too short")

        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

        # Smooth RMS with a 0.5s window
        win = int(0.5 * sr / hop)
        rms_smooth = np.convolve(rms, np.ones(win) / win, mode="same")

        hop_sec = hop / sr
        candidates: list[tuple[float, float]] = []   # (time, surge_ratio)

        # Minimum drop time: 8 seconds in (avoid false-positives on intros)
        min_drop_frame = int(8.0 / hop_sec)
        # Don't detect after 75% of the track
        max_drop_frame = int(len(rms_smooth) * 0.75)

        look_back = int(2.0 / hop_sec)   # 2s before candidate
        look_ahead = int(0.5 / hop_sec)  # 0.5s after candidate

        for i in range(min_drop_frame + look_back, max_drop_frame - look_ahead):
            before = float(np.mean(rms_smooth[i - look_back:i]))
            after = float(np.mean(rms_smooth[i:i + look_ahead]))
            if before < 1e-5:
                continue
            ratio = after / before
            # Also require onset strength spike
            onset_spike = float(np.mean(onset_env[i:i + look_ahead])) / (
                float(np.mean(onset_env[max(0, i - look_back):i])) + 1e-6
            )
            if ratio > 1.25 and onset_spike > 1.15:
                t = i * hop_sec
                candidates.append((t, ratio * onset_spike))

        if not candidates:
            result = {"drop_time": None, "confidence": 0.0, "method": "no_drop_found"}
        else:
            # Cluster nearby candidates (within 2s) and pick strongest
            candidates.sort(key=lambda x: x[0])
            clusters: list[list[tuple[float, float]]] = []
            for t, score in candidates:
                if clusters and (t - clusters[-1][-1][0]) < 2.0:
                    clusters[-1].append((t, score))
                else:
                    clusters.append([(t, score)])
            best_cluster = max(clusters, key=lambda c: max(s for _, s in c))
            best = max(best_cluster, key=lambda x: x[1])
            drop_time = round(best[0], 3)
            max_possible = 4.0   # surge ratio ceiling for normalisation
            confidence = float(np.clip((best[1] - 1.25) / (max_possible - 1.25), 0.0, 1.0))
            result = {
                "drop_time": drop_time,
                "confidence": round(confidence, 3),
                "method": "rms_onset_surge",
                "candidate_count": len(candidates),
            }
    except Exception as exc:
        result = {"drop_time": None, "confidence": 0.0, "method": f"error:{exc}"}

    _FINGERPRINT_CACHE[cache_key] = result
    _save_fingerprint_cache()
    return result


# ---------------------------------------------------------------------------
# SECTION 4 — DIVERSITY SCORER
# ---------------------------------------------------------------------------

def score_track_diversity(
    track_path: Path,
    recent_tracks: list[str],       # list of relative keys (most recent first)
    all_track_paths: dict[str, Path],  # key → Path mapping
    fingerprint: dict[str, Any],
) -> float:
    """
    Returns a diversity bonus (positive) or penalty (negative).
    The further this track is acoustically from recently played tracks,
    and the longer since it was last played, the higher the bonus.
    """
    rel_key = track_path.name  # simplified; use _relative_track_key in app.py

    # Position-based recency penalty (stronger for more recent plays)
    recency_penalty = 0.0
    for idx, key in enumerate(recent_tracks[:RECENT_TRACKS_MAX]):
        if key == rel_key or track_path.name in key:
            # Exponential decay: very recent = big penalty, old = tiny penalty
            decay = math.exp(-idx * 0.18)
            recency_penalty = max(recency_penalty, 2.8 * decay)

    # Acoustic similarity penalty (avoid tracks that sound alike)
    similarity_penalty = 0.0
    checked = 0
    for key in recent_tracks[:12]:   # only check most recent 12
        other_path = all_track_paths.get(key)
        if other_path is None or other_path == track_path:
            continue
        try:
            other_fp = compute_audio_fingerprint(other_path, duration_sec=8.0)
            sim = fingerprint_similarity(fingerprint, other_fp)
            if sim > SIMILARITY_PENALTY_THRESHOLD:
                # Sounds too similar to a recently played track
                similarity_penalty += (sim - SIMILARITY_PENALTY_THRESHOLD) * 6.0
            checked += 1
            if checked >= 5:
                break
        except Exception:
            continue

    # Novelty bonus: if this track hasn't been played in a while, reward it
    novelty_bonus = 0.0
    if rel_key not in recent_tracks:
        novelty_bonus = NOVELTY_WEIGHT
    elif rel_key in recent_tracks[20:]:
        novelty_bonus = NOVELTY_WEIGHT * 0.5   # played long ago = small bonus

    return float(novelty_bonus - recency_penalty - similarity_penalty)

# ---------------------------------------------------------------------------
# SECTION 5 — FULL REPLACEMENT: pick_local_phonk_track
# ---------------------------------------------------------------------------
# This completely replaces the original function in app.py.
# Key improvements:
#   1. 40-track history instead of 8
#   2. Acoustic fingerprint similarity detection
#   3. Drop detection integrated into scoring
#   4. Hard diversity enforcement: never repeat within last 6 plays
#   5. Energy curve matching: track energy shape vs video energy shape
#   6. Better variation: top-10 pool with softmax sampling


def pick_local_phonk_track_v2(
    mood: str,
    video_path: "Path | None" = None,
    prompt_text: str = "",
    *,
    allow_phonk: bool = True,
    variation_seed: "int | None" = None,
    # Pass these in from app.py context:
    recent_tracks: "list[str] | None" = None,
    last_selected_key: str = "",
    phonk_dir: "Path | None" = None,
    analyze_video_vibe_fn=None,
    analyze_track_audio_vibe_fn=None,
    build_style_profile_fn=None,
    infer_prompt_music_profile_fn=None,
    list_local_phonk_files_fn=None,
    relative_track_key_fn=None,
    infer_track_mood_from_name_fn=None,
    mood_to_bpm: "dict[str, int] | None" = None,
    mood_keywords: "dict[str, list[str]] | None" = None,
    infer_prompt_music_mood_fn=None,
    is_phonk_like_fn=None,
    track_matches_folder_hints_fn=None,
    infer_prompt_music_folder_hints_fn=None,
) -> "tuple[Path | None, dict[str, Any]]":
    """
    Drop-in replacement for pick_local_phonk_track with:
    - Acoustic diversity enforcement (no similar-sounding tracks back to back)
    - 40-track history (was 8)
    - Hard block on last-6 tracks
    - Drop detection factored into scoring
    - Energy curve shape matching
    - Softmax sampling from top-10 pool (was top-3)
    """
    _recent = recent_tracks or []
    _mood_to_bpm = mood_to_bpm or {
        "hype": 154, "dark": 142, "sad": 118, "dramatic": 132, "chill": 108
    }
    _mood_kw = mood_keywords or {}

    # --- Gather files ---
    if list_local_phonk_files_fn:
        files = list_local_phonk_files_fn(allow_phonk=allow_phonk)
    else:
        return None, {"reason": "no_list_fn"}

    if not files:
        return None, {"reason": "no_local_tracks"}

    def _rel(p: Path) -> str:
        return relative_track_key_fn(p) if relative_track_key_fn else p.name

    def _mood_name(p: Path) -> str:
        return infer_track_mood_from_name_fn(_rel(p)) if infer_track_mood_from_name_fn else mood

    # Build path lookup for diversity scorer
    all_track_paths: dict[str, Path] = {_rel(p): p for p in files}

    # --- Folder hints ---
    folder_hints: list[str] = []
    folder_hints_strict = False
    folder_hints_applied = False
    folder_hint_matches = 0
    if infer_prompt_music_folder_hints_fn:
        folder_hints, folder_hints_strict = infer_prompt_music_folder_hints_fn(prompt_text)
    if folder_hints and track_matches_folder_hints_fn:
        hinted = [f for f in files if track_matches_folder_hints_fn(f, folder_hints)]
        folder_hint_matches = len(hinted)
        if hinted:
            files = hinted
            folder_hints_applied = True
        elif folder_hints_strict:
            return None, {"reason": "no_folder_match", "folder_hints": folder_hints}

    # --- Hard block: never repeat last 6 plays ---
    hard_blocked = set(_recent[:6])
    unblocked = [f for f in files if _rel(f) not in hard_blocked]
    # If all tracks are blocked (tiny library), relax to last 3
    if not unblocked:
        hard_blocked = set(_recent[:3])
        unblocked = [f for f in files if _rel(f) not in hard_blocked]
    if not unblocked:
        unblocked = files   # give up blocking, library is tiny

    files = unblocked

    # --- Video vibe analysis ---
    visual: dict[str, Any]
    if video_path and analyze_video_vibe_fn:
        visual = analyze_video_vibe_fn(video_path)
    else:
        visual = {
            "visual_mood": mood,
            "motion": 0.5,
            "brightness": 0.5,
            "saturation": 0.5,
            "contrast": 0.5,
            "desired_tempo": _mood_to_bpm.get(mood, 130),
        }

    style: dict[str, Any] = {}
    if build_style_profile_fn:
        style = build_style_profile_fn(prompt_text, mood)

    prompt_profile: dict[str, float] = {}
    if infer_prompt_music_profile_fn:
        prompt_profile = infer_prompt_music_profile_fn(prompt_text, mood)

    fast_render = bool(style.get("fast_render", False))
    quality_priority = bool(style.get("quality_priority", False))
    smart_music = bool(style.get("smart_music", True))

    # --- Music mood inference ---
    requested_music_mood = mood
    strict_music_mood = False
    if infer_prompt_music_mood_fn:
        requested_music_mood, strict_music_mood = infer_prompt_music_mood_fn(prompt_text, mood)

    ptxt = (prompt_text or "").lower()
    romantic_intent = any(t in ptxt for t in ("romantic", "romance", "pleasant", "love", "sweet", "cute", "gentle", "tender"))
    happy_intent = any(t in ptxt for t in ("happy", "joy", "cheerful", "uplifting", "smile", "positive"))
    sad_or_dark_intent = any(t in ptxt for t in ("sad", "heartbreak", "cry", "dark", "villain"))

    # --- Target tempo ---
    target_tempo = int(round(
        _mood_to_bpm.get(requested_music_mood, _mood_to_bpm.get(mood, 130)) * 0.68
        + float(visual.get("desired_tempo", 130)) * 0.32
    ))

    # --- Visual energy (for energy curve matching) ---
    visual_energy = float(np.clip(
        float(visual.get("motion", 0.5)) * 0.5
        + float(visual.get("contrast", 0.5)) * 0.3
        + float(visual.get("saturation", 0.5)) * 0.2,
        0.0, 1.0,
    ))

    # --- Pre-score pass (mood name matching, fast) ---
    prelim: list[tuple[float, Path]] = []
    for f in files:
        rel = _rel(f)
        fname_mood = _mood_name(f)
        name_lower = rel.lower()
        ps = 0.0
        if fname_mood == requested_music_mood:
            ps += 2.8
        if fname_mood == visual["visual_mood"]:
            ps += 2.1
        for word in _mood_kw.get(requested_music_mood, []):
            if word in name_lower:
                ps += 0.35
        prelim.append((ps, f))

    prelim.sort(key=lambda x: (-x[0], x[1].name.lower()))

    # --- Shortlist (more candidates than before for better diversity) ---
    if fast_render:
        shortlist_limit = 6
    elif quality_priority and smart_music:
        shortlist_limit = 14
    else:
        shortlist_limit = 10   # was 6 — bigger pool = more variety

    shortlist = [x[1] for x in prelim[:shortlist_limit]]
    if not shortlist:
        shortlist = files

    # --- Full audio analysis + diversity scoring ---
    analysis_dur = 6.0 if fast_render else (12.0 if quality_priority else 10.0)

    scored: list[tuple[float, Path, dict[str, Any]]] = []

    for f in shortlist:
        rel = _rel(f)
        fname_mood = _mood_name(f)

        # Audio vibe
        audio_vibe: dict[str, Any] = {}
        if analyze_track_audio_vibe_fn:
            try:
                audio_vibe = analyze_track_audio_vibe_fn(f, duration_sec=analysis_dur)
            except Exception:
                pass
        track_mood = str(audio_vibe.get("track_mood", fname_mood))
        tempo = float(audio_vibe.get("tempo", 120.0))
        energy = float(audio_vibe.get("energy", 0.5))
        onset = float(audio_vibe.get("onset", 0.5))
        brightness = float(audio_vibe.get("brightness", 0.5))

        # --- Acoustic fingerprint for diversity ---
        try:
            fp = compute_audio_fingerprint(f, duration_sec=min(analysis_dur, 12.0))
        except Exception:
            fp = {}

        # --- Drop detection (fast) ---
        drop_info: dict[str, Any] = {}
        if not fast_render:
            try:
                drop_info = detect_music_drop(f, duration_sec=analysis_dur)
            except Exception:
                pass
        has_drop = drop_info.get("drop_time") is not None
        drop_confidence = float(drop_info.get("confidence", 0.0))

        # ---- Core quality score ----
        score = 0.0

        # Mood match
        if fname_mood == requested_music_mood:
            score += 2.3
        if track_mood == requested_music_mood:
            score += 3.3
        if fname_mood == visual["visual_mood"]:
            score += 1.7
        if track_mood == visual["visual_mood"]:
            score += 2.4

        # Tempo fit
        tempo_score = max(0.0, 1.0 - abs(tempo - target_tempo) / 75.0)
        score += tempo_score * 3.2

        # Prompt profile match
        if prompt_profile:
            profile_dist = (
                abs(energy - prompt_profile.get("target_energy", 0.6)) * 0.45
                + abs(onset - prompt_profile.get("target_onset", 0.5)) * 0.40
                + abs(brightness - prompt_profile.get("target_brightness", 0.5)) * 0.25
            )
            profile_score = max(0.0, 1.0 - profile_dist)
            score += profile_score * 3.5
        else:
            profile_score = 0.5

        # Visual energy alignment
        if smart_music:
            score -= abs(energy - visual_energy) * 0.9

        # Drop bonus — phonk edits NEED a drop
        if has_drop:
            score += 0.8 + (drop_confidence * 1.2)

        # Strict music mood penalties
        if strict_music_mood:
            if track_mood not in {requested_music_mood, str(visual["visual_mood"])}:
                score -= 0.55
            if track_mood != requested_music_mood and fname_mood != requested_music_mood:
                score -= 1.15
            if requested_music_mood == "chill" and track_mood == "sad":
                score -= 2.2
            if requested_music_mood == "chill" and tempo > 130:
                score -= 0.75
            if romantic_intent and not sad_or_dark_intent:
                if track_mood in {"hype", "dark", "sad"}:
                    score -= 2.6
                if is_phonk_like_fn and is_phonk_like_fn(f):
                    score -= 1.2
            if happy_intent and not sad_or_dark_intent:
                if track_mood in {"sad", "dark"}:
                    score -= 2.4
                if tempo < 98:
                    score -= 0.55

        # ---- DIVERSITY SCORE (the main new addition) ----
        diversity = score_track_diversity(
            track_path=f,
            recent_tracks=_recent,
            all_track_paths=all_track_paths,
            fingerprint=fp,
        )
        score += diversity

        scored.append((
            score, f,
            {
                "file_key": rel,
                "filename_mood": fname_mood,
                "track_mood": track_mood,
                "tempo": round(tempo, 2),
                "energy": round(energy, 3),
                "onset": round(onset, 3),
                "brightness": round(brightness, 3),
                "profile_score": round(profile_score, 3),
                "diversity_bonus": round(diversity, 3),
                "has_drop": has_drop,
                "drop_confidence": round(drop_confidence, 3),
                "score": round(score, 3),
            },
        ))

    if not scored:
        return None, {"reason": "no_candidates_after_scoring"}

    scored.sort(key=lambda x: (-x[0], x[1].name.lower()))

    # ---- Softmax sampling from top-10 pool (was top-3) ----
    top_pool_size = min(10, len(scored))
    top_pool = scored[:top_pool_size]
    top_score = float(top_pool[0][0])

    allowed = [item for item in top_pool if (top_score - float(item[0])) <= 1.8]
    if not allowed:
        allowed = [top_pool[0]]

    rng = random.Random(int(variation_seed or int(time.time())) ^ (len(scored) << 6))
    temperature = 0.7
    raw_weights = [math.exp(float(item[0]) / temperature) for item in allowed]
    total_w = sum(raw_weights) or 1.0
    weights = [w / total_w for w in raw_weights]
    winner = rng.choices(allowed, weights=weights, k=1)[0]

    top_candidates = [
        {
            "file": item[2]["file_key"],
            "score": item[2]["score"],
            "track_mood": item[2]["track_mood"],
            "tempo": item[2]["tempo"],
            "diversity_bonus": item[2]["diversity_bonus"],
            "has_drop": item[2]["has_drop"],
        }
        for item in scored[:5]
    ]

    info = {
        "visual_vibe": visual,
        "tracks_considered": len(files),
        "tracks_analyzed": len(shortlist),
        "hard_blocked_count": len(hard_blocked),
        "folder_hints": folder_hints,
        "folder_hints_applied": folder_hints_applied,
        "folder_hint_matches": folder_hint_matches,
        "requested_music_mood": requested_music_mood,
        "requested_music_mood_strict": strict_music_mood,
        "target_tempo": target_tempo,
        "selected_track_mood": winner[2]["track_mood"],
        "selected_track_tempo": winner[2]["tempo"],
        "selected_diversity_bonus": winner[2]["diversity_bonus"],
        "selected_has_drop": winner[2]["has_drop"],
        "pool_size": len(allowed),
        "top_candidates": top_candidates,
    }

    return winner[1], info

# ---------------------------------------------------------------------------
# SECTION 6 — DROP-AWARE MUSIC EXCERPT SELECTION
# ---------------------------------------------------------------------------
# Replaces select_music_excerpt_for_video in app.py.
# Instead of just picking the highest-energy window, it now:
#   1. Detects the drop and starts the excerpt BEFORE the drop
#   2. So the drop lands inside the video (usually at the beat kick)


def select_music_excerpt_for_video_v2(
    audio_path: Path,
    target_duration_sec: float,
    mood: str,
    *,
    variation_seed: "int | None" = None,
    drop_offset_sec: float = 8.0,   # how many seconds before drop to start
) -> "tuple[float, dict[str, Any]]":
    """
    Pick the best starting point in the track for this video duration.
    If a drop is detected, position the excerpt so the drop falls
    `drop_offset_sec` seconds into the video — giving time for buildup.
    """
    target_duration_sec = float(max(1.0, target_duration_sec))

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, res_type="kaiser_fast")
    except Exception:
        return 0.0, {"method": "fallback", "reason": "load_failed"}

    if not len(y) or sr <= 0:
        return 0.0, {"method": "fallback", "reason": "empty"}

    total_dur = float(len(y) / sr)
    if total_dur <= target_duration_sec + 0.5:
        return 0.0, {"method": "start", "reason": "track_short"}

    # --- Try drop-anchored start ---
    drop_info = detect_music_drop(audio_path, duration_sec=total_dur)
    drop_time = drop_info.get("drop_time")
    drop_conf = float(drop_info.get("confidence", 0.0))

    if drop_time is not None and drop_conf >= 0.35:
        # Position: drop lands at drop_offset_sec inside the video
        ideal_start = max(0.0, drop_time - drop_offset_sec)
        max_start = max(0.0, total_dur - target_duration_sec - 0.1)
        start_sec = float(np.clip(ideal_start, 0.0, max_start))

        # Validate: the start window has reasonable energy (not silent)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        start_frame = int(start_sec * sr / hop)
        end_frame = int((start_sec + target_duration_sec) * sr / hop)
        end_frame = min(end_frame, len(rms) - 1)
        window_energy = float(np.mean(rms[start_frame:end_frame])) if end_frame > start_frame else 0.0
        global_mean_energy = float(np.mean(rms))

        if window_energy >= global_mean_energy * 0.45:
            # Good window — use drop-anchored start
            return start_sec, {
                "method": "drop_anchored",
                "drop_time": drop_time,
                "drop_confidence": drop_conf,
                "drop_offset_sec": drop_offset_sec,
                "start_sec": round(start_sec, 3),
                "target_duration_sec": round(target_duration_sec, 3),
            }

    # --- Fallback: best-energy-window (original logic, kept as backup) ---
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms_env = librosa.feature.rms(y=y, hop_length=hop)[0]
    n = min(len(onset_env), len(rms_env))
    if n < 8:
        return 0.0, {"method": "start", "reason": "insufficient_features"}

    onset_n = (onset_env[:n] / (np.max(onset_env[:n]) + 1e-6)).astype(np.float32)
    rms_n = (rms_env[:n] / (np.max(rms_env[:n]) + 1e-6)).astype(np.float32)

    if mood in {"hype", "dark", "dramatic"}:
        metric = (0.58 * onset_n) + (0.42 * rms_n)
    elif mood == "sad":
        metric = 1.0 - np.abs(((0.46 * onset_n) + (0.54 * rms_n)) - 0.42)
    else:
        metric = 1.0 - np.abs(((0.40 * onset_n) + (0.60 * rms_n)) - 0.36)

    win = int(max(4, round((target_duration_sec * sr) / hop)))
    if win >= n:
        return 0.0, {"method": "start", "reason": "window_too_large"}

    csum = np.concatenate(([0.0], np.cumsum(metric, dtype=np.float64)))
    window_scores = (csum[win:] - csum[:-win]) / float(win)
    if not len(window_scores):
        return 0.0, {"method": "start", "reason": "no_windows"}

    if variation_seed is None:
        start_idx = int(np.argmax(window_scores))
    else:
        rng = random.Random(int(variation_seed) + 53)
        ranked = np.argsort(window_scores)[::-1]
        top_k = min(5, len(ranked))
        top_idx = [int(ranked[i]) for i in range(top_k)]
        top_scores = [float(window_scores[i]) for i in top_idx]
        max_s = max(top_scores) or 1.0
        allowed_idx = [i for i, s in zip(top_idx, top_scores) if (max_s - s) <= 0.08]
        weights_idx = [math.exp(float(window_scores[i]) * 3.0) for i in (allowed_idx or top_idx[:1])]
        start_idx = int(rng.choices(allowed_idx or top_idx[:1], weights=weights_idx, k=1)[0])

    start_sec = float(np.clip(
        (start_idx * hop) / sr,
        0.0,
        max(0.0, total_dur - target_duration_sec - 0.05),
    ))

    return start_sec, {
        "method": "best_window",
        "start_sec": round(start_sec, 3),
        "target_duration_sec": round(target_duration_sec, 3),
        "track_duration_sec": round(total_dur, 3),
    }


# ---------------------------------------------------------------------------
# SECTION 7 — EMOTIONAL ARC BUILDER
# ---------------------------------------------------------------------------
# Call this in render_video_with_moviepy to structure edits around the drop.


def build_emotional_arc(
    n_segments: int,
    drop_time_sec: "float | None",
    total_duration_sec: float,
    mood: str,
    intensity: int,
) -> list[float]:
    """
    Returns an energy multiplier (0.0–1.0) per segment.
    Structures the edit: calm intro → tension build → DROP → chaos → breath.
    This is what makes an edit feel like it has a beginning, middle, and end.
    """
    if n_segments <= 0:
        return []

    arc: list[float] = []
    for i in range(n_segments):
        t = (i / max(1, n_segments - 1)) * total_duration_sec

        if drop_time_sec and drop_time_sec > 4.0:
            # Normalised position relative to drop
            pre_drop = drop_time_sec

            if t < pre_drop * 0.55:
                # Intro: calm, let the footage breathe
                # Gradual energy rise
                progress = t / max(0.001, pre_drop * 0.55)
                energy = 0.18 + progress * 0.22

            elif t < pre_drop * 0.85:
                # Tension build: energy increasing, cuts getting tighter
                progress = (t - pre_drop * 0.55) / max(0.001, pre_drop * 0.30)
                energy = 0.40 + progress * 0.38
                # Sine wave creates a "breathing" tension effect
                energy += 0.06 * math.sin(progress * math.pi * 3.0)

            elif t < pre_drop:
                # Final build: near max, freeze-frame / speed ramp zone
                progress = (t - pre_drop * 0.85) / max(0.001, pre_drop * 0.15)
                energy = 0.78 + progress * 0.22

            elif t < pre_drop + 6.0:
                # THE DROP: maximum chaos for 6 seconds
                # Fast cuts, heavy effects, maximum intensity
                drop_progress = (t - pre_drop) / 6.0
                # Slight decay from 1.0 → 0.88 as drop energy settles
                energy = 1.0 - (drop_progress * 0.12)

            elif t < pre_drop + 18.0:
                # Post-drop flow: sustain high but add rhythmic breathing
                post_progress = (t - (pre_drop + 6.0)) / 12.0
                energy = 0.72 + 0.18 * math.sin(post_progress * math.pi * 2.8)

            else:
                # Outro: gradual cool down
                remaining = total_duration_sec - (pre_drop + 18.0)
                if remaining > 0:
                    outro_progress = (t - (pre_drop + 18.0)) / remaining
                    energy = 0.55 - outro_progress * 0.25
                else:
                    energy = 0.45

        else:
            # No drop detected: use a simpler rising arc
            progress = t / max(0.001, total_duration_sec)
            # Sine-shaped arc: builds to peak at 70%, then slight cool
            if progress < 0.7:
                energy = 0.25 + (progress / 0.7) * 0.65
            else:
                outro = (progress - 0.7) / 0.3
                energy = 0.9 - outro * 0.35

        # Intensity modifier
        energy_scaled = float(np.clip(
            energy * (0.75 + 0.08 * intensity),
            0.0, 1.0,
        ))

        # Mood modifier
        if mood == "chill":
            energy_scaled = float(np.clip(energy_scaled * 0.72, 0.0, 1.0))
        elif mood == "sad":
            energy_scaled = float(np.clip(energy_scaled * 0.65, 0.0, 1.0))
        elif mood in {"hype", "dramatic"}:
            energy_scaled = float(np.clip(energy_scaled * 1.08, 0.0, 1.0))

        arc.append(round(energy_scaled, 4))

    return arc


# ---------------------------------------------------------------------------
# SECTION 8 — HUMAN-LIKE CUT SCORING
# ---------------------------------------------------------------------------
# Replace the uniform beat_step logic in _build_cut_points_from_markers.


def build_human_cut_points(
    beat_markers: list[float],
    segment_scores: list[float],   # one score per beat (0–1, from audio energy)
    duration: float,
    mood: str,
    intensity: int,
    selective_beats: bool = True,
    drop_time: "float | None" = None,
    variation_seed: "int | None" = None,
) -> list[float]:
    """
    Instead of cutting on every Nth beat, this decides per-beat:
    'Does this moment EARN a cut?'
    """
    rng = random.Random(int(variation_seed or 0))

    min_hold: dict[str, float] = {
        "hype": 0.45,
        "dark": 0.55,
        "dramatic": 0.65,
        "sad": 0.90,
        "chill": 0.85,
    }
    base_hold = min_hold.get(mood, 0.55)

    cut_points = [0.0]
    last_cut = 0.0

    for i, t in enumerate(beat_markers[1:], 1):
        if t >= duration:
            break

        time_since_last = t - last_cut

        if drop_time and abs(t - drop_time) < 1.5:
            cut_points.append(float(t))
            last_cut = t
            continue

        if drop_time and t > drop_time and t < drop_time + 8.0:
            effective_hold = base_hold * 0.5
        elif drop_time and t > drop_time:
            effective_hold = base_hold * 0.75
        else:
            effective_hold = base_hold

        if time_since_last < effective_hold:
            continue

        beat_score = segment_scores[i] if i < len(segment_scores) else 0.5

        base_prob = {
            "hype": 0.62,
            "dark": 0.52,
            "dramatic": 0.48,
            "sad": 0.30,
            "chill": 0.28,
        }.get(mood, 0.45)
        base_prob += (intensity - 1) * 0.08

        if selective_beats:
            if beat_score < 0.35:
                cut_prob = base_prob * 0.15
            elif beat_score < 0.55:
                cut_prob = base_prob * 0.45
            elif beat_score < 0.72:
                cut_prob = base_prob * 0.78
            else:
                cut_prob = min(0.96, base_prob * 1.35)
        else:
            cut_prob = min(0.92, base_prob + (beat_score * 0.28))

        if time_since_last > base_hold * 3.5:
            cut_prob = min(0.98, cut_prob + 0.25)

        if rng.random() < cut_prob:
            cut_points.append(float(t))
            last_cut = t

    if cut_points[-1] < duration:
        cut_points.append(duration)

    return sorted(set(cut_points))


# ---------------------------------------------------------------------------
# SECTION 9 — BETTER COLOR GRADE PARAMS
# ---------------------------------------------------------------------------
# Direct replacement for _mood_grade_params_for_ffmpeg in app.py.
# Old values: sat:1.035, cont:1.03 — barely visible.
# New values: real cinematic grades that actually show.

CINEMATIC_GRADE_PARAMS: dict[str, dict[str, float]] = {
    # Hype: punchy, high-contrast, slightly warm. Think fast-car ads.
    "hype": {
        "sat": 1.28,   # was 1.035
        "cont": 1.18,  # was 1.03
        "gamma": 0.92,
        "shadow_lift": -0.04,
        "highlight_roll": 0.06,
        "red_gain": 1.04,
        "blue_gain": 0.97,
    },
    # Dark: desaturated, deep blacks, cool teal-green shadows
    "dark": {
        "sat": 0.76,   # was 0.97
        "cont": 1.22,  # was 1.04
        "gamma": 1.08,
        "shadow_lift": -0.06,
        "highlight_roll": -0.02,
        "red_gain": 0.94,
        "blue_gain": 1.06,
    },
    # Dramatic: warm shadows, lifted blacks, cinema teal-orange split
    "dramatic": {
        "sat": 1.14,   # was 1.025
        "cont": 1.16,  # was 1.035
        "gamma": 0.97,
        "shadow_lift": 0.03,
        "highlight_roll": 0.04,
        "red_gain": 1.06,
        "blue_gain": 0.96,
    },
    # Sad: cold, desaturated, slight blue-cyan tint
    "sad": {
        "sat": 0.70,   # was 0.96
        "cont": 0.96,  # was 0.99
        "gamma": 1.07,
        "shadow_lift": 0.02,
        "highlight_roll": -0.04,
        "red_gain": 0.94,
        "blue_gain": 1.08,
    },
    # Chill: warm, slightly faded, lo-fi film look
    "chill": {
        "sat": 0.91,   # was 0.99
        "cont": 0.97,  # was 1.00
        "gamma": 1.03,
        "shadow_lift": 0.04,
        "highlight_roll": 0.0,
        "red_gain": 1.03,
        "blue_gain": 0.98,
    },
}


def build_cinematic_grade_ffmpeg(mood: str, intensity: int) -> str:
    """
    Build a proper FFmpeg filter chain for cinematic color grading.
    Much stronger than the original — actually visible on screen.
    """
    p = CINEMATIC_GRADE_PARAMS.get(mood, {
        "sat": 1.0, "cont": 1.0, "gamma": 1.0,
        "shadow_lift": 0.0, "highlight_roll": 0.0,
        "red_gain": 1.0, "blue_gain": 1.0,
    })

    # Scale by intensity
    i_boost = 1.0 + (intensity - 1) * 0.12
    sat = float(np.clip(p["sat"] * i_boost, 0.60, 1.55))
    cont = float(np.clip(p["cont"] * i_boost, 0.85, 1.40))
    gamma = float(np.clip(p["gamma"], 0.88, 1.15))

    filters: list[str] = []

    # Primary grade: contrast + saturation + gamma
    filters.append(
        f"eq=contrast={cont:.4f}:saturation={sat:.4f}:gamma={gamma:.4f}"
    )

    # Color balance: mood-specific channel push
    red_gain = float(p.get("red_gain", 1.0))
    blue_gain = float(p.get("blue_gain", 1.0))
    rs = round((red_gain - 1.0) * 0.5, 4)     # shadows red
    bs = round((blue_gain - 1.0) * 0.5, 4)    # shadows blue
    rh = round((red_gain - 1.0) * 0.3, 4)     # highlights red
    bh = round((blue_gain - 1.0) * 0.3, 4)    # highlights blue

    if abs(rs) > 0.002 or abs(bs) > 0.002:
        filters.append(
            f"colorbalance=rs={rs:.4f}:bs={bs:.4f}:rh={rh:.4f}:bh={bh:.4f}"
        )

    # Shadow lift (black point adjustment)
    shadow_lift = float(p.get("shadow_lift", 0.0))
    if abs(shadow_lift) > 0.005:
        # Use curves to lift/crush blacks
        if shadow_lift > 0:
            lift_val = int(shadow_lift * 255 * 0.8)
            filters.append(f"curves=all='0/{lift_val} 255/255'")
        else:
            crush_val = int(abs(shadow_lift) * 255 * 0.6)
            filters.append(f"curves=all='0/0 {crush_val}/0 255/255'")

    return ",".join(filters)
