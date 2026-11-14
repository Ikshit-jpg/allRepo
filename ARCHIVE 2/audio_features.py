from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from cache import file_cache_signature, load_json_cache, write_json_cache


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr / (float(np.max(np.abs(arr))) + 1e-6)


def analyze_audio_features(audio_path: Path, *, duration_sec: float = 45.0) -> dict[str, Any]:
    audio_path = Path(audio_path)
    try:
        signature = f"{file_cache_signature(audio_path)}:{float(duration_sec):.3f}:music-timeline-v2"
        cache_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        cache_path = Path(tempfile.gettempdir()) / "archive_music_cache" / f"{cache_key}.json"
        cached = load_json_cache(cache_path)
        if cached.get("signature") == signature and isinstance(cached.get("features"), dict):
            return cached["features"]
    except Exception:
        signature = ""
        cache_path = None

    features = _analyze_audio_features_uncached(audio_path, duration_sec=duration_sec)
    if signature and cache_path is not None and features.get("ok"):
        write_json_cache(cache_path, {"signature": signature, "features": features})
    return features


def _analyze_audio_features_uncached(audio_path: Path, *, duration_sec: float) -> dict[str, Any]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=duration_sec if duration_sec > 0 else None)
    if y.size == 0 or sr <= 0:
        return {"ok": False, "reason": "empty_audio"}

    duration = float(len(y) / sr)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    harmonic, percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).tolist()

    n = int(min(len(rms), len(onset)))
    rms_n = _norm(rms[:n])
    onset_n = _norm(onset[:n])
    energy_curve = ((0.55 * rms_n) + (0.45 * onset_n)).astype(np.float32) if n else np.array([], dtype=np.float32)
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop).astype(np.float32) if n else np.array([], dtype=np.float32)

    drops: list[float] = []
    if n >= 16:
        smooth = np.convolve(energy_curve, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        for i in range(6, n - 6):
            before = float(np.mean(smooth[max(0, i - 8):i]))
            after = float(np.mean(smooth[i:min(n, i + 8)]))
            local = float(smooth[i])
            if after > before * 1.22 and local > 0.48 and onset_n[i] > 0.42:
                t = float(times[i])
                if t >= 2.0 and all(abs(t - old) > 2.5 for old in drops):
                    drops.append(round(t, 3))
            if len(drops) >= 5:
                break

    tempo_value = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    percussive_ratio = float(np.mean(np.abs(percussive)) / (np.mean(np.abs(harmonic)) + np.mean(np.abs(percussive)) + 1e-6))
    beat_confidence = float(np.clip(len(beat_times) / max(1.0, (len(y) / sr) / max(0.28, 60.0 / max(tempo_value, 1.0))), 0.0, 1.0))
    percussion_spectrum = np.abs(librosa.stft(percussive, n_fft=2048, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    kick_envelope = np.mean(percussion_spectrum[frequencies <= 150], axis=0) if np.any(frequencies <= 150) else np.zeros(percussion_spectrum.shape[1])
    snare_band = (frequencies > 150) & (frequencies <= 5000)
    snare_envelope = np.mean(percussion_spectrum[snare_band], axis=0) if np.any(snare_band) else np.zeros(percussion_spectrum.shape[1])
    percussion_times = librosa.frames_to_time(np.arange(len(kick_envelope)), sr=sr, hop_length=hop)
    kick_times = _peak_times(kick_envelope, percussion_times, minimum_gap=0.12)
    snare_times = _peak_times(snare_envelope, percussion_times, minimum_gap=0.12)
    downbeats = [round(float(t), 3) for t in beat_times[::4]]
    sections = _build_music_sections(times, energy_curve, drops, duration)
    repeated_patterns = _find_repeated_beat_patterns(beat_times, energy_curve, times)
    music_timeline = {
        "duration": round(duration, 3),
        "bpm": round(tempo_value, 3),
        "beats": [round(float(t), 3) for t in beat_times[:1500]],
        "downbeats": downbeats[:400],
        "kicks": kick_times,
        "snares": snare_times,
        "drops": drops,
        "sections": sections,
        "repeated_patterns": repeated_patterns,
        "velocity_opportunities": sorted(set(kick_times + snare_times))[:1000],
    }
    return {
        "ok": True,
        "tempo": round(tempo_value, 3),
        "beat_confidence": round(beat_confidence, 4),
        "rms_mean": round(float(np.mean(rms)) if rms.size else 0.0, 6),
        "rms_peak": round(float(np.max(rms)) if rms.size else 0.0, 6),
        "onset_mean": round(float(np.mean(onset)) if onset.size else 0.0, 6),
        "onset_peak": round(float(np.max(onset)) if onset.size else 0.0, 6),
        "spectral_contrast": round(float(np.mean(contrast)) if contrast.size else 0.0, 6),
        "percussive_ratio": round(percussive_ratio, 4),
        "drops": drops,
        "downbeat_times": downbeats[:400],
        "kick_times": kick_times,
        "snare_times": snare_times,
        "sections": sections,
        "repeated_patterns": repeated_patterns,
        "music_timeline": music_timeline,
        "beat_times": [round(float(t), 3) for t in beat_times[:1500]],
        "energy_curve": [
            {"t": round(float(t), 3), "energy": round(float(e), 4), "onset": round(float(o), 4)}
            for t, e, o in zip(times[::4], energy_curve[::4], onset_n[::4])
        ][:600],
    }


def _peak_times(values: np.ndarray, times: np.ndarray, *, minimum_gap: float) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if values.size < 3 or times.size != values.size:
        return []
    threshold = max(float(np.quantile(values, 0.82)), float(np.mean(values) + 0.7 * np.std(values)))
    peaks = [
        idx for idx in range(1, len(values) - 1)
        if values[idx] >= threshold and values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]
    ]
    selected: list[int] = []
    for idx in peaks:
        if not selected or float(times[idx] - times[selected[-1]]) >= minimum_gap:
            selected.append(idx)
        elif values[idx] > values[selected[-1]]:
            selected[-1] = idx
    return [round(float(times[idx]), 3) for idx in selected[:1000]]


def _build_music_sections(
    times: np.ndarray,
    energy: np.ndarray,
    drops: list[float],
    duration: float,
) -> list[dict[str, Any]]:
    if not len(times) or not len(energy) or duration <= 0:
        return [{"start": 0.0, "end": max(0.0, duration), "label": "unknown", "energy": 0.5}]
    grid = np.arange(0.0, duration, 0.5)
    labels: list[tuple[float, str]] = []
    for t in grid:
        idx = int(np.argmin(np.abs(times - t)))
        current = float(energy[idx])
        before_idx = int(np.argmin(np.abs(times - max(0.0, t - 1.5))))
        before = float(energy[before_idx])
        if any(abs(float(t) - drop) <= 1.0 for drop in drops):
            label = "drop"
        elif current < 0.24:
            label = "quiet"
        elif current - before > 0.18:
            label = "build"
        elif current >= 0.68:
            label = "high_energy"
        else:
            label = "flow"
        labels.append((float(t), label))
    sections: list[dict[str, Any]] = []
    start = labels[0][0]
    active = labels[0][1]
    energies: list[float] = []
    for t, label in labels:
        idx = int(np.argmin(np.abs(times - t)))
        if label != active:
            sections.append({"start": round(start, 3), "end": round(t, 3), "label": active, "energy": round(float(np.mean(energies)) if energies else 0.5, 4)})
            start, active, energies = t, label, []
        energies.append(float(energy[idx]))
    sections.append({"start": round(start, 3), "end": round(duration, 3), "label": active, "energy": round(float(np.mean(energies)) if energies else 0.5, 4)})
    return sections


def _find_repeated_beat_patterns(beat_times: list[float], energy_curve: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    if len(beat_times) < 8 or not len(times):
        return []
    bars: list[tuple[int, np.ndarray]] = []
    for start in range(0, len(beat_times) - 3, 4):
        signature = []
        for beat in beat_times[start : start + 4]:
            idx = int(np.argmin(np.abs(times - beat)))
            signature.append(float(energy_curve[idx]))
        values = np.asarray(signature, dtype=np.float32)
        values = values - float(values.mean())
        norm = float(np.linalg.norm(values))
        bars.append((start, values / norm if norm > 1e-6 else values))
    repeats: list[dict[str, Any]] = []
    for idx, (left_start, left) in enumerate(bars):
        for right_start, right in bars[idx + 2 :]:
            similarity = float(np.dot(left, right)) if np.linalg.norm(left) and np.linalg.norm(right) else 0.0
            if similarity >= 0.92:
                repeats.append({"first_beat_index": left_start, "repeat_beat_index": right_start, "similarity": round(similarity, 4)})
                break
        if len(repeats) >= 40:
            break
    return repeats


def energy_at(audio_features: dict[str, Any] | None, t: float) -> float:
    curve = (audio_features or {}).get("energy_curve")
    if not isinstance(curve, list) or not curve:
        return 0.5
    best = min(curve, key=lambda item: abs(float(item.get("t", 0.0)) - float(t)))
    return float(np.clip(best.get("energy", 0.5), 0.0, 1.0))
