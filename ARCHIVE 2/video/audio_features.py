from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr / (float(np.max(np.abs(arr))) + 1e-6)


def analyze_audio_features(audio_path: Path, *, duration_sec: float = 45.0) -> dict[str, Any]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=duration_sec if duration_sec > 0 else None)
    if y.size == 0 or sr <= 0:
        return {"ok": False, "reason": "empty_audio"}

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
        "beat_times": [round(float(t), 3) for t in beat_times[:1500]],
        "energy_curve": [
            {"t": round(float(t), 3), "energy": round(float(e), 4), "onset": round(float(o), 4)}
            for t, e, o in zip(times[::4], energy_curve[::4], onset_n[::4])
        ][:600],
    }


def energy_at(audio_features: dict[str, Any] | None, t: float) -> float:
    curve = (audio_features or {}).get("energy_curve")
    if not isinstance(curve, list) or not curve:
        return 0.5
    best = min(curve, key=lambda item: abs(float(item.get("t", 0.0)) - float(t)))
    return float(np.clip(best.get("energy", 0.5), 0.0, 1.0))
