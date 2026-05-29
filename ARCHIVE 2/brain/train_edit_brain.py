from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


NUMERIC_FEATURES = [
    "intensity",
    "scene_energy",
    "motion",
    "flow_mag",
    "contrast",
    "brightness",
    "saturation",
    "scene_change",
    "face_count",
    "face_area",
    "edit_priority",
]

TARGET_COLUMNS = [
    "zoom_applied",
    "shake_applied",
    "glitch_applied",
    "glow_applied",
    "flash_applied",
    "grade_applied",
    "ken_burns_applied",
    "anime_tone_applied",
    "chroma_pulse_applied",
    "focus_zoom_applied",
]


def to_bool(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes"} else 0


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    moods = sorted({str(r.get("mood", "chill")) for r in rows})
    archetypes = sorted({str(r.get("archetype", "neutral")) for r in rows})
    mood_to_idx = {m: i for i, m in enumerate(moods)}
    arch_to_idx = {a: i for i, a in enumerate(archetypes)}

    X_num: list[list[float]] = []
    X_cat: list[list[float]] = []
    Y: list[list[int]] = []

    for row in rows:
        num_vals = [to_float(row.get(col, 0.0), 0.0) for col in NUMERIC_FEATURES]
        X_num.append(num_vals)

        cat_vec = [0.0] * (len(moods) + len(archetypes))
        mood = str(row.get("mood", "chill"))
        arch = str(row.get("archetype", "neutral"))
        if mood in mood_to_idx:
            cat_vec[mood_to_idx[mood]] = 1.0
        if arch in arch_to_idx:
            cat_vec[len(moods) + arch_to_idx[arch]] = 1.0
        X_cat.append(cat_vec)

        y_row = [to_bool(row.get(col, "False")) for col in TARGET_COLUMNS]
        Y.append(y_row)

    X = np.hstack([np.array(X_num, dtype=np.float32), np.array(X_cat, dtype=np.float32)])
    y = np.array(Y, dtype=np.int32)
    meta = {
        "numeric_features": NUMERIC_FEATURES,
        "mood_vocab": moods,
        "archetype_vocab": archetypes,
        "target_columns": TARGET_COLUMNS,
    }
    return X, y, meta


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[MultiOutputClassifier, dict[str, Any]]:
    test_size = 0.2 if len(X) >= 30 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model = MultiOutputClassifier(base)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    per_target: dict[str, Any] = {}
    f1_values: list[float] = []
    for i, col in enumerate(TARGET_COLUMNS):
        y_true = y_test[:, i]
        y_pred = pred[:, i]
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        per_target[col] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": f1,
            "positive_rate": float(np.mean(y_true)),
        }
        f1_values.append(f1)

    metrics = {
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "macro_f1": float(np.mean(f1_values) if f1_values else 0.0),
        "targets": per_target,
    }
    return model, metrics


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    csv_path = outputs_dir / "segment_labels.csv"
    model_path = outputs_dir / "edit_brain_model.joblib"
    metrics_path = outputs_dir / "edit_brain_metrics.json"

    if not csv_path.exists():
        raise SystemExit("No dataset found: outputs/segment_labels.csv")

    rows = load_rows(csv_path)
    if len(rows) < 12:
        raise SystemExit("Dataset too small. Need at least 12 labeled rows.")

    X, y, meta = build_features(rows)
    model, metrics = train_model(X, y)

    artifact = {
        "model": model,
        "meta": meta,
    }
    joblib.dump(artifact, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Samples: {metrics['samples_total']} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
