"""Verifies the taste filter kills effect/transition repetition.

Run:  python -m pytest "ARCHIVE 2/tests/test_taste_system.py" -v
(or)  python "ARCHIVE 2/tests/test_taste_system.py"
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "brain"))
sys.path.insert(0, ROOT)

from taste_system import TasteFilter, rhythm_durations  # noqa: E402


def _repetitive_timeline(n=10):
    """Simulates the OLD broken behavior: identical effect group +
    identical transition on every shot."""
    return [{
        "start": i * 2.0, "end": (i + 1) * 2.0,
        "intent": "impact", "energy": 0.8, "is_drop": (i % 4 == 0),
        "transition": "flash_cut",
        "effects": [{"name": "impact_flash_combo", "strength": 1.0},
                    {"name": "zoom_burst_shake", "strength": 1.0}],
    } for i in range(n)]


def test_no_identical_effect_set_three_in_a_row():
    out = TasteFilter(seed=7).filter_timeline(_repetitive_timeline())
    sets = [frozenset(e["name"] for e in shot["effects"]) for shot in out]
    for i in range(2, len(sets)):
        assert not (sets[i] == sets[i - 1] == sets[i - 2]), \
            f"effect set repeated 3x at shot {i}: {sets[i]}"


def test_no_transition_three_in_a_row():
    out = TasteFilter(seed=7).filter_timeline(_repetitive_timeline())
    trans = [shot["transition"] for shot in out]
    for i in range(2, len(trans)):
        assert not (trans[i] == trans[i - 1] == trans[i - 2]), \
            f"transition repeated 3x at shot {i}: {trans[i]}"


def test_strengths_are_adaptive_not_fixed():
    out = TasteFilter(seed=7).filter_timeline(_repetitive_timeline())
    strengths = [e["strength"] for shot in out for e in shot["effects"]]
    assert len(set(strengths)) > 1, "strengths must vary, not be fixed"
    assert all(0.0 < s <= 1.0 for s in strengths)


def test_dialogue_suppresses_heavy_effects():
    timeline = _repetitive_timeline()
    for shot in timeline:
        shot["has_speech"] = True
        shot["intent"] = "dialogue"
    out = TasteFilter(seed=7).filter_timeline(timeline)
    heavy = {"impact_flash_combo", "zoom_burst_shake", "god_of_war",
             "camera_slam", "glitch_cut", "flash_cut"}
    for shot in out:
        names = {e["name"] for e in shot["effects"]}
        assert not (names & heavy), f"heavy effect during dialogue: {names}"
        assert len(shot["effects"]) <= 1


def test_rhythm_durations_follow_bpm():
    durs = rhythm_durations([], 120.0, ["impact", "calm", "build-up"], seed=1)
    beat = 60.0 / 120.0
    assert all(abs((d / beat) - round(d / beat)) < 1e-6 for d in durs)


def test_archive_brain_applies_taste_filter_to_segment_plan():
    from brain.archive_brain import (  # noqa: E402
        DirectedEffectPlan,
        NarrativeSection,
        SceneRead,
        UnifiedRenderEngine,
    )

    engine = UnifiedRenderEngine()
    taste_filter = engine._make_taste_filter({"variation_seed": 7})
    scene = SceneRead(start=0.0, end=1.0, duration=1.0, scene_type="dialogue")
    section = NarrativeSection(type="verse", start=0.0, end=8.0, energy=0.6, intensity=0.6)
    plan = DirectedEffectPlan(
        effect="impact_flash_combo",
        animation="camera_slam",
        effect_intensity=1.0,
        transition="flash_cut",
        is_drop=True,
    )

    out = engine._apply_taste_filter(plan, taste_filter, scene, section, 0.9, 0)

    assert engine.last_render_meta["taste_filtered_segments"] == 1
    assert out.effect in {"cinematic_grade", "portrait_enhance", "atmospheric_grade"}
    assert out.effect_intensity <= 0.55


def test_archive_brain_uses_cinematic_director_timeline():
    from brain.archive_brain import (  # noqa: E402
        DirectedEffectPlan,
        NarrativeSection,
        SceneRead,
        UnifiedRenderEngine,
    )

    engine = UnifiedRenderEngine()
    assignments = [
        (
            SceneRead(start=0.0, end=1.0, duration=1.0, scene_type="action", motion=0.2, edit_value=0.5),
            NarrativeSection(type="verse", start=0.0, end=1.0, energy=0.25, intensity=0.25),
        ),
        (
            SceneRead(start=1.0, end=2.0, duration=1.0, scene_type="impact", motion=0.9, is_climax=True, edit_value=0.9),
            NarrativeSection(type="drop", start=1.0, end=2.0, energy=0.95, intensity=0.95),
        ),
    ]

    timeline = engine._build_cinematic_timeline(assignments, {"bpm": 120.0}, {"variation_seed": 11})
    assert engine.last_render_meta["cinematic_director_segments"] == 2
    assert engine.last_render_meta["narrative_director_connected"] is True
    assert engine.last_render_meta["director_explainability_log"]
    assert engine.last_render_meta["director_narrative_summary"]
    assert timeline[1]["intent"] == "impact"
    assert timeline[1]["camera"]["motion"] == "snap_zoom"

    plan = DirectedEffectPlan(effect="cinematic_grade", animation="subtle_drift", effect_intensity=0.3)
    filtered = engine._apply_taste_filter(
        plan,
        engine._make_taste_filter({"variation_seed": 11}),
        assignments[1][0],
        assignments[1][1],
        0.95,
        1,
        timeline[1],
    )
    assert engine.last_render_meta["taste_filtered_segments"] == 1
    assert filtered.transition


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
