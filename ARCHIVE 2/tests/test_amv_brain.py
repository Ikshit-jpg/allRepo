import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "brain"))
sys.path.insert(0, ROOT)

from amv_brain import AMVBrain  # noqa: E402
from director import ShotFeatures  # noqa: E402
from quality_judge import QualityJudge  # noqa: E402


def _shots():
    return [
        ShotFeatures(start=0.0, end=1.0, audio_energy=0.22, motion=0.12, brightness=0.3, shot_type="wide"),
        ShotFeatures(start=1.0, end=2.0, audio_energy=0.55, motion=0.45, has_faces=True),
        ShotFeatures(start=2.0, end=3.0, audio_energy=0.68, motion=0.62, face_count=2),
        ShotFeatures(start=3.0, end=4.0, audio_energy=0.96, motion=0.88, is_drop=True, brightness=0.28),
        ShotFeatures(start=4.0, end=5.0, audio_energy=0.38, motion=0.18, has_speech=True),
    ]


def test_amv_brain_outputs_full_pipeline_metadata():
    brain = AMVBrain(seed=13)
    timeline = brain.plan(_shots(), bpm=120.0)

    assert len(timeline) == 5
    assert timeline[2]["strategy"]["payoff_state"] in {"delay", "tension"}
    assert timeline[3]["intent"] == "impact"
    assert timeline[3]["effect_budget"]["budget"] >= 0.78
    assert timeline[3]["effect_grammar"]
    assert "narrative_director" in timeline[0]["brain_pipeline"]
    assert "editorial_planner" in timeline[0]["brain_pipeline"]
    assert "revision_loop" in timeline[0]["brain_pipeline"]
    assert timeline[0]["brain_pipeline"].index("effect_budget") < timeline[0]["brain_pipeline"].index("effect_grammar")
    assert timeline[0]["brain_pipeline"].index("effect_grammar") < timeline[0]["brain_pipeline"].index("taste_filter")
    assert timeline[0]["quality_report"]["score"] >= 0.0
    assert timeline[3]["explainability_log"]
    assert any("Story climax" in log["reason"] for log in timeline[3]["explainability_log"])


def test_quality_judge_flags_bad_timeline_transparently():
    bad = [
        {
            "start": i,
            "end": i + 1,
            "intent": "build-up",
            "transition": "flash_cut",
            "effects": [{"name": "impact_flash_combo", "strength": 0.9}],
            "has_speech": i == 1,
            "energy": 0.5,
        }
        for i in range(3)
    ]

    report = QualityJudge().evaluate(bad)

    assert report.score < 1.0
    assert any(issue.code == "repeated_effect_combo" for issue in report.issues)
    assert any(issue.code == "speech_heavy_effect" for issue in report.issues)


def test_quality_judge_and_revision_loop_target_flash_overuse():
    from revision_loop import RevisionLoop  # noqa: E402

    bad = [
        {
            "start": i,
            "end": i + 1,
            "intent": "build-up",
            "transition": "flash_cut",
            "effects": [{"name": "impact_flash_combo", "strength": 0.75}],
            "has_speech": False,
            "energy": 0.65,
            "explainability_log": [],
        }
        for i in range(4)
    ]

    report = QualityJudge().evaluate(bad)
    fixed = RevisionLoop(passes=1).run(bad)

    assert any(issue.code == "too_many_flashes" for issue in report.issues)
    assert any(
        action["issue"] == "too_many_flashes"
        for shot in fixed
        for action in shot.get("revision_actions", [])
    )
    assert any(shot["transition"] == "cut" for shot in fixed)


def test_budget_guard_preserves_explicit_zero_values():
    timeline = [
        {
            "effects": [{"name": "cinematic_grade", "strength": 0.8}],
            "effect_budget": {"budget": 0.0},
            "is_drop": False,
            "intent": "calm",
            "has_speech": False,
        }
    ]

    AMVBrain._guard_taste_with_budget(timeline)

    assert timeline[0]["effects"][0]["strength"] == 0.0
    assert timeline[0]["effects"][0]["budget_guarded"] is True


def test_energy_arc_preserves_zero_energy():
    assert AMVBrain._float_value(0.0, 0.5) == 0.0
    assert AMVBrain._float_value(None, 0.5) == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
