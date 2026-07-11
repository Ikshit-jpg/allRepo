import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "brain"))
sys.path.insert(0, ROOT)

from director import Director, DirectorBrain, EditingExperience, ShotFeatures, attach_audio_features  # noqa: E402


def test_director_outputs_hierarchical_timeline_schema():
    shots = [
        ShotFeatures(start=0.0, end=1.0, audio_energy=0.2, motion=0.1, composition_score=0.7),
        ShotFeatures(start=1.0, end=2.0, audio_energy=0.58, motion=0.35, composition_score=0.72),
        ShotFeatures(start=2.0, end=3.0, audio_energy=0.92, motion=0.8, is_drop=True, composition_score=0.8),
    ]

    timeline = Director(seed=3).plan(shots, bpm=120.0)

    assert len(timeline) == 3
    assert timeline[0]["start_sec"] == timeline[0]["start"]
    assert timeline[0]["end_sec"] == timeline[0]["end"]
    assert isinstance(timeline[0]["camera"], dict)
    assert timeline[0]["camera_move"] == timeline[0]["camera"]["motion"]
    assert "semantics" in timeline[1]
    assert "memory" in timeline[1]
    assert timeline[2]["intent"] == "impact"
    assert timeline[2]["camera"]["motion"] == "snap_zoom"
    assert timeline[2]["suggested_duration"] in {0.5, 1.0}


def test_director_memory_can_force_release_after_fatigue():
    shots = [
        ShotFeatures(start=i, end=i + 1.0, audio_energy=0.95, motion=0.85, is_drop=True)
        for i in range(5)
    ]
    shots.append(ShotFeatures(start=5.0, end=6.0, audio_energy=0.45, motion=0.15))

    timeline = Director(seed=5).plan(shots, bpm=100.0)

    assert timeline[-1]["intent"] == "release"
    assert timeline[-1]["memory"]["fatigue"] >= 0.0


def test_attach_audio_features_accepts_dict_energy_curve():
    shots = [
        ShotFeatures(start=0.0, end=1.0),
        ShotFeatures(start=1.0, end=2.0),
    ]
    beat_analysis = {
        "tempo": 120.0,
        "beat_times": [0.0, 0.5, 1.0, 1.5],
        "drops": [1.2],
        "energy_curve": [
            {"t": 0.0, "energy": 0.2},
            {"t": 0.5, "energy": 0.3},
            {"t": 1.0, "energy": 0.8},
            {"t": 1.5, "energy": 0.9},
        ],
    }

    out = attach_audio_features(shots, beat_analysis)

    assert out[0].on_beat is True
    assert out[1].is_drop is True
    assert out[1].audio_energy > out[0].audio_energy


def test_director_brain_adds_narrative_layers_and_explainability():
    shots = [
        ShotFeatures(start=0.0, end=1.0, audio_energy=0.2, motion=0.2, brightness=0.25, shot_type="wide"),
        ShotFeatures(start=1.0, end=2.0, audio_energy=0.52, motion=0.48, has_faces=True),
        ShotFeatures(start=2.0, end=3.0, audio_energy=0.64, motion=0.58, face_count=2),
        ShotFeatures(start=3.0, end=4.0, audio_energy=0.95, motion=0.9, is_drop=True, brightness=0.32),
        ShotFeatures(start=4.0, end=5.0, audio_energy=0.42, motion=0.2, brightness=0.28),
    ]

    brain = DirectorBrain(seed=9)
    timeline = brain.run(shots, bpm=120.0)

    assert brain.story_arc_planner.last_plan.climax_index == 3
    assert brain.narrative_entity_tracker.last_timeline is not None
    assert any(
        entity.entity_type in {"character", "weapon", "symbol", "location"}
        for entities in brain.narrative_entity_tracker.last_timeline.states.values()
        for entity in entities
    )
    assert brain.motif_memory.motifs
    assert timeline[2]["narrative"]["directing_decision"]["delay_climax"] is True
    assert timeline[3]["explainability_log"]
    assert "Climax" in timeline[3]["explainability_log"][0]["reason"]


def test_learning_is_explicit_and_never_changes_edits_automatically():
    shots = [
        ShotFeatures(start=0.0, end=1.0, audio_energy=0.3, motion=0.2),
        ShotFeatures(start=1.0, end=2.0, audio_energy=0.9, motion=0.8, is_drop=True),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        brain = Director(seed=11, experience_dir=tmp_dir)
        first = brain.plan(shots)
        second = brain.plan(shots)

        assert [shot["effects"] for shot in first] == [shot["effects"] for shot in second]
        session_id = first[0]["experience_session_id"]
        experience = EditingExperience(tmp_dir)
        experience.submit_feedback(session_id, 0.8, "felt intentional")
        assert experience.learn({}, min_sessions=5) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
