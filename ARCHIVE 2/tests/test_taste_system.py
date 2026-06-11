"""Verifies the taste filter kills effect/transition repetition.

Run:  python -m pytest "ARCHIVE 2/tests/test_taste_system.py" -v
(or)  python "ARCHIVE 2/tests/test_taste_system.py"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
