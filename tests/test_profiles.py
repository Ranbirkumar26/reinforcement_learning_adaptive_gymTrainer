from __future__ import annotations

from src.profiles import UserProfile, load_profile, save_profile, update_profile_history


def test_profile_round_trip(tmp_path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile(
        user_id="u1",
        skill_level="intermediate",
        height_cm=176,
        baseline_angles={"knee_min": 85},
    )
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.user_id == "u1"
    assert loaded.skill_encoding == 0.5


def test_update_profile_history_counts_mistakes() -> None:
    profile = UserProfile(user_id="u1", height_cm=176)
    updated = update_profile_history(profile, ["knee_tracking", "none", "knee_tracking"])
    assert updated.history["sessions"] == 1
    assert updated.history["recurring_mistakes"]["knee_tracking"] == 2
