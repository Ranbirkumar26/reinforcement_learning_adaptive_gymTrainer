from __future__ import annotations

from src.biomechanics import extract_rep_features, segment_reps
from src.profiles import UserProfile
from src.synthetic import generate_synthetic_squat_landmarks


def test_segment_reps_counts_synthetic_cycles() -> None:
    knees = [
        (0, 170),
        (5, 120),
        (10, 70),
        (15, 125),
        (20, 170),
        (25, 120),
        (30, 72),
        (35, 126),
        (40, 171),
    ]
    reps = segment_reps(knees, fps=10)
    assert len(reps) == 2


def test_extract_rep_features_from_synthetic_session() -> None:
    profile = UserProfile(
        user_id="test",
        skill_level="beginner",
        height_cm=170,
        baseline_angles={"knee_min": 82, "hip_min": 76, "torso_lean_max": 24},
        history={},
    )
    landmarks = generate_synthetic_squat_landmarks(reps=5, fps=30, frames_per_rep=42)
    features = extract_rep_features(landmarks, profile, fps=30)
    assert len(features) >= 4
    assert all(0 <= rep.fatigue_score <= 1 for rep in features)
    assert all(0 <= rep.injury_risk <= 1 for rep in features)
    assert features[0].rep_id == 1
