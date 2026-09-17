from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis import analyze_landmark_csv
from src.coaching import HeuristicPolicy, make_coaching_outputs
from src.evaluate import evaluate
from src.io_utils import write_coaching_outputs, write_landmarks_csv
from src.profiles import UserProfile
from src.synthetic import generate_synthetic_squat_landmarks


def test_synthetic_session_writes_declared_contracts(tmp_path: Path) -> None:
    profile = UserProfile(
        user_id="integration",
        height_cm=170,
        baseline_angles={"knee_min": 136, "hip_min": 148, "torso_lean_max": 24},
    )
    landmark_path = tmp_path / "input_landmarks.csv"
    write_landmarks_csv(generate_synthetic_squat_landmarks(), landmark_path)

    analysis = analyze_landmark_csv(landmark_path, tmp_path / "outputs", profile)
    coaching = make_coaching_outputs(analysis.rep_features, profile, HeuristicPolicy())
    coach_path = tmp_path / "outputs" / "latest_session" / "coach_outputs.json"
    write_coaching_outputs(coaching, coach_path)

    assert analysis.landmarks_csv.exists()
    assert analysis.rep_features_csv.exists()
    assert coach_path.exists()

    with analysis.rep_features_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analysis.rep_features)
    assert {"rep_id", "fatigue_score", "injury_risk", "mistake_label"} <= set(rows[0])

    outputs = json.loads(coach_path.read_text(encoding="utf-8"))
    assert len(outputs) == len(analysis.rep_features)
    assert {"state", "action", "reward", "problem", "reason", "correction", "evidence_frame"} <= set(outputs[0])


def test_evaluate_writes_required_outputs(tmp_path: Path) -> None:
    paths = evaluate(tmp_path)

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    with paths["metrics"].open("r", newline="", encoding="utf-8") as handle:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert rows["detected_reps"] == "5"
    assert rows["rep_count_accuracy"] == "1.0"
