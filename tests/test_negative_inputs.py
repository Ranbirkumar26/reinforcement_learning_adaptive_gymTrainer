from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.actions import action_name
from src.analysis import analyze_landmark_csv
from src.io_utils import read_landmarks_csv, write_landmarks_csv
from src.pose import extract_video_landmarks, validate_landmark_csv
from src.profiles import UserProfile
from src.synthetic import generate_synthetic_squat_landmarks


def test_validate_landmark_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("frame,x,y\n1,0.1,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        validate_landmark_csv(path)


def test_read_landmarks_csv_rejects_malformed_numeric_value(tmp_path: Path) -> None:
    path = tmp_path / "bad_numeric.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timestamp", "landmark_name", "x", "y", "z", "visibility"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame": "not-int",
                "timestamp": "0.0",
                "landmark_name": "left_knee",
                "x": "0.1",
                "y": "0.2",
                "z": "0.0",
                "visibility": "0.9",
            }
        )

    with pytest.raises(ValueError):
        read_landmarks_csv(path)


def test_empty_landmark_csv_returns_zero_rep_analysis(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timestamp", "landmark_name", "x", "y", "z", "visibility"],
        )
        writer.writeheader()
    profile = UserProfile(user_id="u1", height_cm=170)

    result = analyze_landmark_csv(path, tmp_path / "out", profile)

    assert result.rep_features == []


def test_missing_video_path_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Video file does not exist"):
        extract_video_landmarks(tmp_path / "missing.mp4", tmp_path / "landmarks.csv")


def test_invalid_action_id_raises() -> None:
    with pytest.raises(ValueError, match="Unknown coaching action index"):
        action_name(99)


def test_invalid_profile_height_rejected() -> None:
    with pytest.raises(ValidationError):
        UserProfile(user_id="bad", height_cm=20)


def test_one_rep_synthetic_session_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "one_rep.csv"
    write_landmarks_csv(generate_synthetic_squat_landmarks(reps=1), path)
    profile = UserProfile(user_id="u1", height_cm=170, baseline_angles={"knee_min": 136})

    result = analyze_landmark_csv(path, tmp_path / "out", profile)

    assert len(result.rep_features) == 1
