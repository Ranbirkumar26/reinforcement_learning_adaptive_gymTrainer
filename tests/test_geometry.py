from __future__ import annotations

from src.geometry import angle_degrees, sequence_smoothness, torso_lean_degrees


def test_angle_degrees_right_angle() -> None:
    assert round(angle_degrees((0, 1), (0, 0), (1, 0)), 1) == 90.0


def test_angle_degrees_repeated_landmark_is_safe() -> None:
    assert angle_degrees((0, 0), (0, 0), (1, 0)) == 0.0


def test_sequence_smoothness_zero_for_linear_series() -> None:
    assert sequence_smoothness([1, 2, 3, 4, 5]) == 0.0


def test_torso_lean_degrees_vertical_body() -> None:
    assert round(torso_lean_degrees((0, 1), (0, 0)), 1) == 0.0
