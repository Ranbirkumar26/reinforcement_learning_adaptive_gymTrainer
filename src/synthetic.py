from __future__ import annotations

from math import cos, pi, sin
from random import Random

from src.contracts import Landmark


LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def generate_synthetic_squat_landmarks(
    reps: int = 5,
    fps: int = 30,
    frames_per_rep: int = 42,
    seed: int = 7,
) -> list[Landmark]:
    rng = Random(seed)
    rows: list[Landmark] = []
    total_frames = reps * frames_per_rep
    for frame in range(total_frames):
        rep_index = frame // frames_per_rep
        phase = (frame % frames_per_rep) / (frames_per_rep - 1)
        squat = (1 - cos(2 * pi * phase)) / 2
        fatigue = rep_index / max(1, reps - 1)
        jitter = 0.003 * fatigue * sin(frame * 1.7)

        hip_y = 0.48 + 0.17 * squat + jitter
        shoulder_y = 0.26 + 0.09 * squat + jitter
        knee_y = 0.68 + 0.04 * squat
        ankle_y = 0.90
        torso_forward = 0.02 + 0.05 * squat + 0.025 * fatigue
        knee_inward = 0.015 * squat + 0.015 * fatigue
        depth_loss = 0.035 * fatigue
        hip_y -= depth_loss * squat

        left_hip = (0.43 + torso_forward, hip_y, 0.0)
        right_hip = (0.57 + torso_forward, hip_y, 0.0)
        left_shoulder = (0.41 + torso_forward * 1.8, shoulder_y, 0.0)
        right_shoulder = (0.59 + torso_forward * 1.8, shoulder_y, 0.0)
        left_knee = (0.39 + knee_inward, knee_y, 0.0)
        right_knee = (0.61 - knee_inward, knee_y, 0.0)
        left_ankle = (0.37, ankle_y, 0.0)
        right_ankle = (0.63, ankle_y, 0.0)
        values = {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
        }
        for name in LANDMARK_NAMES:
            x, y, z = values[name]
            rows.append(
                Landmark(
                    frame=frame,
                    timestamp=frame / fps,
                    landmark_name=name,
                    x=x + rng.uniform(-0.001, 0.001),
                    y=y + rng.uniform(-0.001, 0.001),
                    z=z,
                    visibility=0.99,
                )
            )
    return rows
