from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Landmark:
    frame: int
    timestamp: float
    landmark_name: str
    x: float
    y: float
    z: float
    visibility: float

    def to_row(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class RepFeature:
    rep_id: int
    start_frame: int
    end_frame: int
    knee_angle_min: float
    hip_angle_min: float
    torso_lean_max: float
    tempo_sec: float
    smoothness: float
    fatigue_score: float
    injury_risk: float
    mistake_label: str
    knee_tracking_proxy: float
    depth_proxy: float
    evidence_frame: int

    def to_row(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class CoachingOutput:
    rep_id: int
    state: list[float]
    action: str
    reward: float
    problem: str
    reason: str
    correction: str
    evidence_frame: int

    def to_row(self) -> dict[str, int | float | str | list[float]]:
        return asdict(self)
