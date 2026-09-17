from __future__ import annotations

import csv
import json
from pathlib import Path

from src.contracts import CoachingOutput, Landmark, RepFeature


LANDMARK_FIELDS = ("frame", "timestamp", "landmark_name", "x", "y", "z", "visibility")
REP_FIELDS = (
    "rep_id",
    "start_frame",
    "end_frame",
    "knee_angle_min",
    "hip_angle_min",
    "torso_lean_max",
    "tempo_sec",
    "smoothness",
    "fatigue_score",
    "injury_risk",
    "mistake_label",
    "knee_tracking_proxy",
    "depth_proxy",
    "evidence_frame",
)


def write_landmarks_csv(rows: list[Landmark], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LANDMARK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def read_landmarks_csv(path: Path) -> list[Landmark]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            Landmark(
                frame=int(row["frame"]),
                timestamp=float(row["timestamp"]),
                landmark_name=row["landmark_name"],
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                visibility=float(row["visibility"]),
            )
            for row in reader
        ]


def write_rep_features_csv(rows: list[RepFeature], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def write_coaching_outputs(rows: list[CoachingOutput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row.to_row() for row in rows], indent=2), encoding="utf-8")
