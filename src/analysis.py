from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.biomechanics import extract_rep_features, most_common_mistake
from src.contracts import RepFeature
from src.io_utils import read_landmarks_csv, write_landmarks_csv, write_rep_features_csv
from src.pose import extract_video_landmarks, validate_landmark_csv, write_evidence_frames
from src.profiles import UserProfile
from src.synthetic import generate_synthetic_squat_landmarks


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AnalysisResult:
    landmarks_csv: Path
    rep_features_csv: Path
    rep_features: list[RepFeature]
    overlay_video_path: Path | None
    evidence_frames: list[Path]

    @property
    def average_fatigue(self) -> float:
        if not self.rep_features:
            return 0.0
        return sum(rep.fatigue_score for rep in self.rep_features) / len(self.rep_features)

    @property
    def average_injury_risk(self) -> float:
        if not self.rep_features:
            return 0.0
        return sum(rep.injury_risk for rep in self.rep_features) / len(self.rep_features)

    @property
    def top_mistake(self) -> str:
        return most_common_mistake(self.rep_features)


def bundled_synthetic_landmarks() -> Path:
    path = ROOT / "data" / "sample_landmarks" / "synthetic_squat_landmarks.csv"
    if path.exists():
        return path
    rows = generate_synthetic_squat_landmarks()
    write_landmarks_csv(rows, path)
    return path


def analyze_landmark_csv(path: Path, output_dir: Path, profile: UserProfile, fps: int = 30) -> AnalysisResult:
    validate_landmark_csv(path)
    landmarks = read_landmarks_csv(path)
    features = extract_rep_features(landmarks, profile, fps=fps)
    session_dir = output_dir / "latest_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    landmarks_out = session_dir / "landmarks.csv"
    features_out = session_dir / "rep_features.csv"
    write_landmarks_csv(landmarks, landmarks_out)
    write_rep_features_csv(features, features_out)
    return AnalysisResult(
        landmarks_csv=landmarks_out,
        rep_features_csv=features_out,
        rep_features=features,
        overlay_video_path=None,
        evidence_frames=[],
    )


def analyze_video(video_path: Path, output_dir: Path, profile: UserProfile) -> AnalysisResult:
    session_dir = output_dir / "latest_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    landmarks_out = session_dir / "landmarks.csv"
    overlay_path = session_dir / "pose_overlay.mp4"
    landmarks = extract_video_landmarks(video_path, landmarks_out, overlay_path)
    features = extract_rep_features(landmarks, profile)
    features_out = session_dir / "rep_features.csv"
    write_rep_features_csv(features, features_out)
    evidence_source = overlay_path if overlay_path.exists() else video_path
    evidence_frames = write_evidence_frames(
        evidence_source,
        [rep.evidence_frame for rep in features[:3]],
        session_dir / "evidence_frames",
    )
    return AnalysisResult(
        landmarks_csv=landmarks_out,
        rep_features_csv=features_out,
        rep_features=features,
        overlay_video_path=overlay_path,
        evidence_frames=evidence_frames,
    )
