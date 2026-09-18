from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from src.contracts import Landmark
from src.io_utils import write_landmarks_csv


MEDIAPIPE_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class PoseDependencyError(RuntimeError):
    pass


def _make_browser_safe_mp4(video_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not video_path.exists():
        return False

    converted = video_path.with_name(f"{video_path.stem}.browser.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(converted),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        converted.unlink(missing_ok=True)
        return False

    if converted.exists() and converted.stat().st_size > 0:
        converted.replace(video_path)
        return True
    return False


def extract_video_landmarks(video_path: Path, output_csv: Path, overlay_path: Path | None = None) -> list[Landmark]:
    if not video_path.exists():
        raise ValueError(f"Video file does not exist: {video_path}")

    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PoseDependencyError(
            "Video pose extraction requires opencv-python and mediapipe. Install requirements.txt."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    if overlay_path is not None:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    drawing = mp.solutions.drawing_utils
    rows: list[Landmark] = []
    frame_id = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            for idx, name in MEDIAPIPE_NAMES.items():
                landmark = result.pose_landmarks.landmark[idx]
                rows.append(
                    Landmark(
                        frame=frame_id,
                        timestamp=frame_id / fps,
                        landmark_name=name,
                        x=float(landmark.x),
                        y=float(landmark.y),
                        z=float(landmark.z),
                        visibility=float(landmark.visibility),
                    )
                )
            if writer is not None:
                drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        if writer is not None:
            writer.write(frame)
        frame_id += 1

    pose.close()
    capture.release()
    if writer is not None:
        writer.release()
    if overlay_path is not None:
        _make_browser_safe_mp4(overlay_path)
    write_landmarks_csv(rows, output_csv)
    return rows


def write_evidence_frames(video_path: Path, frame_ids: list[int], output_dir: Path) -> list[Path]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    wanted = set(frame_ids)
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id in wanted:
            path = output_dir / f"evidence_frame_{frame_id}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
        frame_id += 1
    capture.release()
    return paths


def validate_landmark_csv(path: Path) -> None:
    required = {"frame", "timestamp", "landmark_name", "x", "y", "z", "visibility"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Landmark CSV missing columns: {sorted(missing)}")
