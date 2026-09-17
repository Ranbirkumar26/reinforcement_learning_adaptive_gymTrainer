from __future__ import annotations

from collections import Counter, defaultdict

from src.contracts import Landmark, RepFeature
from src.geometry import angle_degrees, clamp, sequence_smoothness, torso_lean_degrees
from src.profiles import UserProfile


FrameLandmarks = dict[str, tuple[float, float, float]]


def group_landmarks_by_frame(rows: list[Landmark], min_visibility: float = 0.3) -> dict[int, FrameLandmarks]:
    grouped: dict[int, FrameLandmarks] = defaultdict(dict)
    for row in rows:
        if row.visibility >= min_visibility:
            grouped[row.frame][row.landmark_name] = (row.x, row.y, row.z)
    return dict(grouped)


def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def frame_metrics(frame: FrameLandmarks) -> dict[str, float]:
    required = {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing landmarks: {sorted(missing)}")

    left_knee = angle_degrees(frame["left_hip"], frame["left_knee"], frame["left_ankle"])
    right_knee = angle_degrees(frame["right_hip"], frame["right_knee"], frame["right_ankle"])
    left_hip = angle_degrees(frame["left_shoulder"], frame["left_hip"], frame["left_knee"])
    right_hip = angle_degrees(frame["right_shoulder"], frame["right_hip"], frame["right_knee"])
    mid_hip = _midpoint(frame["left_hip"], frame["right_hip"])
    mid_shoulder = _midpoint(frame["left_shoulder"], frame["right_shoulder"])
    hip_width = max(0.001, abs(frame["right_hip"][0] - frame["left_hip"][0]))
    left_knee_offset = abs(frame["left_knee"][0] - frame["left_ankle"][0]) / hip_width
    right_knee_offset = abs(frame["right_knee"][0] - frame["right_ankle"][0]) / hip_width
    knee_asymmetry = abs(left_knee - right_knee) / 180

    return {
        "knee_angle": (left_knee + right_knee) / 2,
        "hip_angle": (left_hip + right_hip) / 2,
        "torso_lean": torso_lean_degrees((mid_hip[0], mid_hip[1]), (mid_shoulder[0], mid_shoulder[1])),
        "knee_tracking_proxy": max(left_knee_offset, right_knee_offset),
        "knee_asymmetry": knee_asymmetry,
    }


def metric_series(grouped: dict[int, FrameLandmarks]) -> list[dict[str, float]]:
    series: list[dict[str, float]] = []
    for frame_id in sorted(grouped):
        try:
            metrics = frame_metrics(grouped[frame_id])
        except ValueError:
            continue
        metrics["frame"] = float(frame_id)
        series.append(metrics)
    return series


def segment_reps(knee_angles: list[tuple[int, float]], fps: int = 30) -> list[tuple[int, int, int]]:
    if len(knee_angles) < 3:
        return []
    values = [angle for _, angle in knee_angles]
    low = min(values) + 0.38 * (max(values) - min(values))
    high = min(values) + 0.72 * (max(values) - min(values))
    min_rep_frames = max(8, int(fps * 0.45))

    reps: list[tuple[int, int, int]] = []
    state = "standing"
    start_frame = knee_angles[0][0]
    bottom_frame = knee_angles[0][0]
    last_end = -min_rep_frames

    for frame, angle in knee_angles:
        if state == "standing" and angle <= low and frame - last_end >= min_rep_frames:
            state = "bottom"
            bottom_frame = frame
        elif state == "bottom" and angle < dict(knee_angles).get(bottom_frame, angle):
            bottom_frame = frame
        elif state == "bottom" and angle >= high:
            if frame - start_frame >= min_rep_frames:
                reps.append((start_frame, bottom_frame, frame))
                last_end = frame
            start_frame = frame
            state = "standing"
        elif state == "standing":
            start_frame = frame
    return reps


def _label_mistake(
    knee_angle_min: float,
    torso_lean_max: float,
    knee_tracking_proxy: float,
    smoothness: float,
    profile: UserProfile,
) -> str:
    baseline_knee = profile.baseline_angles.get("knee_min", 85)
    baseline_torso = profile.baseline_angles.get("torso_lean_max", 28)
    if knee_tracking_proxy > 0.35:
        return "knee_tracking"
    if torso_lean_max > baseline_torso + 12:
        return "forward_lean"
    if knee_angle_min > baseline_knee + 18:
        return "shallow_squat"
    if smoothness > 4.0:
        return "unstable_motion"
    return "none"


def _fatigue_score(rep_index: int, tempos: list[float], ranges: list[float], smoothness: float) -> float:
    baseline_tempo = max(0.1, sum(tempos[: max(1, min(2, len(tempos)))]) / max(1, min(2, len(tempos))))
    tempo_drift = max(0.0, tempos[rep_index] - baseline_tempo) / baseline_tempo
    baseline_range = max(1.0, max(ranges[: max(1, min(2, len(ranges)))]))
    range_loss = max(0.0, baseline_range - ranges[rep_index]) / baseline_range
    jitter = min(1.0, smoothness / 8.0)
    return clamp(0.45 * tempo_drift + 0.35 * range_loss + 0.20 * jitter, 0.0, 1.0)


def _injury_risk(
    torso_lean_max: float,
    knee_tracking_proxy: float,
    asymmetry: float,
    profile: UserProfile,
) -> float:
    torso_base = profile.baseline_angles.get("torso_lean_max", 28)
    torso_risk = max(0.0, torso_lean_max - torso_base) / 35
    knee_risk = max(0.0, knee_tracking_proxy - 0.18) / 0.35
    asymmetry_risk = asymmetry / 0.25
    return clamp(0.45 * torso_risk + 0.40 * knee_risk + 0.15 * asymmetry_risk, 0.0, 1.0)


def extract_rep_features(rows: list[Landmark], profile: UserProfile, fps: int = 30) -> list[RepFeature]:
    grouped = group_landmarks_by_frame(rows)
    series = metric_series(grouped)
    if not series:
        return []
    knee_series = [(int(item["frame"]), item["knee_angle"]) for item in series]
    reps = segment_reps(knee_series, fps=fps)
    if not reps:
        first = int(series[0]["frame"])
        last = int(series[-1]["frame"])
        bottom = min(series, key=lambda item: item["knee_angle"])
        reps = [(first, int(bottom["frame"]), last)]

    by_frame = {int(item["frame"]): item for item in series}
    features_without_scores: list[dict[str, float | int | str]] = []
    tempos: list[float] = []
    ranges: list[float] = []

    for rep_id, (start, bottom, end) in enumerate(reps, start=1):
        rep_metrics = [item for item in series if start <= int(item["frame"]) <= end]
        if not rep_metrics:
            continue
        knee_values = [item["knee_angle"] for item in rep_metrics]
        hip_values = [item["hip_angle"] for item in rep_metrics]
        torso_values = [item["torso_lean"] for item in rep_metrics]
        knee_tracking = max(item["knee_tracking_proxy"] for item in rep_metrics)
        asymmetry = max(item["knee_asymmetry"] for item in rep_metrics)
        smoothness = sequence_smoothness(knee_values)
        tempo = max(0.1, (end - start) / fps)
        depth_proxy = clamp((180 - min(knee_values)) / 110, 0.0, 1.0)
        tempos.append(tempo)
        ranges.append(max(knee_values) - min(knee_values))
        features_without_scores.append(
            {
                "rep_id": rep_id,
                "start_frame": start,
                "end_frame": end,
                "knee_angle_min": min(knee_values),
                "hip_angle_min": min(hip_values),
                "torso_lean_max": max(torso_values),
                "tempo_sec": tempo,
                "smoothness": smoothness,
                "knee_tracking_proxy": knee_tracking,
                "depth_proxy": depth_proxy,
                "evidence_frame": bottom if bottom in by_frame else int(rep_metrics[len(rep_metrics) // 2]["frame"]),
                "asymmetry": asymmetry,
            }
        )

    output: list[RepFeature] = []
    for index, raw in enumerate(features_without_scores):
        fatigue = _fatigue_score(index, tempos, ranges, float(raw["smoothness"]))
        risk = _injury_risk(
            float(raw["torso_lean_max"]),
            float(raw["knee_tracking_proxy"]),
            float(raw["asymmetry"]),
            profile,
        )
        mistake = _label_mistake(
            float(raw["knee_angle_min"]),
            float(raw["torso_lean_max"]),
            float(raw["knee_tracking_proxy"]),
            float(raw["smoothness"]),
            profile,
        )
        output.append(
            RepFeature(
                rep_id=int(raw["rep_id"]),
                start_frame=int(raw["start_frame"]),
                end_frame=int(raw["end_frame"]),
                knee_angle_min=float(raw["knee_angle_min"]),
                hip_angle_min=float(raw["hip_angle_min"]),
                torso_lean_max=float(raw["torso_lean_max"]),
                tempo_sec=float(raw["tempo_sec"]),
                smoothness=float(raw["smoothness"]),
                fatigue_score=fatigue,
                injury_risk=risk,
                mistake_label=mistake,
                knee_tracking_proxy=float(raw["knee_tracking_proxy"]),
                depth_proxy=float(raw["depth_proxy"]),
                evidence_frame=int(raw["evidence_frame"]),
            )
        )
    return output


def most_common_mistake(features: list[RepFeature]) -> str:
    labels = [rep.mistake_label for rep in features if rep.mistake_label != "none"]
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]
