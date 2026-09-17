from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.analysis import analyze_landmark_csv, analyze_video, bundled_synthetic_landmarks
from src.coaching import load_policy, make_coaching_outputs
from src.profiles import UserProfile, load_profile, save_profile


ROOT = Path(__file__).parent
PROFILE_DIR = ROOT / "data" / "profiles"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "coach_policy"


def _profile_selector() -> UserProfile:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = sorted(PROFILE_DIR.glob("*.json"))
    labels = [path.stem for path in profiles]
    selected = st.sidebar.selectbox("User profile", labels or ["default_user"])
    profile_path = PROFILE_DIR / f"{selected}.json"
    if profile_path.exists():
        return load_profile(profile_path)
    profile = UserProfile(
        user_id="default_user",
        skill_level="beginner",
        height_cm=170,
        injury_notes=[],
        baseline_angles={"knee_min": 136, "hip_min": 148, "torso_lean_max": 24},
        history={"sessions": 0, "recurring_mistakes": {}},
    )
    save_profile(profile, profile_path)
    return profile


def main() -> None:
    st.set_page_config(page_title="Adaptive RL Gym Coach", layout="wide")
    st.title("Adaptive RL Gym Coach")
    st.caption("Squat-only MVP with pose features, RL coaching, and explainable feedback.")

    profile = _profile_selector()
    st.sidebar.write(f"Skill: {profile.skill_level}")
    st.sidebar.write(f"Height: {profile.height_cm} cm")

    uploaded = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "mkv"])
    use_synthetic = st.checkbox("Use bundled synthetic squat session", value=uploaded is None)
    run = st.button("Run analysis", type="primary")

    if not run:
        st.info("Upload a squat video or use the bundled synthetic session.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if uploaded is not None and not use_synthetic:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.read())
            video_path = Path(handle.name)
        analysis = analyze_video(video_path, OUTPUT_DIR, profile)
    else:
        csv_path = bundled_synthetic_landmarks()
        analysis = analyze_landmark_csv(csv_path, OUTPUT_DIR, profile)

    policy = load_policy(MODEL_PATH)
    coaching_outputs = make_coaching_outputs(analysis.rep_features, profile, policy)

    st.subheader("Session Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected reps", len(analysis.rep_features))
    c2.metric("Avg fatigue", f"{analysis.average_fatigue:.2f}")
    c3.metric("Avg injury risk", f"{analysis.average_injury_risk:.2f}")
    c4.metric("Top issue", analysis.top_mistake or "none")

    st.subheader("Per-Rep Coaching")
    rows = [
        {
            "rep": output.rep_id,
            "action": output.action,
            "problem": output.problem,
            "reason": output.reason,
            "correction": output.correction,
            "reward": round(output.reward, 3),
        }
        for output in coaching_outputs
    ]
    st.dataframe(rows, use_container_width=True)

    st.subheader("Risk and Fatigue")
    chart_rows = [
        {
            "rep": rep.rep_id,
            "fatigue_score": rep.fatigue_score,
            "injury_risk": rep.injury_risk,
        }
        for rep in analysis.rep_features
    ]
    st.line_chart(chart_rows, x="rep", y=["fatigue_score", "injury_risk"])

    if analysis.overlay_video_path and analysis.overlay_video_path.exists():
        st.subheader("Pose Overlay")
        st.video(str(analysis.overlay_video_path))

    if analysis.evidence_frames:
        st.subheader("Evidence Frames")
        columns = st.columns(min(3, len(analysis.evidence_frames)))
        for column, frame_path in zip(columns, analysis.evidence_frames[:3], strict=False):
            column.image(str(frame_path), caption=frame_path.name)


if __name__ == "__main__":
    main()
