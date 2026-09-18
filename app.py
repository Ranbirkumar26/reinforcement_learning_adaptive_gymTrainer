from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.analysis import AnalysisResult, analyze_video
from src.coaching import load_policy, make_coaching_outputs
from src.contracts import CoachingOutput
from src.dashboard import (
    build_profile_from_inputs,
    build_technical_explanations,
    build_user_summary,
    can_run_analysis,
    can_show_results,
    slugify_user_name,
)
from src.io_utils import write_coaching_outputs
from src.profiles import UserProfile, save_profile


ROOT = Path(__file__).parent
PROFILE_DIR = ROOT / "data" / "profiles"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "coach_policy"
SAMPLE_VIDEO = ROOT / "data" / "sample_videos" / "real_squat_sample.mov"


def _init_state() -> None:
    st.session_state.setdefault("analysis_complete", False)
    st.session_state.setdefault("results_visible", False)
    st.session_state.setdefault("analysis_bundle", None)
    st.session_state.setdefault("input_key", "")


def _reset_if_input_changed(input_key: str) -> None:
    if st.session_state.input_key != input_key:
        st.session_state.analysis_complete = False
        st.session_state.results_visible = False
        st.session_state.analysis_bundle = None
        st.session_state.input_key = input_key


def _uploaded_video_to_temp(uploaded: object) -> Path:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.read())
        return Path(handle.name)


def _create_profile(name: str, skill_level: str, height_cm: float, injury_notes: str) -> UserProfile:
    profile = build_profile_from_inputs(name, skill_level, height_cm, injury_notes)
    save_profile(profile, PROFILE_DIR / f"{profile.user_id}.json")
    return profile


def _run_video_analysis(video_path: Path, profile: UserProfile) -> tuple[AnalysisResult, list[CoachingOutput]]:
    analysis = analyze_video(video_path, OUTPUT_DIR, profile)
    if not analysis.rep_features:
        raise ValueError("No squat reps detected. Use a clear full-body squat video and try again.")
    policy = load_policy(MODEL_PATH)
    coaching_outputs = make_coaching_outputs(analysis.rep_features, profile, policy)
    write_coaching_outputs(coaching_outputs, OUTPUT_DIR / "latest_session" / "coach_outputs.json")
    technical = build_technical_explanations(analysis.rep_features, coaching_outputs, profile, policy)
    st.session_state.analysis_bundle = {
        "profile": profile,
        "analysis": analysis,
        "coaching_outputs": coaching_outputs,
        "technical": technical,
    }
    return analysis, coaching_outputs


def _text_metric(column: object, label: str, value: str | int) -> None:
    column.caption(label)
    column.markdown(f"**{value}**")


def _render_user_side(analysis: AnalysisResult, coaching_outputs: list[CoachingOutput]) -> None:
    summary = build_user_summary(analysis.rep_features, coaching_outputs)
    st.subheader("User side")
    st.caption("Gym-trainer explanation")
    st.write(summary.trainer_message)

    c1, c2, c3 = st.columns(3)
    _text_metric(c1, "Form", summary.form_label)
    _text_metric(c2, "Fatigue", summary.fatigue_label)
    _text_metric(c3, "Injury risk", summary.injury_risk_label)

    st.write(f"Main issue: `{summary.main_issue.replace('_', ' ')}`")
    if summary.rest_reps:
        st.write(f"Rest recommended on reps: {', '.join(str(rep_id) for rep_id in summary.rest_reps)}")

    st.write("Trainer notes")
    for output in coaching_outputs:
        st.markdown(f"- Rep {output.rep_id}: {output.correction}")


def _render_technical_side(bundle: dict[str, object]) -> None:
    profile = bundle["profile"]
    analysis = bundle["analysis"]
    coaching_outputs = bundle["coaching_outputs"]
    technical = bundle["technical"]
    assert isinstance(profile, UserProfile)
    assert isinstance(analysis, AnalysisResult)
    assert isinstance(coaching_outputs, list)
    assert isinstance(technical, list)

    st.subheader("Technical side")
    st.caption("Pose features, thresholds, policy trace")
    st.markdown(
        "\n".join(
            [
                "- MediaPipe detected shoulder, hip, knee, and ankle landmarks.",
                "- Knee-angle phase changes segmented squat reps.",
                "- Per-rep features drove fatigue, injury-risk, and mistake labels.",
                "- State vector was sent to DQN when available.",
                "- Safety policy overrode unsafe output when fatigue, risk, or known mistakes were high.",
            ]
        )
    )
    st.write(
        f"Profile baseline: knee_min `{profile.baseline_angles.get('knee_min')}`, "
        f"hip_min `{profile.baseline_angles.get('hip_min')}`, "
        f"torso_lean_max `{profile.baseline_angles.get('torso_lean_max')}`"
    )

    for rep, output, explanation in zip(analysis.rep_features, coaching_outputs, technical, strict=False):
        with st.expander(f"Rep {rep.rep_id}: {output.problem}"):
            st.write(
                {
                    "knee_angle_min": round(rep.knee_angle_min, 2),
                    "hip_angle_min": round(rep.hip_angle_min, 2),
                    "torso_lean_max": round(rep.torso_lean_max, 2),
                    "tempo_sec": round(rep.tempo_sec, 2),
                    "smoothness": round(rep.smoothness, 2),
                    "fatigue_score": round(rep.fatigue_score, 3),
                    "injury_risk": round(rep.injury_risk, 3),
                    "knee_tracking_proxy": round(rep.knee_tracking_proxy, 3),
                    "mistake_label": rep.mistake_label,
                }
            )
            st.write(f"Triggered condition: `{explanation.triggered_condition}`")
            st.write(f"Raw model action: `{explanation.raw_model_action}`")
            st.write(f"Final action: `{explanation.final_action}`")
            st.write(f"Decision source: `{explanation.decision_source}`")
            st.write(f"Decision reason: {explanation.decision_reason}")
            st.write(f"Reward: `{explanation.reward:.3f}`")
            st.write(f"Evidence frame: `{explanation.evidence_frame}`")
            st.code(json.dumps([round(value, 4) for value in explanation.state_vector], indent=2), language="json")


def _render_results(bundle: dict[str, object]) -> None:
    analysis = bundle["analysis"]
    coaching_outputs = bundle["coaching_outputs"]
    assert isinstance(analysis, AnalysisResult)
    assert isinstance(coaching_outputs, list)
    summary = build_user_summary(analysis.rep_features, coaching_outputs)

    st.subheader("Session summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    _text_metric(c1, "Detected reps", summary.detected_reps)
    _text_metric(c2, "Bad-form reps", summary.bad_form_reps)
    _text_metric(c3, "Fatigue", summary.fatigue_label)
    _text_metric(c4, "Injury risk", summary.injury_risk_label)
    _text_metric(c5, "Main issue", summary.main_issue.replace("_", " "))

    left, right = st.columns(2)
    with left:
        _render_user_side(analysis, coaching_outputs)
    with right:
        _render_technical_side(bundle)

    st.subheader("Charts")
    chart_rows = [
        {"rep": rep.rep_id, "fatigue_score": rep.fatigue_score, "injury_risk": rep.injury_risk}
        for rep in analysis.rep_features
    ]
    st.line_chart(chart_rows, x="rep", y=["fatigue_score", "injury_risk"])

    st.subheader("Rep table")
    rows = [
        {
            "rep": output.rep_id,
            "issue": output.problem,
            "action": output.action,
            "fatigue": round(analysis.rep_features[index].fatigue_score, 3),
            "risk": round(analysis.rep_features[index].injury_risk, 3),
            "reward": round(output.reward, 3),
        }
        for index, output in enumerate(coaching_outputs)
    ]
    st.dataframe(rows, use_container_width=True)

    if analysis.overlay_video_path and analysis.overlay_video_path.exists():
        st.subheader("Pose overlay")
        st.video(str(analysis.overlay_video_path))

    if analysis.evidence_frames:
        st.subheader("Evidence frames")
        columns = st.columns(min(3, len(analysis.evidence_frames)))
        for column, frame_path in zip(columns, analysis.evidence_frames[:3], strict=False):
            column.image(str(frame_path), caption=frame_path.name)


def main() -> None:
    st.set_page_config(page_title="Adaptive RL Gym Coach", layout="wide")
    _init_state()
    st.title("Adaptive RL Gym Coach")
    st.caption("Squat-only dashboard with plain-language coaching and technical reasoning.")

    with st.sidebar:
        st.header("Client")
        name = st.text_input("Name", value="Ranbir", placeholder="Enter client name")
        skill_level = st.selectbox("Skill level", ["beginner", "intermediate", "advanced"])
        height_cm = st.number_input("Height in cm", min_value=91.0, max_value=229.0, value=170.0, step=1.0)
        injury_notes = st.text_area("Injury notes", placeholder="One note per line")

    st.subheader("Input video")
    uploaded = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "mkv"])
    use_sample = st.checkbox("Use bundled real sample video", value=uploaded is None)

    input_key = f"{slugify_user_name(name)}|{skill_level}|{height_cm}|{uploaded.name if uploaded else 'sample' if use_sample else 'none'}"
    _reset_if_input_changed(input_key)

    run_disabled = not can_run_analysis(name, uploaded is not None, use_sample and SAMPLE_VIDEO.exists())
    run = st.button("Run analysis", type="primary", disabled=run_disabled)

    if run:
        try:
            profile = _create_profile(name, skill_level, height_cm, injury_notes)
            video_path = SAMPLE_VIDEO if use_sample and uploaded is None else _uploaded_video_to_temp(uploaded)
            with st.spinner("Analyzing squat video and generating coaching feedback."):
                _run_video_analysis(video_path, profile)
            st.session_state.analysis_complete = True
            st.session_state.results_visible = False
            st.success("Analysis complete. Show results is now enabled.")
        except Exception as exc:
            st.session_state.analysis_complete = False
            st.session_state.results_visible = False
            st.session_state.analysis_bundle = None
            st.error(f"Could not analyze video: {exc}")

    show_results = st.button("Show results", disabled=not can_show_results(st.session_state.analysis_complete))
    if show_results:
        st.session_state.results_visible = True

    if not st.session_state.analysis_complete:
        st.info("Enter client details, upload a squat video or use the bundled sample, then run analysis.")
        return

    if not st.session_state.results_visible:
        st.info("Analysis is ready. Click Show results to view user and technical explanations.")
        return

    _render_results(st.session_state.analysis_bundle)


if __name__ == "__main__":
    main()
