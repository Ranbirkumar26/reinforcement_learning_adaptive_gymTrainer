from __future__ import annotations

import html
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


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --coach-accent: #ff4b4b;
            --coach-accent-soft: rgba(255, 75, 75, 0.14);
            --coach-panel: rgba(127, 127, 127, 0.08);
            --coach-border: rgba(127, 127, 127, 0.22);
            --coach-muted: rgba(127, 127, 127, 0.82);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--coach-border);
        }
        h1 {
            letter-spacing: 0;
            line-height: 1.08;
            margin-bottom: 0.25rem;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 8px;
            border: 1px solid var(--coach-border);
            background: var(--coach-panel);
        }
        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.6rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--coach-border);
        }
        .coach-hero {
            border-bottom: 1px solid var(--coach-border);
            padding-top: 0.15rem;
            padding-bottom: 1.15rem;
            margin-bottom: 1.35rem;
        }
        .coach-eyebrow {
            color: var(--coach-accent);
            display: block;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            line-height: 1.5;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .coach-subtitle {
            color: var(--coach-muted);
            max-width: 780px;
            line-height: 1.5;
        }
        .coach-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }
        .coach-status-item,
        .coach-metric-card,
        .coach-panel {
            border: 1px solid var(--coach-border);
            border-radius: 8px;
            background: var(--coach-panel);
        }
        .coach-status-item {
            padding: 0.78rem 0.9rem;
        }
        .coach-status-label,
        .coach-metric-label {
            color: var(--coach-muted);
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }
        .coach-status-value {
            font-size: 0.92rem;
            font-weight: 800;
        }
        .coach-status-item.active {
            border-color: var(--coach-accent);
            background: var(--coach-accent-soft);
        }
        .coach-metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1.3rem;
        }
        .coach-metric-card {
            min-height: 92px;
            padding: 0.85rem 0.95rem;
        }
        .coach-metric-card.warn {
            border-color: rgba(255, 75, 75, 0.48);
            background: var(--coach-accent-soft);
        }
        .coach-mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 0.85rem;
        }
        .coach-metric-value {
            font-size: 1.35rem;
            font-weight: 850;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .coach-panel {
            padding: 1rem 1rem 0.75rem;
            margin-bottom: 1rem;
        }
        .coach-panel-title {
            font-weight: 850;
            margin-bottom: 0.2rem;
        }
        .coach-panel-caption {
            color: var(--coach-muted);
            font-size: 0.82rem;
            margin-bottom: 0.85rem;
        }
        .coach-trainer-note {
            border-left: 4px solid var(--coach-accent);
            padding: 0.75rem 0 0.75rem 0.95rem;
            margin-bottom: 1rem;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .coach-chip {
            display: inline-block;
            border: 1px solid var(--coach-border);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            margin: 0 0.35rem 0.35rem 0;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .coach-chip.warn {
            border-color: rgba(255, 75, 75, 0.58);
            color: var(--coach-accent);
            background: var(--coach-accent-soft);
        }
        .coach-code-line {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .coach-status-grid,
            .coach-metric-grid,
            .coach-mini-grid {
                grid-template-columns: 1fr;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    st.session_state.setdefault("analysis_complete", False)
    st.session_state.setdefault("analysis_just_completed", False)
    st.session_state.setdefault("results_visible", False)
    st.session_state.setdefault("analysis_bundle", None)
    st.session_state.setdefault("input_key", "")


def _reset_if_input_changed(input_key: str) -> None:
    if st.session_state.input_key != input_key:
        st.session_state.analysis_complete = False
        st.session_state.analysis_just_completed = False
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


def _metric_card(label: str, value: str | int, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        f'<div class="coach-metric-card{tone_class}">'
        f'<div class="coach-metric-label">{html.escape(label)}</div>'
        f'<div class="coach-metric-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def _status_strip(input_ready: bool, analysis_complete: bool, results_visible: bool) -> None:
    st.markdown(
        f"""
        <div class="coach-status-grid">
          <div class="coach-status-item {'active' if input_ready else ''}">
            <div class="coach-status-label">Client and video</div>
            <div class="coach-status-value">{'Ready' if input_ready else 'Needed'}</div>
          </div>
          <div class="coach-status-item {'active' if analysis_complete else ''}">
            <div class="coach-status-label">Analysis</div>
            <div class="coach-status-value">{'Complete' if analysis_complete else 'Waiting'}</div>
          </div>
          <div class="coach-status-item {'active' if results_visible else ''}">
            <div class="coach-status-label">Results</div>
            <div class="coach-status-value">{'Visible' if results_visible else 'Locked'}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_panel(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="coach-panel">
          <div class="coach-panel-title">{html.escape(title)}</div>
          <div class="coach-panel-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_user_side(analysis: AnalysisResult, coaching_outputs: list[CoachingOutput]) -> None:
    summary = build_user_summary(analysis.rep_features, coaching_outputs)
    st.markdown('<div class="coach-panel-title">User side</div>', unsafe_allow_html=True)
    st.caption("Gym-trainer explanation")
    st.markdown(f'<div class="coach-trainer-note">{html.escape(summary.trainer_message)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="coach-mini-grid">'
        + "".join(
            [
                _metric_card("Form", summary.form_label),
                _metric_card("Fatigue", summary.fatigue_label),
                _metric_card("Injury risk", summary.injury_risk_label),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="coach-chip warn">Main issue: {html.escape(summary.main_issue.replace("_", " "))}</span>', unsafe_allow_html=True)
    if summary.rest_reps:
        st.markdown(
            f'<span class="coach-chip">Rest reps: {html.escape(", ".join(str(rep_id) for rep_id in summary.rest_reps))}</span>',
            unsafe_allow_html=True,
        )

    trainer_rows = [
        {
            "rep": output.rep_id,
            "issue": output.problem,
            "trainer cue": output.correction,
        }
        for output in coaching_outputs
    ]
    st.dataframe(trainer_rows, use_container_width=True, hide_index=True)


def _render_technical_side(bundle: dict[str, object]) -> None:
    profile = bundle["profile"]
    analysis = bundle["analysis"]
    coaching_outputs = bundle["coaching_outputs"]
    technical = bundle["technical"]
    assert isinstance(profile, UserProfile)
    assert isinstance(analysis, AnalysisResult)
    assert isinstance(coaching_outputs, list)
    assert isinstance(technical, list)

    st.markdown('<div class="coach-panel-title">Technical side</div>', unsafe_allow_html=True)
    st.caption("Pose features, thresholds, policy trace")
    st.markdown(
        """
        <div class="coach-code-line">MediaPipe landmarks -> knee-angle rep segmentation -> biomechanics features -> DQN action -> safety policy -> explanation</div>
        """,
        unsafe_allow_html=True,
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


def _render_score_chart(analysis: AnalysisResult) -> None:
    if not analysis.rep_features:
        st.info("No rep-level scores available for charting.")
        return

    import matplotlib.pyplot as plt

    reps = [rep.rep_id for rep in analysis.rep_features]
    fatigue = [rep.fatigue_score for rep in analysis.rep_features]
    risk = [rep.injury_risk for rep in analysis.rep_features]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(reps, fatigue, marker="o", linewidth=2.2, color="#2f7dd1", label="Fatigue")
    ax.plot(reps, risk, marker="o", linewidth=2.2, color="#ff4b4b", label="Injury risk")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rep")
    ax.set_ylabel("Score")
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_results(bundle: dict[str, object]) -> None:
    analysis = bundle["analysis"]
    coaching_outputs = bundle["coaching_outputs"]
    assert isinstance(analysis, AnalysisResult)
    assert isinstance(coaching_outputs, list)
    summary = build_user_summary(analysis.rep_features, coaching_outputs)

    st.subheader("Session summary")
    st.markdown(
        '<div class="coach-metric-grid">'
        + _metric_card("Detected reps", summary.detected_reps)
        + _metric_card("Bad-form reps", summary.bad_form_reps, "warn")
        + _metric_card("Fatigue", summary.fatigue_label)
        + _metric_card("Injury risk", summary.injury_risk_label, "warn" if summary.injury_risk_label == "high" else "")
        + _metric_card("Main issue", summary.main_issue.replace("_", " "))
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        _render_user_side(analysis, coaching_outputs)
    with right:
        _render_technical_side(bundle)

    st.subheader("Charts")
    _render_score_chart(analysis)

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
    _apply_theme()
    _init_state()
    st.markdown(
        """
        <div class="coach-hero">
          <div class="coach-eyebrow">Adaptive RL Gym Coach</div>
          <h1>Squat analysis dashboard</h1>
          <div class="coach-subtitle">Client-ready form feedback with technical traceability for every coaching decision.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Client")
        name = st.text_input("Name", value="Ranbir", placeholder="Enter client name")
        skill_level = st.selectbox("Skill level", ["beginner", "intermediate", "advanced"])
        height_cm = st.number_input("Height in cm", min_value=91.0, max_value=229.0, value=170.0, step=1.0)
        injury_notes = st.text_area("Injury notes", placeholder="One note per line")

    _section_panel("Input video", "Supported formats: MP4, MOV, AVI, MKV.")
    uploaded = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "mkv"])
    use_sample = st.checkbox("Use bundled real sample video", value=uploaded is None)

    input_key = f"{slugify_user_name(name)}|{skill_level}|{height_cm}|{uploaded.name if uploaded else 'sample' if use_sample else 'none'}"
    _reset_if_input_changed(input_key)

    run_disabled = not can_run_analysis(name, uploaded is not None, use_sample and SAMPLE_VIDEO.exists())
    _status_strip(not run_disabled, st.session_state.analysis_complete, st.session_state.results_visible)
    run_column, results_column, _ = st.columns([1, 1, 4])
    with run_column:
        run = st.button("Run analysis", type="primary", disabled=run_disabled, use_container_width=True)
    with results_column:
        show_results = st.button(
            "Show results",
            disabled=not can_show_results(st.session_state.analysis_complete),
            use_container_width=True,
        )

    if run:
        try:
            profile = _create_profile(name, skill_level, height_cm, injury_notes)
            video_path = SAMPLE_VIDEO if use_sample and uploaded is None else _uploaded_video_to_temp(uploaded)
            with st.spinner("Analyzing squat video and generating coaching feedback."):
                _run_video_analysis(video_path, profile)
            st.session_state.analysis_complete = True
            st.session_state.analysis_just_completed = True
            st.session_state.results_visible = False
            st.rerun()
        except Exception as exc:
            st.session_state.analysis_complete = False
            st.session_state.analysis_just_completed = False
            st.session_state.results_visible = False
            st.session_state.analysis_bundle = None
            st.error(f"Could not analyze video: {exc}")

    if show_results:
        st.session_state.results_visible = True
        st.session_state.analysis_just_completed = False
        st.rerun()

    if not st.session_state.analysis_complete:
        st.info("Enter client details, upload a squat video or use the bundled sample, then run analysis.")
        return

    if not st.session_state.results_visible:
        if st.session_state.analysis_just_completed:
            st.success("Analysis complete. Show results is now enabled.")
        st.info("Analysis is ready. Click Show results to view user and technical explanations.")
        return

    _render_results(st.session_state.analysis_bundle)


if __name__ == "__main__":
    main()
