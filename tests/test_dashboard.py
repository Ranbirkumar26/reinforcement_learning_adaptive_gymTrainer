from __future__ import annotations

from src.actions import action_id
from src.coaching import SafetyPolicy
from src.contracts import CoachingOutput, RepFeature
from src.dashboard import (
    build_profile_from_inputs,
    build_technical_explanations,
    build_user_summary,
    can_run_analysis,
    can_show_results,
    form_summary_label,
    score_label,
    slugify_user_name,
    triggered_condition,
)
from src.profiles import UserProfile


def _rep(rep_id: int, mistake: str = "none", fatigue: float = 0.2, risk: float = 0.2) -> RepFeature:
    return RepFeature(
        rep_id=rep_id,
        start_frame=0,
        end_frame=30,
        knee_angle_min=92,
        hip_angle_min=80,
        torso_lean_max=22,
        tempo_sec=1.0,
        smoothness=1.0,
        fatigue_score=fatigue,
        injury_risk=risk,
        mistake_label=mistake,
        knee_tracking_proxy=0.42 if mistake == "knee_tracking" else 0.1,
        depth_proxy=0.8,
        evidence_frame=12,
    )


def _output(rep_id: int, action: str = "joint_highlight") -> CoachingOutput:
    return CoachingOutput(
        rep_id=rep_id,
        state=[0.0] * 9,
        action=action,
        reward=-0.1,
        problem="Knee tracking issue",
        reason="Knee tracking proxy reached 0.42.",
        correction="Keep knees aligned with toes.",
        evidence_frame=12,
    )


def test_slugify_user_name_creates_stable_profile_id() -> None:
    assert slugify_user_name(" Ranbir Kumar 26 ") == "ranbir_kumar_26"
    assert slugify_user_name(" !!! ") == "guest_user"


def test_profile_from_inputs_uses_dashboard_defaults() -> None:
    profile = build_profile_from_inputs("Asha", "beginner", 170, "old knee pain\n")
    assert profile.user_id == "asha"
    assert profile.skill_level == "beginner"
    assert profile.height_cm == 170
    assert profile.injury_notes == ["old knee pain"]
    assert profile.baseline_angles["knee_min"] == 136


def test_score_labels_and_form_summary_bands() -> None:
    assert score_label(0.34) == "low"
    assert score_label(0.35) == "medium"
    assert score_label(0.65) == "high"
    assert form_summary_label(0, 0) == "no reps detected"
    assert form_summary_label(10, 0) == "good form"
    assert form_summary_label(10, 3) == "mostly good with small corrections"
    assert form_summary_label(10, 6) == "needs correction"
    assert form_summary_label(10, 7) == "poor form pattern"


def test_dashboard_button_state_helpers() -> None:
    assert can_run_analysis("Sam", has_video=True, use_sample_video=False)
    assert can_run_analysis("Sam", has_video=False, use_sample_video=True)
    assert not can_run_analysis("", has_video=True, use_sample_video=False)
    assert not can_run_analysis("Sam", has_video=False, use_sample_video=False)
    assert can_show_results(True)
    assert not can_show_results(False)


def test_user_summary_reads_like_trainer_feedback() -> None:
    features = [_rep(1, "knee_tracking", risk=0.7), _rep(2, "none", fatigue=0.7)]
    outputs = [_output(1), _output(2, "recommend_rest")]
    summary = build_user_summary(features, outputs)
    assert summary.detected_reps == 2
    assert summary.bad_form_reps == 1
    assert summary.main_issue == "knee_tracking"
    assert summary.rest_reps == [2]
    assert "2 squats detected" in summary.trainer_message


def test_technical_explanation_includes_threshold_and_policy_trace() -> None:
    class UnsafePolicy:
        def predict_action(self, state: list[float], rep: RepFeature) -> int:
            return action_id("no_feedback")

    profile = UserProfile(
        user_id="test",
        skill_level="beginner",
        height_cm=170,
        baseline_angles={"knee_min": 136, "hip_min": 148, "torso_lean_max": 24},
    )
    rep = _rep(1, "knee_tracking", risk=0.8)
    output = _output(1)
    explanations = build_technical_explanations([rep], [output], profile, SafetyPolicy(UnsafePolicy()))

    assert triggered_condition(rep, profile) == "knee_tracking_proxy 0.42 > 0.35"
    assert explanations[0].raw_model_action == "no_feedback"
    assert explanations[0].final_action == "joint_highlight"
    assert explanations[0].decision_source == "safety_policy"
    assert "injury_risk" in explanations[0].decision_reason
