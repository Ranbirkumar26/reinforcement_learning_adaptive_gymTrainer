from __future__ import annotations

"""Dashboard summary helpers kept outside Streamlit for deterministic tests."""

import re
from collections import Counter
from dataclasses import dataclass

from src.actions import ACTION_NAMES, action_id
from src.coaching import Policy, explain_policy_decision, state_vector
from src.contracts import CoachingOutput, RepFeature
from src.profiles import UserProfile


DEFAULT_BASELINE_ANGLES = {"knee_min": 136, "hip_min": 148, "torso_lean_max": 24}


@dataclass(frozen=True)
class UserSummary:
    detected_reps: int
    bad_form_reps: int
    form_label: str
    fatigue_label: str
    injury_risk_label: str
    main_issue: str
    rest_reps: list[int]
    trainer_message: str


@dataclass(frozen=True)
class TechnicalRepExplanation:
    rep_id: int
    triggered_condition: str
    state_vector: list[float]
    raw_model_action: str
    final_action: str
    decision_source: str
    decision_reason: str
    reward: float
    evidence_frame: int


def slugify_user_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "guest_user"


def build_profile_from_inputs(
    name: str,
    skill_level: str = "beginner",
    height_cm: float = 170,
    injury_notes_text: str = "",
) -> UserProfile:
    notes = [line.strip() for line in injury_notes_text.splitlines() if line.strip()]
    return UserProfile(
        user_id=slugify_user_name(name),
        skill_level=skill_level,  # type: ignore[arg-type]
        height_cm=height_cm,
        injury_notes=notes,
        baseline_angles=dict(DEFAULT_BASELINE_ANGLES),
        history={"sessions": 0, "recurring_mistakes": {}},
    )


def score_label(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


def form_summary_label(total_reps: int, bad_form_reps: int) -> str:
    if total_reps <= 0:
        return "no reps detected"
    if bad_form_reps == 0:
        return "good form"
    ratio = bad_form_reps / total_reps
    if ratio <= 0.30:
        return "mostly good with small corrections"
    if ratio <= 0.60:
        return "needs correction"
    return "poor form pattern"


def can_run_analysis(name: str, has_video: bool, use_sample_video: bool) -> bool:
    return bool(name.strip()) and (has_video or use_sample_video)


def can_show_results(analysis_complete: bool) -> bool:
    return analysis_complete


def build_user_summary(features: list[RepFeature], outputs: list[CoachingOutput]) -> UserSummary:
    total = len(features)
    bad = sum(1 for rep in features if rep.mistake_label != "none")
    mistakes = Counter(rep.mistake_label for rep in features if rep.mistake_label != "none")
    main_issue = mistakes.most_common(1)[0][0] if mistakes else "none"
    avg_fatigue = sum(rep.fatigue_score for rep in features) / total if total else 0.0
    avg_risk = sum(rep.injury_risk for rep in features) / total if total else 0.0
    rest_reps = [output.rep_id for output in outputs if output.action == "recommend_rest"]
    form_label = form_summary_label(total, bad)
    fatigue_label = score_label(avg_fatigue)
    injury_label = score_label(avg_risk)

    if total == 0:
        trainer_message = "No squat reps were detected. Record full-body side view squats and try again."
    elif main_issue == "none":
        trainer_message = f"{total} squats detected. Form looked controlled overall. Fatigue was {fatigue_label}."
    else:
        issue_text = main_issue.replace("_", " ")
        rest_text = ""
        if rest_reps:
            reps = ", ".join(str(rep_id) for rep_id in rest_reps)
            rest_text = f" Rest was recommended on reps {reps} because fatigue or risk increased."
        trainer_message = (
            f"{total} squats detected. {bad} reps need attention, so overall form is {form_label}. "
            f"Main issue was {issue_text}. Keep knees aligned with toes and move with control.{rest_text}"
        )

    return UserSummary(
        detected_reps=total,
        bad_form_reps=bad,
        form_label=form_label,
        fatigue_label=fatigue_label,
        injury_risk_label=injury_label,
        main_issue=main_issue,
        rest_reps=rest_reps,
        trainer_message=trainer_message,
    )


def triggered_condition(rep: RepFeature, profile: UserProfile) -> str:
    baseline_knee = profile.baseline_angles.get("knee_min", DEFAULT_BASELINE_ANGLES["knee_min"])
    baseline_torso = profile.baseline_angles.get("torso_lean_max", DEFAULT_BASELINE_ANGLES["torso_lean_max"])
    if rep.mistake_label == "knee_tracking":
        return f"knee_tracking_proxy {rep.knee_tracking_proxy:.2f} > 0.35"
    if rep.mistake_label == "forward_lean":
        return f"torso_lean_max {rep.torso_lean_max:.1f} > baseline {baseline_torso:.1f} + 12"
    if rep.mistake_label == "shallow_squat":
        return f"knee_angle_min {rep.knee_angle_min:.1f} > baseline {baseline_knee:.1f} + 18"
    if rep.mistake_label == "unstable_motion":
        return f"smoothness {rep.smoothness:.2f} > 4.00"
    if rep.fatigue_score >= 0.65:
        return f"fatigue_score {rep.fatigue_score:.2f} >= 0.65"
    return "No mistake threshold triggered"


def build_technical_explanations(
    features: list[RepFeature],
    outputs: list[CoachingOutput],
    profile: UserProfile,
    policy: Policy,
) -> list[TechnicalRepExplanation]:
    previous_action = action_id("no_feedback")
    repeated_counts: Counter[str] = Counter()
    explanations: list[TechnicalRepExplanation] = []
    output_by_rep = {output.rep_id: output for output in outputs}

    for rep in features:
        if rep.mistake_label != "none":
            repeated_counts[rep.mistake_label] += 1
        repeated = repeated_counts[rep.mistake_label] if rep.mistake_label != "none" else 0
        state = state_vector(rep, profile, previous_action, repeated)
        decision = explain_policy_decision(policy, state, rep)
        output = output_by_rep[rep.rep_id]
        explanations.append(
            TechnicalRepExplanation(
                rep_id=rep.rep_id,
                triggered_condition=triggered_condition(rep, profile),
                state_vector=state,
                raw_model_action=decision.raw_action,
                final_action=output.action,
                decision_source=decision.decision_source,
                decision_reason=decision.reason,
                reward=output.reward,
                evidence_frame=output.evidence_frame,
            )
        )
        previous_action = ACTION_NAMES.index(output.action)
    return explanations
