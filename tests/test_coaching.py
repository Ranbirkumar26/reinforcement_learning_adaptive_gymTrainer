from __future__ import annotations

from src.actions import ACTION_NAMES, action_id, action_name
from src.coaching import HeuristicPolicy, SafetyPolicy, make_coaching_outputs, reward_for_transition
from src.contracts import RepFeature
from src.profiles import UserProfile


def _rep(rep_id: int, fatigue: float, risk: float, mistake: str = "none") -> RepFeature:
    return RepFeature(
        rep_id=rep_id,
        start_frame=0,
        end_frame=30,
        knee_angle_min=90,
        hip_angle_min=85,
        torso_lean_max=25,
        tempo_sec=1.0,
        smoothness=1.0,
        fatigue_score=fatigue,
        injury_risk=risk,
        mistake_label=mistake,
        knee_tracking_proxy=0.2,
        depth_proxy=0.8,
        evidence_frame=12,
    )


def test_action_mapping_round_trip() -> None:
    for action in ACTION_NAMES:
        assert action_name(action_id(action)) == action


def test_heuristic_policy_recommends_rest_for_high_fatigue() -> None:
    rep = _rep(1, fatigue=0.8, risk=0.2)
    action = action_name(HeuristicPolicy().predict_action([0] * 9, rep))
    assert action == "recommend_rest"


def test_reward_penalizes_high_risk_without_matching_action() -> None:
    current = _rep(1, fatigue=0.2, risk=0.9, mistake="knee_tracking")
    reward = reward_for_transition(current, None, "verbal_cue", repeated_mistakes=2)
    assert reward < 0


def test_make_coaching_outputs_has_explanations() -> None:
    profile = UserProfile(user_id="test", skill_level="beginner", height_cm=170)
    outputs = make_coaching_outputs([_rep(1, 0.1, 0.1)], profile)
    assert outputs[0].action == "no_feedback"
    assert outputs[0].reason
    assert outputs[0].correction


def test_safety_policy_overrides_unsafe_no_feedback() -> None:
    class UnsafePolicy:
        def predict_action(self, state: list[float], rep: RepFeature) -> int:
            return action_id("no_feedback")

    rep = _rep(1, fatigue=0.2, risk=0.8, mistake="knee_tracking")
    action = action_name(SafetyPolicy(UnsafePolicy()).predict_action([0] * 9, rep))

    assert action == "joint_highlight"
