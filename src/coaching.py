from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Protocol

from src.actions import ACTION_ADVICE, ACTION_NAMES, action_name
from src.contracts import CoachingOutput, RepFeature
from src.profiles import UserProfile


class Policy(Protocol):
    def predict_action(self, state: list[float], rep: RepFeature) -> int:
        ...


class HeuristicPolicy:
    def predict_action(self, state: list[float], rep: RepFeature) -> int:
        if rep.fatigue_score >= 0.65:
            return ACTION_NAMES.index("recommend_rest")
        if rep.injury_risk >= 0.55:
            return ACTION_NAMES.index("joint_highlight")
        if rep.mistake_label == "unstable_motion":
            return ACTION_NAMES.index("slow_tempo")
        if rep.mistake_label == "shallow_squat":
            return ACTION_NAMES.index("adjust_difficulty")
        if rep.mistake_label in {"forward_lean", "knee_tracking"}:
            return ACTION_NAMES.index("joint_highlight")
        return ACTION_NAMES.index("no_feedback")


class StableBaselinesPolicy:
    def __init__(self, model_path: Path) -> None:
        from stable_baselines3 import DQN  # type: ignore[import-not-found]

        self.model = DQN.load(str(model_path))

    def predict_action(self, state: list[float], rep: RepFeature) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)


class SafetyPolicy:
    def __init__(self, base_policy: Policy) -> None:
        self.base_policy = base_policy
        self.heuristic = HeuristicPolicy()

    def predict_action(self, state: list[float], rep: RepFeature) -> int:
        if (
            rep.fatigue_score >= 0.65
            or rep.injury_risk >= 0.45
            or rep.mistake_label in {"knee_tracking", "forward_lean", "unstable_motion", "shallow_squat"}
        ):
            return self.heuristic.predict_action(state, rep)
        return self.base_policy.predict_action(state, rep)


def load_policy(model_path: Path) -> Policy:
    zip_path = model_path.with_suffix(".zip")
    if zip_path.exists():
        try:
            return SafetyPolicy(StableBaselinesPolicy(zip_path))
        except Exception:
            return HeuristicPolicy()
    return HeuristicPolicy()


def state_vector(rep: RepFeature, profile: UserProfile, previous_action_id: int, repeated_mistakes: int) -> list[float]:
    return [
        rep.knee_angle_min / 180,
        rep.hip_angle_min / 180,
        rep.torso_lean_max / 90,
        min(1.0, rep.rep_id / 20),
        rep.fatigue_score,
        rep.injury_risk,
        min(1.0, repeated_mistakes / 5),
        previous_action_id / max(1, len(ACTION_NAMES) - 1),
        profile.skill_encoding,
    ]


def reward_for_transition(current: RepFeature, next_rep: RepFeature | None, action: str, repeated_mistakes: int) -> float:
    risk_penalty = current.injury_risk * 0.45
    fatigue_penalty = current.fatigue_score * 0.20
    repeat_penalty = min(0.5, repeated_mistakes * 0.12)
    correction_penalty = 0.08 if action != "no_feedback" else 0.0
    improvement_reward = 0.0
    if next_rep is not None:
        current_quality = current.depth_proxy - current.injury_risk - current.fatigue_score * 0.3
        next_quality = next_rep.depth_proxy - next_rep.injury_risk - next_rep.fatigue_score * 0.3
        improvement_reward = max(-0.5, min(0.8, next_quality - current_quality))
    if action == "recommend_rest" and current.fatigue_score > 0.65:
        improvement_reward += 0.25
    if action == "joint_highlight" and current.injury_risk > 0.50:
        improvement_reward += 0.20
    if action == "no_feedback" and current.mistake_label == "none":
        improvement_reward += 0.18
    return improvement_reward - risk_penalty - fatigue_penalty - repeat_penalty - correction_penalty


def _specific_explanation(rep: RepFeature, action: str) -> tuple[str, str, str]:
    if rep.mistake_label == "knee_tracking":
        return (
            "Knee tracking issue",
            f"Knee tracking proxy reached {rep.knee_tracking_proxy:.2f}, raising injury risk.",
            "Keep knees aligned with toes during descent and ascent.",
        )
    if rep.mistake_label == "forward_lean":
        return (
            "Excessive forward lean",
            f"Torso lean peaked at {rep.torso_lean_max:.1f} degrees.",
            "Brace the trunk and keep chest position more stable.",
        )
    if rep.mistake_label == "shallow_squat":
        return (
            "Reduced squat depth",
            f"Minimum knee angle stayed at {rep.knee_angle_min:.1f} degrees.",
            "Use controlled depth that matches the safe baseline.",
        )
    if rep.fatigue_score > 0.65:
        return (
            "Fatigue signal",
            f"Fatigue score reached {rep.fatigue_score:.2f}.",
            "Rest before continuing the set.",
        )
    advice = ACTION_ADVICE[action]
    return advice.problem, advice.reason, advice.correction


def make_coaching_outputs(features: list[RepFeature], profile: UserProfile, policy: Policy | None = None) -> list[CoachingOutput]:
    policy = policy or HeuristicPolicy()
    outputs: list[CoachingOutput] = []
    previous_action = ACTION_NAMES.index("no_feedback")
    mistake_counter: Counter[str] = Counter()
    for index, rep in enumerate(features):
        if rep.mistake_label != "none":
            mistake_counter[rep.mistake_label] += 1
        repeated = mistake_counter[rep.mistake_label] if rep.mistake_label != "none" else 0
        state = state_vector(rep, profile, previous_action, repeated)
        predicted = policy.predict_action(state, rep)
        action = action_name(predicted)
        next_rep = features[index + 1] if index + 1 < len(features) else None
        reward = reward_for_transition(rep, next_rep, action, repeated)
        problem, reason, correction = _specific_explanation(rep, action)
        outputs.append(
            CoachingOutput(
                rep_id=rep.rep_id,
                state=state,
                action=action,
                reward=reward,
                problem=problem,
                reason=reason,
                correction=correction,
                evidence_frame=rep.evidence_frame,
            )
        )
        previous_action = predicted
    return outputs


def write_policy_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "policy": "heuristic_fallback",
                "actions": list(ACTION_NAMES),
                "note": "Stable-Baselines3 DQN writes coach_policy.zip when installed.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
