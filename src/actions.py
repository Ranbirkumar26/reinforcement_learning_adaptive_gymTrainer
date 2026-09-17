from __future__ import annotations

from dataclasses import dataclass


ACTION_NAMES: tuple[str, ...] = (
    "verbal_cue",
    "joint_highlight",
    "slow_tempo",
    "adjust_difficulty",
    "recommend_rest",
    "no_feedback",
    "demonstration",
    "breathing_cue",
)


@dataclass(frozen=True)
class ActionAdvice:
    problem: str
    reason: str
    correction: str


ACTION_ADVICE: dict[str, ActionAdvice] = {
    "verbal_cue": ActionAdvice(
        problem="Technique drift detected",
        reason="The rep moved outside the user baseline but risk stayed moderate.",
        correction="Keep chest lifted and sit into the squat with control.",
    ),
    "joint_highlight": ActionAdvice(
        problem="Joint tracking issue",
        reason="Knee or torso metrics contributed most to the current risk score.",
        correction="Watch the highlighted joint and align it with the foot path.",
    ),
    "slow_tempo": ActionAdvice(
        problem="Unstable tempo",
        reason="Movement smoothness decreased across the rep.",
        correction="Slow the next rep and pause briefly near the bottom position.",
    ),
    "adjust_difficulty": ActionAdvice(
        problem="Range target missed repeatedly",
        reason="The current difficulty does not match the user's baseline control.",
        correction="Reduce depth or load until reps match the safe baseline.",
    ),
    "recommend_rest": ActionAdvice(
        problem="Fatigue is high",
        reason="Tempo slowed and range of motion decreased compared with earlier reps.",
        correction="Rest before the next set to avoid reinforcing poor mechanics.",
    ),
    "no_feedback": ActionAdvice(
        problem="No major issue",
        reason="The rep stayed within baseline range and risk stayed low.",
        correction="Continue the same movement pattern.",
    ),
    "demonstration": ActionAdvice(
        problem="Repeated form error",
        reason="The same mistake persisted after prior feedback.",
        correction="Review the target squat pattern before continuing.",
    ),
    "breathing_cue": ActionAdvice(
        problem="Brace timing issue",
        reason="Torso control degraded while lower-body angles stayed usable.",
        correction="Inhale before descent, brace through the bottom, then exhale upward.",
    ),
}


def action_id(action: str) -> int:
    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown coaching action: {action}")
    return ACTION_NAMES.index(action)


def action_name(index: int) -> str:
    if index < 0 or index >= len(ACTION_NAMES):
        raise ValueError(f"Unknown coaching action index: {index}")
    return ACTION_NAMES[index]
